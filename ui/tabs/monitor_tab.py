"""ui/tabs/monitor_tab.py — Training monitor inside the WebUI console.

Reads the same run directories as the standalone monitor page
(``chronos monitor``) through :class:`chronos.monitor.RunMonitor`: run spec
and training parameters, heartbeat/derived status, progress and ETA, every
logged metric, validation events, whatever evaluation metrics the run logs
(discovered, never assumed), per-tensor statistics, MoE routing, samples of
any task, log tail and checkpoints.  Read-only; it never touches the trainer.
"""
from __future__ import annotations

import html
import json
import math
import os
import threading
import time

from ui.gradio_compat import gr
from ui.i18n import get_current_lang, register_translatable, t

try:  # pandas ships with gradio; the slim CI shim may lack it
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None

DEFAULT_RUNS_DIR = os.environ.get("CHRONOS_RUNS_DIR", "runs")
_MONITORS: dict[str, object] = {}
_LOCK = threading.Lock()

# Line plots: (title key, [(series key, label key)...]); one y-axis per plot.
PLOTS = {
    "overview": [
        ("monitor.p_ce", [("primary_loss", "monitor.s_train"), ("valid_ce", "monitor.s_valid")]),
        ("monitor.p_loss", [("loss", "monitor.s_total")]),
        ("monitor.p_lr", [("lr", "monitor.s_lr")]),
        ("monitor.p_grad", [("grad_norm", "monitor.s_grad")]),
        ("monitor.p_tps", [("tokens_per_s", "monitor.s_counted"), ("processed_tokens_per_s", "monitor.s_processed")]),
        ("monitor.p_mem", [("peak_mem_gb", "monitor.s_peak"), ("active_mem_gb", "monitor.s_active"),
                           ("cache_mem_gb", "monitor.s_cache")]),
    ],
    "recurrent": [
        ("monitor.p_depth", [("rec_depth_mean", "monitor.s_depth")]),
        ("monitor.p_contraction", [("rec_contraction", "monitor.s_contraction")]),
        ("monitor.p_exit", [("rec_early_exit_ratio", "monitor.s_exit")]),
        ("monitor.p_route_entropy", [("route_entropy", "monitor.s_entropy")]),
        ("monitor.p_load_cv", [("load_cv", "monitor.s_load_cv")]),
        ("monitor.p_max_load", [("max_expert_load", "monitor.s_max_load")]),
    ],
}
LOSS_TERMS = ("primary_loss", "balance_loss", "lookahead_loss", "lookahead_topk_loss", "lookahead_union_loss",
              "temporal_loss", "locality_loss", "anchor_loss", "recurrent_ponder_loss", "recurrent_distill_loss",
              "recurrent_stability_loss", "recurrent_contraction_loss")
STATE_ICON = {"running": "●", "paused": "‖", "evaluating": "◐", "checkpointing": "◐", "starting": "○",
              "finished": "✓", "failed": "✕", "stopped": "■", "pending": "○"}
# Sequential blue ramp (light→dark), steps 150..700 of the reference palette.
SEQ = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
SEQ_INK = ("#0b0b0b", "#0b0b0b", "#0b0b0b", "#ffffff", "#ffffff", "#ffffff", "#ffffff")

MONITOR_CSS = """
.chronos-mon{font-family:system-ui,-apple-system,"Segoe UI","PingFang SC",sans-serif;font-size:13px}
.chronos-mon .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:10px}
.chronos-mon .pill{display:inline-flex;gap:6px;align-items:center;padding:3px 10px;border-radius:999px;
  border:1px solid var(--border-color-primary);background:var(--background-fill-secondary);font-size:12px}
.chronos-mon .muted{color:var(--body-text-color-subdued)}
.chronos-mon .bar{height:8px;border-radius:4px;background:var(--background-fill-secondary);overflow:hidden;margin:6px 0 4px}
.chronos-mon .bar>span{display:block;height:100%;background:#2a78d6;border-radius:4px}
.chronos-mon .tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.chronos-mon .tile{border:1px solid var(--border-color-primary);border-radius:8px;padding:10px 12px;background:var(--block-background-fill)}
.chronos-mon .tile b{display:block;font-size:20px;font-weight:600;margin-top:4px;font-variant-numeric:tabular-nums}
.chronos-mon .tile small{color:var(--body-text-color-subdued);font-size:11px}
.chronos-mon .note{border:1px solid var(--border-color-primary);border-radius:8px;padding:8px 12px;margin-bottom:10px}
.chronos-mon .warn{border-color:#fab219}.chronos-mon .err{border-color:#d03b3b}
.chronos-mon table.heat{border-collapse:separate;border-spacing:2px;font-size:10px;width:auto;margin:6px 0 0}
.chronos-mon table.heat,.chronos-mon table.heat tr,.chronos-mon table.heat td,.chronos-mon table.heat th{border:0!important;background-clip:padding-box}
.chronos-mon table.heat td{min-width:16px;height:18px;border-radius:2px;text-align:center;padding:0 2px;font-variant-numeric:tabular-nums}
.chronos-mon table.heat th{font-weight:400;color:var(--body-text-color-subdued);padding:0 4px;white-space:nowrap}
.chronos-mon .samp{border-bottom:1px solid var(--border-color-primary);padding:8px 0;line-height:1.6}
"""


# ---- data access -------------------------------------------------------------
def _monitor(runs_dir: str):
    from chronos.monitor import RunMonitor
    root = os.path.abspath(os.path.expanduser(runs_dir or DEFAULT_RUNS_DIR))
    with _LOCK:
        mon = _MONITORS.get(root)
        if mon is None:
            if len(_MONITORS) > 8:
                _MONITORS.clear()
            mon = _MONITORS[root] = RunMonitor(root)
        return mon


def _f(v, digits: int = 4) -> str:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return "—"
    a = abs(v)
    if a and (a < 1e-4 or a >= 1e6):
        return f"{v:.2e}"
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:.{digits}g}"


def _count(v) -> str:
    if not isinstance(v, (int, float)):
        return "—"
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(v) >= div:
            return f"{v / div:.1f}{suf}"
    return str(round(v))


def _dur(sec) -> str:
    if not isinstance(sec, (int, float)):
        return "—"
    sec = max(0, int(sec))
    d, h, m = sec // 86400, sec % 86400 // 3600, sec % 3600 // 60
    return f"{d}d {h}h" if d else f"{h}h {m}m" if h else f"{m}m"


def _esc(v) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def _df(rows: list[dict], columns: list[tuple[str, str]], numeric: tuple[str, ...] = ()):
    """``columns``: (row key, i18n header key); ``numeric`` keys are pre-formatted (tiny values as 1.23e-06)."""
    data = [[_f(r.get(k)) if k in numeric and r.get(k) is not None else r.get(k) for k, _ in columns] for r in rows]
    if pd is None:
        return data
    return pd.DataFrame(data, columns=[t(h) for _, h in columns])


def _series_df(parts: list[tuple[str, list[dict], str]]):
    """Long format for gr.LinePlot: ``parts`` = (label, rows, metric key)."""
    recs = []
    for label, rows, key in parts:
        for r in rows:
            v, step = r.get(key), r.get("step")
            if isinstance(v, (int, float)) and isinstance(step, (int, float)) and math.isfinite(v):
                recs.append({"step": step, "value": float(v), "series": label})
    if pd is None:
        return recs
    return pd.DataFrame(recs, columns=["step", "value", "series"])


PALETTE = ("#2a78d6", "#eb6834", "#1baf7a")


def _plot_update(parts):
    """Fixed colour per series slot (blue, orange, green). Multi-series plots
    name their colours in the translated label; the tooltip carries the name."""
    labels = list(dict.fromkeys(label for label, _, _ in parts))
    return gr.update(value=_series_df(parts), color_map={lab: PALETTE[i % 3] for i, lab in enumerate(labels)})


# ---- HTML fragments (all dynamic text escaped) ----------------------------------
def _heat_html(title: str, labels: list, grid: list, fmt=lambda v: _f(v, 3), col_label="E") -> str:
    flat = [v for row in grid or [] for v in row if isinstance(v, (int, float))]
    if not flat:
        return f"<div class='chronos-mon'><b>{_esc(title)}</b><p class='muted'>{_esc(t('monitor.none'))}</p></div>"
    lo, hi = min(flat), max(flat)
    cols = max(len(r) for r in grid)
    head = "".join(f"<th>{j}</th>" for j in range(cols))
    body = []
    for i, row in enumerate(grid):
        cells = []
        for j in range(cols):
            v = row[j] if j < len(row) else None
            if not isinstance(v, (int, float)):
                cells.append("<td></td>")
                continue
            b = min(6, int((v - lo) / (hi - lo) * 7)) if hi > lo else 3
            tip = f"{labels[i] if i < len(labels) else i} · {col_label}{j}: {fmt(v)}"
            cells.append(f"<td title='{_esc(tip)}' style='background:{SEQ[b]};color:{SEQ_INK[b]}'></td>")
        body.append(f"<tr><th>{_esc(labels[i] if i < len(labels) else i)}</th>{''.join(cells)}</tr>")
    ramp = "".join(f"<span style='display:inline-block;width:18px;height:8px;background:{c}'></span>" for c in SEQ)
    return (f"<div class='chronos-mon'><b>{_esc(title)}</b><table class='heat'><tr><th></th>{head}</tr>{''.join(body)}</table>"
            f"<div class='muted'>{_esc(fmt(lo))} {ramp} {_esc(fmt(hi))}</div></div>")


def _status_html(snap: dict) -> str:
    if not snap.get("run"):
        return f"<div class='chronos-mon'><div class='note'>{_esc(t('monitor.no_runs'))} <code>{_esc(snap.get('root'))}</code></div></div>"
    st, p, latest = snap.get("status") or {}, snap.get("progress") or {}, snap.get("latest") or {}
    state = st.get("state", "pending")
    notes = []
    if st.get("error"):
        notes.append(f"<div class='note err'>✕ {_esc(t('monitor.error'))}: {_esc(st['error'])}</div>")
    if state == "paused":
        notes.append(f"<div class='note warn'>‖ {_esc(t('monitor.paused'))}</div>")
    if st.get("stale"):
        notes.append(f"<div class='note warn'>! {_esc(t('monitor.stale').format(age=_dur(st.get('last_update_age_s'))))}</div>")
    if st.get("source") == "derived" and not snap.get("parameters"):
        notes.append(f"<div class='note'>i {_esc(t('monitor.derived'))}</div>")
    frac = p.get("phase_fraction")
    head = (f"<div class='row'><span class='pill'>{STATE_ICON.get(state, '○')} {_esc(t('monitor.state.' + state))}</span>"
            f"<b>{_esc(snap['run'])}</b><span>{_esc(p.get('current') or '—')} · {_esc(t('monitor.step'))} "
            f"{_esc(_f(latest.get('step')))} / {_esc(_f(latest.get('total')))}"
            f"{' · ' + format(frac * 100, '.1f') + '%' if isinstance(frac, (int, float)) else ''}</span>"
            f"<span class='muted'>{_esc(t('monitor.eta'))} {_esc((p.get('eta') or '—').replace('T', ' '))} · "
            f"{_esc(t('monitor.remaining'))} {_esc(_dur(p.get('run_remaining_s')))} · {_esc(t('monitor.budget'))} "
            f"{_esc(p.get('budget_hours'))} h</span></div>")
    bars = []
    for ph in p.get("phases") or []:
        f = 1.0 if ph["state"] == "complete" else (frac or 0.0) if ph["state"] == "active" else 0.0
        bars.append(f"<div style='flex:{max(0.05, ph.get('budget_h') or 1)}'><div class='bar'><span style='width:{f * 100:.2f}%'></span></div>"
                    f"<small><b>{_esc(ph['name'])}</b> {_esc(t('monitor.ph.' + ph['state']))} · {_esc(ph.get('budget_h'))} h</small></div>")
    best = (snap.get("best_valid") or {}).get(p.get("current")) or {}
    mem = snap.get("memory") or {}
    spec = snap.get("spec") or {}
    cfg = spec.get("model_config") or {}
    tiles = [
        ("monitor.k_loss", _f(latest.get("loss")), f"{t('monitor.step')} {_f(latest.get('step'))}"),
        ("monitor.k_ce", _f(latest.get("primary_loss")),
         f"ppl {_f(math.exp(latest['primary_loss']), 3)}" if isinstance(latest.get("primary_loss"), (int, float)) else ""),
        ("monitor.k_valid", _f(best.get("valid_ce")), f"{t('monitor.step')} {_f(best.get('step'))}" if best else ""),
        ("monitor.k_tps", f"{_count(latest.get('tokens_per_s'))} tok/s", f"{_count(latest.get('processed_tokens_per_s'))} {t('monitor.s_processed')}"),
        ("monitor.k_mem", f"{_f(mem.get('footprint_gb') or latest.get('peak_mem_gb'), 3)} GB",
         f"MLX peak {_f(latest.get('peak_mem_gb'), 3)} · active {_f(latest.get('active_mem_gb'), 3)} GB"),
        ("monitor.k_grad", _f(latest.get("grad_norm")), ""),
        ("monitor.k_lr", _f(latest.get("lr"), 3), ""),
        ("monitor.k_depth", _f(latest.get("rec_depth_mean"), 3), f"max {_esc(cfg.get('recurrent_depth', '—'))}"),
        ("monitor.k_route", _f(latest.get("route_entropy"), 3), f"max {_f(latest.get('route_entropy_max'), 3)} · CV {_f(latest.get('load_cv'), 3)}"),
    ]
    tiles_html = "".join(f"<div class='tile'><small>{_esc(t(k))}</small><b>{_esc(v)}</b><small>{_esc(sub)}</small></div>" for k, v, sub in tiles)
    foot = (f"<p class='muted'>{_esc(t('monitor.source'))}: {_esc(st.get('source'))} · pid {_esc(st.get('pid'))} "
            f"({_esc(st.get('ps_state') or '-')}) · {_esc(time.strftime('%H:%M:%S'))}</p>")
    return (f"<div class='chronos-mon'>{''.join(notes)}{head}<div class='row' style='gap:4px;flex-wrap:nowrap'>{''.join(bars)}</div>"
            f"<div class='tiles'>{tiles_html}</div>{foot}</div>")


def _events_rows(snap: dict) -> list[dict]:
    out = []
    for e in reversed(snap.get("events") or []):
        detail = {k: v for k, v in e.items() if k not in ("time", "event", "phase", "step")}
        out.append({"time": e.get("time"), "event": e.get("event"), "phase": e.get("phase"), "step": e.get("step"),
                    "valid_ce": e.get("valid_ce"), "detail": json.dumps(detail, ensure_ascii=False)[:400]})
    return out


def _samples_html(snap: dict) -> str:
    """Samples of any task: the snapshot normalises them to input/reference/output + tags."""
    rows = list(reversed(snap.get("samples") or []))[:12]
    if not rows:
        return f"<div class='chronos-mon muted'>{_esc(t('monitor.none'))}</div>"
    items = []
    for r in rows:
        tags = " · ".join(f"{k}={v}" for k, v in (r.get("tags") or {}).items())
        head = " · ".join(x for x in (_esc(r.get("phase")), f"{_esc(t('monitor.step'))} {_esc(r.get('step'))}", _esc(tags)) if x)
        body = "".join(f"<br><b>{_esc(t(key))}</b> {_esc(r.get(role))}"
                       for role, key in (("input", "monitor.sample_input"), ("reference", "monitor.sample_ref"),
                                         ("output", "monitor.sample_output")) if r.get(role) not in (None, ""))
        items.append(f"<div class='samp'><small class='muted'>{head}</small>{body}</div>")
    return f"<div class='chronos-mon'>{''.join(items)}</div>"


def _eval_family(snap: dict, metric: str | None) -> tuple[list[str], str | None, dict | None]:
    """Metric families the run logs (discovered, not assumed) and the chosen one."""
    families = (snap.get("evals") or {}).get("families") or []
    names = [f["metric"] for f in families]
    chosen = metric if metric in names else (names[0] if names else None)
    return names, chosen, next((f for f in families if f["metric"] == chosen), None)


# ---- refresh -------------------------------------------------------------------------
def _phase_rows(snap: dict, phase: str | None) -> tuple[str | None, list[dict]]:
    hist = snap.get("history") or {}
    phase = phase if phase in hist else (snap.get("progress") or {}).get("current")
    if phase not in hist:
        phase = next(reversed(hist), None) if hist else None
    return phase, hist.get(phase, []) if phase else []


def _loss_parts(rows: list[dict], term: str):
    term = term if term in LOSS_TERMS else "primary_loss"
    return [(term, rows, term)]


def refresh(runs_dir: str, run: str | None, phase: str | None, loss_term: str, eval_metric: str | None = None):
    """Everything the tab shows, in output order (see ``build_monitor_tab``)."""
    mon = _monitor(runs_dir)
    try:
        snap = mon.snapshot(run or None)
    except KeyError:
        snap = mon.snapshot(None)
    names = [r["name"] for r in snap.get("runs") or []]
    run_update = gr.update(choices=names, value=snap.get("run"))
    phases = list((snap.get("history") or {}).keys())
    phase, rows = _phase_rows(snap, phase)
    phase_update = gr.update(choices=phases, value=phase)
    valid = [e for e in snap.get("events") or [] if e.get("event") == "validation" and e.get("phase") == phase]
    plots = []
    for group in ("overview", "recurrent"):
        for _title, series in PLOTS[group]:
            parts = [(t(label), valid if key == "valid_ce" else rows, key) for key, label in series]
            plots.append(_plot_update(parts))
    present = [k for k in LOSS_TERMS if any(isinstance(r.get(k), (int, float)) for r in rows)] or ["primary_loss"]
    term = loss_term if loss_term in present else present[0]
    loss_update = _plot_update(_loss_parts(rows, term))
    term_update = gr.update(choices=present, value=term)

    params = snap.get("parameters") or {}
    routing = params.get("routing") or {}
    experts = params.get("experts") or {}
    groups = [{"group": k, **v} for k, v in (params.get("groups") or {}).items()]
    if params.get("stats"):
        groups.append({"group": "total", **params["stats"], "tensors": params.get("tensor_count")})
    group_df = _df(groups, [("group", "monitor.c_group"), ("tensors", "monitor.c_tensors"), ("numel", "monitor.c_numel"),
                            ("rms", "monitor.c_rms"), ("grad_norm", "monitor.c_grad_norm"), ("update_ratio", "monitor.c_update")],
                   numeric=("rms", "grad_norm", "update_ratio"))
    full = mon.parameters(snap.get("run")) if params else None
    tensors = (full or {}).get("tensors") or []
    tensor_df = _df([{**x, "shape": "×".join(map(str, x.get("shape") or []))} for x in tensors],
                    [("name", "monitor.c_name"), ("group", "monitor.c_group"), ("shape", "monitor.c_shape"), ("dtype", "monitor.c_dtype"),
                     ("rms", "monitor.c_rms"), ("std", "monitor.c_std"), ("absmax", "monitor.c_absmax"),
                     ("grad_rms", "monitor.c_grad_rms"), ("update_ratio", "monitor.c_update")],
                    numeric=("rms", "std", "absmax", "grad_rms", "update_ratio"))
    tel_note = (t("monitor.telemetry_at").format(step=params.get("step"), ms=round((params.get("telemetry_s") or 0) * 1000))
                if params else t("monitor.no_telemetry"))
    load_html = _heat_html(t("monitor.h_load"), routing.get("labels") or [], routing.get("load") or [],
                           fmt=lambda v: f"{v * 100:.1f}%")
    imp_html = _heat_html(t("monitor.h_importance"), routing.get("labels") or [], routing.get("importance") or [])
    layers = [f"L{x}" for x in experts.get("layers") or []]
    wrms_html = _heat_html(t("monitor.h_wrms"), layers, experts.get("weight_rms") or [])
    grms_html = _heat_html(t("monitor.h_grms"), layers, experts.get("grad_rms") or [])

    events_df = _df(_events_rows(snap), [("time", "monitor.c_time"), ("event", "monitor.c_event"), ("phase", "monitor.c_phase"),
                                         ("step", "monitor.step"), ("valid_ce", "monitor.c_valid_ce"), ("detail", "monitor.c_detail")])
    ckpt_df = _df(snap.get("checkpoints") or [], [("file", "monitor.c_file"), ("size_gb", "monitor.c_size"),
                                                   ("modified", "monitor.c_time")])
    names, chosen, family = _eval_family(snap, eval_metric)
    eval_parts = [(sr["label"], valid, sr["key"]) for sr in (family or {}).get("series", [])[:len(PALETTE)]]
    eval_update = gr.update(choices=names, value=chosen)
    eval_plot = _plot_update(eval_parts)
    training_params = mon.training_parameters(snap.get("run")) if snap.get("run") else {}
    log = mon.log_tail(snap.get("run"), 24000) if snap.get("run") else ""
    return (run_update, phase_update, _status_html(snap), *plots, term_update, loss_update,
            tel_note, group_df, tensor_df, load_html, imp_html, wrms_html, grms_html,
            eval_update, eval_plot, events_df, _samples_html(snap), training_params, log, ckpt_df)


# ---- layout ------------------------------------------------------------------------------
def _labelled(component, key: str):
    register_translatable(component, key)
    return component


def build_monitor_tab():
    with gr.Tab(t("tab.monitor")) as tab:
        register_translatable(tab, "tab.monitor")
        gr.Markdown(t("monitor.intro"))
        with gr.Row():
            runs_dir = _labelled(gr.Textbox(value=DEFAULT_RUNS_DIR, label=t("monitor.runs_dir"), scale=3), "monitor.runs_dir")
            run = _labelled(gr.Dropdown(choices=[], value=None, label=t("monitor.run"), scale=2, allow_custom_value=True), "monitor.run")
            phase = _labelled(gr.Dropdown(choices=[], value=None, label=t("monitor.phase"), scale=1, allow_custom_value=True), "monitor.phase")
            auto = _labelled(gr.Checkbox(value=True, label=t("monitor.auto"), scale=1), "monitor.auto")
            refresh_btn = _labelled(gr.Button(t("monitor.refresh"), scale=1), "monitor.refresh")
        status = gr.HTML()
        plots = []
        with gr.Tab(t("monitor.t_overview")) as sub:
            register_translatable(sub, "monitor.t_overview")
            for i in range(0, len(PLOTS["overview"]), 2):
                with gr.Row():
                    for title, _series in PLOTS["overview"][i:i + 2]:
                        plots.append(_labelled(gr.LinePlot(x="step", y="value", color="series", label=t(title), y_title="",
                                                           height=300, tooltip=["series", "step", "value"]), title))
        with gr.Tab(t("monitor.t_recurrent")) as sub:
            register_translatable(sub, "monitor.t_recurrent")
            for i in range(0, len(PLOTS["recurrent"]), 3):
                with gr.Row():
                    for title, _series in PLOTS["recurrent"][i:i + 3]:
                        plots.append(_labelled(gr.LinePlot(x="step", y="value", color="series", label=t(title), y_title="",
                                                           height=270, tooltip=["series", "step", "value"]), title))
        with gr.Tab(t("monitor.t_loss")) as sub:
            register_translatable(sub, "monitor.t_loss")
            loss_term = _labelled(gr.Dropdown(choices=list(LOSS_TERMS), value="primary_loss", label=t("monitor.loss_term")), "monitor.loss_term")
            loss_plot = gr.LinePlot(x="step", y="value", color="series", y_title="", height=340, tooltip=["series", "step", "value"])
        with gr.Tab(t("monitor.t_params")) as sub:
            register_translatable(sub, "monitor.t_params")
            tel_note = gr.Markdown()
            group_df = _labelled(gr.Dataframe(label=t("monitor.groups"), interactive=False, wrap=True), "monitor.groups")
            with gr.Row():
                load_html, imp_html = gr.HTML(), gr.HTML()
            with gr.Row():
                wrms_html, grms_html = gr.HTML(), gr.HTML()
            tensor_df = _labelled(gr.Dataframe(label=t("monitor.tensors"), interactive=False, max_height=520,
                                               column_widths=["30%", "10%", "8%", "8%", "8.8%", "8.8%", "8.8%", "8.8%", "8.8%"]),
                                  "monitor.tensors")
        with gr.Tab(t("monitor.t_evals")) as sub:
            register_translatable(sub, "monitor.t_evals")
            eval_note = gr.Markdown(t("monitor.eval_note"))
            register_translatable(eval_note, "monitor.eval_note")
            eval_metric = _labelled(gr.Dropdown(choices=[], value=None, label=t("monitor.eval_metric"),
                                                allow_custom_value=True), "monitor.eval_metric")
            eval_plot = gr.LinePlot(x="step", y="value", color="series", y_title="", height=300,
                                    tooltip=["series", "step", "value"])
            events_df = _labelled(gr.Dataframe(label=t("monitor.events"), interactive=False, wrap=True, max_height=420), "monitor.events")
            samples_html = gr.HTML()
        with gr.Tab(t("monitor.t_config")) as sub:
            register_translatable(sub, "monitor.t_config")
            params_json = _labelled(gr.JSON(label=t("monitor.training_params")), "monitor.training_params")
        with gr.Tab(t("monitor.t_log")) as sub:
            register_translatable(sub, "monitor.t_log")
            log_box = _labelled(gr.Textbox(label=t("monitor.log"), lines=24, max_lines=24, interactive=False, autoscroll=True), "monitor.log")
            ckpt_df = _labelled(gr.Dataframe(label=t("monitor.checkpoints"), interactive=False), "monitor.checkpoints")

        outputs = [run, phase, status, *plots, loss_term, loss_plot, tel_note, group_df, tensor_df,
                   load_html, imp_html, wrms_html, grms_html, eval_metric, eval_plot, events_df, samples_html,
                   params_json, log_box, ckpt_df]
        inputs = [runs_dir, run, phase, loss_term, eval_metric]
        refresh_btn.click(fn=refresh, inputs=inputs, outputs=outputs)
        run.input(fn=refresh, inputs=inputs, outputs=outputs)
        phase.input(fn=refresh, inputs=inputs, outputs=outputs)
        loss_term.input(fn=refresh, inputs=inputs, outputs=outputs)
        eval_metric.input(fn=refresh, inputs=inputs, outputs=outputs)
        runs_dir.submit(fn=refresh, inputs=inputs, outputs=outputs)

        tab.select(fn=refresh, inputs=inputs, outputs=outputs)         # first paint when the tab opens
        timer = gr.Timer(15.0)
        timer.tick(fn=refresh, inputs=inputs, outputs=outputs)
        auto.change(fn=lambda on: gr.Timer(active=bool(on)), inputs=[auto], outputs=[timer])
    tab.monitor_wiring = (refresh, inputs, outputs)                  # app.load() paints it on page open
    return tab
