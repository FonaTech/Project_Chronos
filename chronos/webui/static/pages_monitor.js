// Run monitor: any run folder under runs_dir (Web UI jobs and the CLI long runs).
// Reads only; never signals the training process.
import { state, t, has, h, api, card, tile, pageHead, badge, num, int, params, duration, dateTime, logBox, table, toast } from "./core.js";
import { lineChart } from "./charts.js";
import { probeSection } from "./pages_probe.js";

// Small multiples: one chart per measure family, one y-axis each.
const CHARTS = [
  { id: "loss", keys: ["loss", "primary_loss"] },
  { id: "aux", keys: ["balance_loss", "lookahead_loss", "temporal_loss", "locality_loss"] },
  { id: "throughput", keys: ["tokens_per_s"] },
  { id: "memory", keys: ["peak_mem_gb", "active_mem_gb"] },
  { id: "optimizer", keys: ["grad_norm"] },
  { id: "lr", keys: ["lr"] },
  { id: "recurrent", keys: ["rec_depth_mean", "rec_contraction", "rec_early_exit_ratio"] },
  { id: "routing", keys: ["route_entropy", "load_cv"] },
];
const metricLabel = (key) => (has(`metric.${key}`) ? t(`metric.${key}`) : key);
const chartFmt = { lr: (v) => num(v, 6), throughput: (v) => num(v, 0) };

function phaseLabel(name) { return has(`stage.${name}`) ? t(`stage.${name}`) : String(name ?? "–"); }

export async function renderMonitor(main, page) {
  const runSel = h("select", { id: "mon-run" });
  const head = h("div", { class: "row" }, h("label", { class: "compact", for: "mon-run", text: t("mon.run") }), runSel);
  const body = h("div", { class: "stack" }, h("p", { class: "note", text: t("ui.loading") }));
  main.append(pageHead("mon.title", "mon.intro", head), body);
  runSel.addEventListener("change", () => { state.values.monitorRun = runSel.value; page.refresh(); });
  page.poll(async () => {
    const run = state.values.monitorRun;
    let snap;
    try {
      snap = await api(`/api/monitor/state?history=1${run ? `&run=${encodeURIComponent(run)}` : ""}`);
    } catch (err) {
      if (err.status === 404 && run) { state.values.monitorRun = ""; toast(t("err.not_found"), "error"); return 100; }
      throw err;
    }
    const names = (snap.runs || []).map((r) => r.name);
    if (runSel.options.length !== names.length || [...runSel.options].some((o, i) => o.value !== names[i])) {
      runSel.replaceChildren(...names.map((n) => h("option", { value: n, text: n })));
    }
    if (snap.run) runSel.value = snap.run;
    if (!snap.run) { body.replaceChildren(h("div", { class: "empty", text: t("mon.no_runs") })); return 10000; }
    paint(body, snap);
    return snap.status && snap.status.state === "running" ? 5000 : 15000;
  });
}

function paint(body, snap) {
  const st = snap.status || {};
  const pr = snap.progress || {};
  const latest = snap.latest || {};
  const mem = snap.memory || {};
  const best = snap.best_valid || {};
  const current = pr.current || latest.phase;
  const bestNow = best[current];
  const phaseBar = (pr.phases || []).map((p) => h("div", { class: "phase-row" },
    h("span", { class: "phase-name", text: phaseLabel(p.name) }),
    h("div", { class: "progress", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": p.total || 0, "aria-valuenow": p.step || 0,
      "aria-label": phaseLabel(p.name) }, bar(p.total ? Math.min(1, p.step / p.total) : (p.state === "complete" ? 1 : 0))),
    h("span", { class: "note", text: `${int(p.step)} / ${p.total ? int(p.total) : "–"} · ${t(`phase.${p.state}`)}` })));
  body.replaceChildren(
    h("div", { class: "row between" },
      h("div", { class: "row" }, badge(st.stale ? "stale" : st.state), st.stale ? h("span", { class: "note", text: t("mon.stale") }) : null,
        st.last_update_age_s !== null && st.last_update_age_s !== undefined ? h("span", { class: "note", text: t("mon.updated", { s: int(st.last_update_age_s) }) }) : null,
        st.pid ? h("span", { class: "note mono", text: `PID ${st.pid}` }) : null),
      st.error ? h("span", { class: "error", text: st.error }) : null),
    h("div", { class: "tiles" },
      tile(t("ui.phase"), phaseLabel(current)),
      tile(t("ui.step"), int(latest.step), latest.total ? `/ ${int(latest.total)}` : null),
      tile(t("mon.loss"), num(latest.loss, 4), latest.primary_loss !== undefined ? `${metricLabel("primary_loss")} ${num(latest.primary_loss, 4)}` : null),
      tile(t("mon.valid_ce"), bestNow ? num(bestNow.valid_ce, 4) : "–", bestNow ? `${t("mon.best")} · ${t("ui.step")} ${int(bestNow.step)}` : null),
      tile(t("mon.throughput"), num(latest.tokens_per_s, 0), latest.tokens_seen ? `${params(latest.tokens_seen)} ${t("mon.tokens_seen")}` : null),
      tile(t("mon.memory"), mem.footprint_gb !== undefined ? `${num(mem.footprint_gb, 1)} GB` : (latest.peak_mem_gb !== undefined ? `${num(latest.peak_mem_gb, 1)} GB` : "–"),
        mem.state === "measured" ? t("label.measured") : null),
      tile(t("mon.eta"), duration(pr.run_remaining_s), pr.eta ? dateTime(pr.eta) : null),
      tile(t("mon.lr"), num(latest.lr, 6), latest.grad_norm !== undefined ? `${t("mon.grad_norm")} ${num(latest.grad_norm, 3)}` : null)),
    phaseBar.length ? card(t("mon.progress"), ...phaseBar) : null,
    probeSection(snap),
    h("div", { class: "grid two" }, CHARTS.map((c) => chart(c, snap))),
    evalCharts(snap),
    eventsCard(snap),
    samplesCard(snap),
    checkpointsCard(snap),
    logCard(snap));
}

function bar(fraction) {
  const fill = h("div");
  fill.style.width = `${Math.round(Math.max(0, Math.min(1, fraction)) * 1000) / 10}%`;
  return fill;
}
// Steps restart in every phase, so charts show one phase at a time (chosen in the card header).
function viewPhase(snap) {
  const phases = Object.keys(snap.history || {});
  const want = state.values.monitorPhase;
  return phases.includes(want) ? want : ((snap.progress && snap.progress.current) || phases[phases.length - 1]);
}

function chart(c, snap) {
  const rows = (snap.history || {})[viewPhase(snap)] || [];
  const series = c.keys.map((key, i) => ({ label: metricLabel(key), color: `var(--s${i + 1})`,
    points: rows.filter((r) => r[key] !== null && r[key] !== undefined).map((r) => [r.step, Number(r[key])]) }));
  return card(null, lineChart({ title: t(`mon.chart.${c.id}`), series, xLabel: t("ui.step"), yFmt: chartFmt[c.id] || ((v) => num(v, 3)) }));
}

function evalCharts(snap) {
  const phase = viewPhase(snap);
  const vals = (snap.events || []).filter((e) => e.event === "validation" && e.phase === phase);
  const keys = (snap.evals && snap.evals.keys) || ["valid_ce"];
  const charts = keys.filter((k) => k !== "valid_ppl" && vals.some((e) => e[k] !== undefined && e[k] !== null)).map((k, i) =>
    card(null, lineChart({ title: metricLabel(k), xLabel: t("ui.step"),
      series: [{ label: metricLabel(k), color: `var(--s${(i % 8) + 1})`, points: vals.filter((e) => e[k] !== null && e[k] !== undefined).map((e) => [e.step, Number(e[k])]) }] })));
  const phases = Object.keys(snap.history || {});
  const picker = h("select", { id: "mon-phase" }, phases.map((p) => h("option", { value: p, text: phaseLabel(p) })));
  picker.value = phase || "";
  picker.addEventListener("change", () => { state.values.monitorPhase = picker.value; document.dispatchEvent(new CustomEvent("chronos:refresh")); });
  return card(t("mon.validation"), h("div", { class: "row" }, h("label", { class: "compact", for: "mon-phase", text: t("ui.phase") }), picker,
    h("span", { class: "note", text: t("mon.phase_note") })),
  charts.length ? h("div", { class: "grid two mt" }, charts) : h("p", { class: "note mt", text: t("ui.empty") }));
}

function eventsCard(snap) {
  const events = (snap.events || []).slice(-40).reverse();
  if (!events.length) return null;
  const known = ["event", "phase", "step", "time"];
  return card(t("mon.events"), table([
    { label: t("ui.time"), fmt: (e) => dateTime(e.time) },
    { label: t("ui.phase"), fmt: (e) => phaseLabel(e.phase) },
    { label: t("ui.step"), num: true, fmt: (e) => int(e.step) },
    { label: t("mon.event"), fmt: (e) => (has(`event.${e.event}`) ? t(`event.${e.event}`) : e.event) },
    { label: t("ui.value"), fmt: (e) => Object.entries(e).filter(([k, v]) => !known.includes(k) && v !== null && typeof v !== "object")
      .slice(0, 6).map(([k, v]) => `${metricLabel(k)} ${typeof v === "number" ? num(v, 4) : v}`).join(" · ") },
  ], events));
}

function samplesCard(snap) {
  const samples = (snap.samples || []).slice(-6).reverse();
  if (!samples.length) return null;
  return card(t("mon.samples"), samples.map((x) => h("div", { class: "sample" },
    h("div", { class: "note", text: `${phaseLabel(x.phase)} · ${t("ui.step")} ${int(x.step)}` }),
    x.input ? h("div", null, h("b", { text: `${t("mon.input")}: ` }), x.input) : null,
    x.reference ? h("div", null, h("b", { text: `${t("mon.reference")}: ` }), x.reference) : null,
    h("div", { class: "output" }, x.output || ""))));
}

function checkpointsCard(snap) {
  const rows = snap.checkpoints || [];
  if (!rows.length) return null;
  return card(t("mon.checkpoints"), table([
    { label: t("ui.path"), fmt: (r) => h("span", { class: "mono", text: r.file }) },
    { label: t("mon.size"), num: true, fmt: (r) => `${num(r.size_gb, 3)} GB` },
    { label: t("ui.time"), fmt: (r) => dateTime(r.modified) },
  ], rows.slice(0, 12)));
}

function logCard(snap) {
  const box = h("div", null, h("p", { class: "note", text: t("ui.loading") }));
  api(`/api/monitor/log?run=${encodeURIComponent(snap.run)}&bytes=12000`).then((r) => box.replaceChildren(logBox(r.text)))
    .catch(() => box.replaceChildren(h("p", { class: "note", text: t("ui.empty") })));
  return h("details", { class: "card" }, h("summary", { text: t("mon.log") }), box);
}
