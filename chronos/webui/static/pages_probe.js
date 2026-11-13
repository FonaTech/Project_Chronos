// Lazy-expert probe results in the run monitor: efficiency, hit rates and
// fidelity of every probed variant against the full-memory reference.
import { state, t, has, h, card, num, int, pct, dateTime, table } from "./core.js";
import { lineChart } from "./charts.js";

const SKIPS = [
  [/^dense model/, "probe.skip.dense"], [/^no checkpoint/, "probe.skip.no_checkpoint"],
  [/^previous probe/, "probe.skip.busy"], [/^checkpoint still/, "probe.skip.writing"],
  [/^superseded by step (\d+)/, "probe.skip.superseded"],
];
function skipText(reason) {
  for (const [re, key] of SKIPS) {
    const m = re.exec(String(reason || ""));
    if (m) return t(key, { step: m[1] || "" });
  }
  return String(reason || "");
}

const phaseLabel = (name) => (has(`stage.${name}`) ? t(`stage.${name}`) : String(name ?? "–"));
const yesNo = (v) => (v === null || v === undefined ? "–" : t(v ? "ui.yes" : "ui.no"));

const secs = (x) => (x === null || x === undefined ? "–" : num(x, Math.abs(Number(x)) >= 10 ? 1 : 3));
const pair = (a, b) => (a == null && b == null ? "–" : `${int(a)} / ${int(b)}`);
const mOf = (v) => v.measured || {};
const fOf = (v) => v.fidelity || {};

// Columns that no row fills are dropped (e.g. draft acceptance without speculative variants).
function groupTable(title, columns, rows) {
  const live = columns.filter((c) => c.keep || rows.some((r) => { const x = c.value(r); return x !== null && x !== undefined && x !== ""; }));
  return h("div", { class: "mt" }, h("h4", { text: title }),
    table(live.map((c) => ({ label: c.label, num: c.num, fmt: (r) => (r.error && !c.keep && !c.always ? "" : c.fmt(c.value(r), r)) })), rows));
}
const col = (key, get, fmt) => ({ label: t(key), num: true, value: get, fmt });
function variantTables(latest) {
  const variants = latest.variants || [];
  const refs = Object.entries(latest.reference_measured || {}).map(([b, m]) => ({
    label: t("probe.ref_row", { backend: b }), measured: m && !m.error ? m : {}, ...(m && m.error ? { error: m.error } : {}) }));
  const name = { label: t("probe.variant"), keep: true, value: (v) => v.label, fmt: (x) => h("span", { class: "mono", text: x }) };
  const errCol = { label: t("ui.error"), always: true, fmt: (x) => (x ? h("span", { class: "error", text: x }) : "") };
  return [
    groupTable(t("probe.group.eff"), [name,
      col("probe.tps", (v) => mOf(v).tokens_per_s, (x) => num(x, 1)), col("probe.dtps", (v) => mOf(v).decode_tokens_per_s, (x) => num(x, 1)),
      col("probe.vs_full", (v) => mOf(v).speed_vs_full, (x) => (x === null || x === undefined ? "–" : `${num(x, 2)}×`)),
      col("probe.prefill", (v) => mOf(v).prefill_time_s, secs), col("probe.decode", (v) => mOf(v).decode_time_s, secs),
      col("probe.rss", (v) => mOf(v).peak_rss_gb, (x) => num(x, 2)), col("probe.accept", (v) => mOf(v).spec_acceptance_rate, (x) => pct(x)),
      { ...errCol, value: (v) => v.error }], [...variants, ...refs]),
    groupTable(t("probe.group.paging"), [name,
      col("probe.loads", (v) => mOf(v).on_demand_loads, int), col("probe.load_s", (v) => mOf(v).on_demand_load_time_s, secs),
      col("probe.sync", (v) => mOf(v).sync_ssd_loads, int), col("probe.cluster_loads", (v) => mOf(v).cluster_loads, int),
      col("probe.wait", (v) => mOf(v).prefetch_wait_time_s, secs),
      { label: t("probe.pages"), value: (v) => mOf(v).paging_resolved, fmt: (x, r) => `${x ?? "–"} · ${int(mOf(r).num_pages)}` },
      col("probe.budget", (v) => mOf(v).effective_expert_budget, int), col("probe.ideal", (v) => mOf(v).ideal_active_experts, int),
      { label: t("probe.evict"), value: (v) => mOf(v).eviction_policy, fmt: (x) => x ?? "–" }], variants),
    groupTable(t("probe.group.hit"), [name,
      col("probe.hit", (v) => mOf(v).resident_hit_rate, (x) => pct(x)),
      col("probe.hits", (v) => (mOf(v).cache_hits ?? mOf(v).cache_misses ?? null), (_x, r) => pair(mOf(r).cache_hits, mOf(r).cache_misses)),
      col("probe.pred_hit", (v) => mOf(v).prediction_hit_rate, (x) => pct(x)),
      col("probe.pred_n", (v) => (mOf(v).prediction_total || null), (_x, r) => pair(mOf(r).prediction_hits, mOf(r).prediction_total)),
      col("probe.cold_hit", (v) => mOf(v).cold_miss_predict_hit_rate, (x) => pct(x)),
      col("probe.cold_n", (v) => (mOf(v).cold_prediction_total || null), (_x, r) => pair(mOf(r).cold_prediction_hits, mOf(r).cold_prediction_total))],
    variants),
    groupTable(t("probe.group.fid"), [name,
      col("probe.exact", (v) => fOf(v).exact_match_rate, (x) => pct(x)), col("probe.agree", (v) => fOf(v).positionwise_agreement, (x) => pct(x)),
      col("probe.prefix", (v) => fOf(v).min_matching_prefix, int), col("probe.ref_tokens", (v) => fOf(v).reference_tokens, int),
      { ...errCol, value: (v) => v.error }], variants),
  ];
}

export function probeSection(snap) {
  const probes = snap.lazy_probe || [];
  if (!probes.length) return card(t("probe.title"), h("p", { class: "note", text: t("probe.none") }));
  const done = probes.filter((p) => Array.isArray(p.variants));
  const last = probes[probes.length - 1];
  const latest = done[done.length - 1];
  const parts = [];
  if (last !== latest && (last.skipped || last.error)) {
    parts.push(h("p", { class: "note", text: `${phaseLabel(last.phase)} · ${t("ui.step")} ${int(last.step)}: ${last.skipped ? t("probe.skipped", { reason: skipText(last.skipped) }) : last.error}` }));
  }
  if (latest) {
    const c = latest.conditions || {};
    parts.push(
      h("div", { class: "row" },
        h("span", { class: "pill", text: `${phaseLabel(latest.phase)} · ${t("ui.step")} ${int(latest.step)}` }),
        h("span", { class: "note", text: dateTime(latest.time) }),
        h("span", { class: "note", text: `${t("probe.trained_clusters")}: ${yesNo(latest.trained_with_clusters)}` }),
        h("span", { class: "note", text: `${t("probe.auto_paging")}: ${latest.auto_paging ?? "–"}` }),
        latest.truncated ? h("span", { class: "note", text: t("train.variants_capped", { n: int(latest.variants_requested), cap: int(latest.variants.length) }) }) : null),
      h("p", { class: "note", text: t("probe.conditions", { n: int(c.prompts), t: int(c.max_new_tokens) }) }),
      ...variantTables(latest),
      !latest.reference_measured && Object.keys(latest.references || {}).length ? h("p", { class: "note", text: `${t("probe.reference")}: ${Object.entries(latest.references)
        .map(([b, tps]) => `${b} ${num(tps, 1)} ${t("probe.tps")}`).join(" · ")}` }) : null,
      historyCharts(done));
  }
  return card(t("probe.title"), ...parts);
}

// History of one chosen variant over training (sync_on_demand first by default).
function historyCharts(all) {
  // Steps restart in every phase: plot the newest phase only.
  const phase = all[all.length - 1].phase;
  const done = all.filter((p) => p.phase === phase);
  if (done.length < 2) return null;
  const labels = [...new Set(done.flatMap((p) => p.variants.filter((v) => !v.error).map((v) => v.label)))];
  if (!labels.length) return null;
  const want = state.values.probeVariant;
  const pick = labels.includes(want) ? want : (labels.find((l) => l.includes("sync_on_demand")) || labels[0]);
  const at = (p) => p.variants.find((x) => x.label === pick && !x.error);
  const series = (defs) => defs.map(([key, get], i) => ({ label: t(key), color: `var(--s${i + 1})`,
    points: done.map((p) => { const v = at(p); return v ? [p.step, get(v)] : null; })
      .filter((pt) => pt && pt[1] !== null && pt[1] !== undefined) }));
  const chart = (title, defs, yFmt) => lineChart({ title: `${title} · ${phaseLabel(phase)}`, series: series(defs), xLabel: t("ui.step"), yFmt });
  const sel = h("select", { id: "probe-variant" }, labels.map((l) => h("option", { value: l, text: l })));
  sel.value = pick;
  sel.addEventListener("change", () => { state.values.probeVariant = sel.value; document.dispatchEvent(new CustomEvent("chronos:refresh")); });
  return h("div", { class: "mt" },
    h("div", { class: "row" }, h("label", { class: "compact", for: "probe-variant", text: t("probe.variant") }), sel),
    h("p", { class: "note", text: t("probe.history_note") }),
    h("div", { class: "grid two mt" },
      chart(t("probe.history"), [["probe.hit", (v) => mOf(v).resident_hit_rate], ["probe.pred_hit", (v) => mOf(v).prediction_hit_rate],
        ["probe.agree", (v) => fOf(v).positionwise_agreement], ["probe.exact", (v) => fOf(v).exact_match_rate]], (v) => pct(v, 0)),
      chart(t("probe.c_speed"), [["probe.tps", (v) => mOf(v).tokens_per_s], ["probe.dtps", (v) => mOf(v).decode_tokens_per_s]], (v) => num(v, 1)),
      chart(t("probe.c_loads"), [["probe.loads", (v) => mOf(v).on_demand_loads], ["probe.sync", (v) => mOf(v).sync_ssd_loads],
        ["probe.cluster_loads", (v) => mOf(v).cluster_loads]], (v) => num(v, 0)),
      chart(t("probe.c_rss"), [["probe.rss", (v) => mOf(v).peak_rss_gb]], (v) => num(v, 2))));
}
