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

function variantColumns() {
  const m = (key, fmt) => (v) => (v.error ? "" : fmt((v.measured || {})[key]));
  const f = (key, fmt) => (v) => (v.error ? "" : fmt((v.fidelity || {})[key]));
  return [
    { label: t("probe.variant"), fmt: (v) => h("span", { class: "mono", text: v.label }) },
    { label: t("probe.exact"), num: true, fmt: f("exact_match_rate", (x) => pct(x)) },
    { label: t("probe.agree"), num: true, fmt: f("positionwise_agreement", (x) => pct(x)) },
    { label: t("probe.prefix"), num: true, fmt: f("min_matching_prefix", (x) => int(x)) },
    { label: t("probe.hit"), num: true, fmt: m("resident_hit_rate", (x) => pct(x)) },
    { label: t("probe.pred_hit"), num: true, fmt: m("prediction_hit_rate", (x) => pct(x)) },
    { label: t("probe.cold_hit"), num: true, fmt: m("cold_miss_predict_hit_rate", (x) => pct(x)) },
    { label: t("probe.loads"), num: true, fmt: m("on_demand_loads", (x) => int(x)) },
    { label: t("probe.wait"), num: true, fmt: m("prefetch_wait_time_s", (x) => num(x, 3)) },
    { label: t("probe.tps"), num: true, fmt: m("tokens_per_s", (x) => num(x, 1)) },
    { label: t("probe.vs_full"), num: true, fmt: m("speed_vs_full", (x) => (x === null || x === undefined ? "–" : `${num(x, 2)}×`)) },
    { label: t("probe.accept"), num: true, fmt: m("spec_acceptance_rate", (x) => pct(x)) },
    { label: t("probe.rss"), num: true, fmt: m("peak_rss_gb", (x) => num(x, 2)) },
    { label: t("probe.pages"), num: true, fmt: (v) => (v.error ? "" : `${(v.measured || {}).paging_resolved ?? "–"} · ${int((v.measured || {}).num_pages)}`) },
    { label: t("ui.error"), fmt: (v) => (v.error ? h("span", { class: "error", text: v.error }) : "") },
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
      table(variantColumns(), latest.variants),
      Object.keys(latest.references || {}).length ? h("p", { class: "note", text: `${t("probe.reference")}: ${Object.entries(latest.references)
        .map(([b, tps]) => `${b} ${num(tps, 1)} ${t("probe.tps")}`).join(" · ")}` }) : null,
      historyChart(done));
  }
  return card(t("probe.title"), ...parts);
}

// History of one chosen variant over training (sync_on_demand first by default).
function historyChart(all) {
  // Steps restart in every phase: plot the newest phase only.
  const phase = all[all.length - 1].phase;
  const done = all.filter((p) => p.phase === phase);
  if (done.length < 2) return null;
  const labels = [...new Set(done.flatMap((p) => p.variants.filter((v) => !v.error).map((v) => v.label)))];
  if (!labels.length) return null;
  const want = state.values.probeVariant;
  const pick = labels.includes(want) ? want : (labels.find((l) => l.includes("sync_on_demand")) || labels[0]);
  const series = [
    ["probe.hit", (v) => v.measured.resident_hit_rate],
    ["probe.pred_hit", (v) => v.measured.prediction_hit_rate],
    ["probe.agree", (v) => v.fidelity.positionwise_agreement],
    ["probe.exact", (v) => v.fidelity.exact_match_rate],
  ].map(([key, get], i) => ({ label: t(key), color: `var(--s${i + 1})`,
    points: done.map((p) => { const v = p.variants.find((x) => x.label === pick && !x.error); return v ? [p.step, get(v)] : null; })
      .filter((pt) => pt && pt[1] !== null && pt[1] !== undefined) }));
  const sel = h("select", { id: "probe-variant" }, labels.map((l) => h("option", { value: l, text: l })));
  sel.value = pick;
  sel.addEventListener("change", () => { state.values.probeVariant = sel.value; document.dispatchEvent(new CustomEvent("chronos:refresh")); });
  return h("div", { class: "mt" },
    h("div", { class: "row" }, h("label", { class: "compact", for: "probe-variant", text: t("probe.variant") }), sel),
    lineChart({ title: `${t("probe.history")} · ${phaseLabel(phase)}`, series, xLabel: t("ui.step"), yFmt: (v) => pct(v, 0) }));
}
