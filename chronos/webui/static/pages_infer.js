// Inference: lazy experts vs full memory on one prompt; architecture comes from the checkpoint.
import { state, t, has, h, api, post, card, pageHead, button, num, int, pct, table, logBox, badge, toast } from "./core.js";
import { renderGroup, formState, submit } from "./forms.js";
import { lineChart, barChart } from "./charts.js";
import { ensureConfig } from "./pages_config.js";

const COLS = [
  ["tokens", (v) => int(v)], ["tokens_per_sec", (v) => num(v, 1)], ["response_time_s", (v) => num(v, 2)],
  ["prefill_time_s", (v) => num(v, 3)], ["decode_time_s", (v) => num(v, 2)],
  ["cache_hit_rate", (v) => pct(v)], ["prediction_hit_rate", (v) => pct(v)], ["cold_miss_predict_hit_rate", (v) => pct(v)],
  ["fallback_weight_rate", (v) => pct(v)], ["spec_acceptance_rate", (v) => pct(v)],
  ["rss_delta_gb", (v) => num(v, 3)], ["rss_after_decode_gb", (v) => num(v, 2)],
];
const colLabel = (key) => (has(`infer.col.${key}`) ? t(`infer.col.${key}`) : key);
function modeLabel(mode) {
  const m = /^lazy_offload_([\d.]+)x$/.exec(String(mode));
  if (m) return `${t("infer.lazy_output")} · ${m[1]}×`;
  return has(`infer.mode.${mode}`) ? t(`infer.mode.${mode}`) : String(mode ?? "");
}

export async function renderInference(main, page) {
  await ensureConfig();
  const values = formState("infer", "inference");
  const v = state.values.inferPage || (state.values.inferPage = { prompt: "", job: null });
  const root = h("div");
  const out = h("div", { class: "stack" });
  const prompt = h("textarea", { id: "infer-prompt", rows: 4, placeholder: t("infer.prompt") });
  prompt.value = v.prompt;
  prompt.addEventListener("input", () => { v.prompt = prompt.value; });
  const go = button("infer.generate", () => submit(go, root, async () => {
    const r = await post("/api/infer", { config: state.config, infer: values, prompt: v.prompt });
    v.job = r.job.id;
    page.refresh();
  }), { primary: true });
  root.append(
    h("div", { class: "field wide", dataset: { key: "prompt" }, tip: t("infer.prompt_help") }, h("label", { class: "label", for: "infer-prompt", text: t("infer.prompt") }), prompt),
    h("div", { class: "mt" }, ["inference", "lazy", "speculative"].map((g, i) => renderGroup("inference", g, values, null, { open: i < 2 }))),
    h("p", { class: "note", text: t("infer.arch_note") }),
    h("div", { class: "row end" }, go));
  main.append(pageHead("infer.title", "infer.intro"), card(null, root), out);
  page.poll(async () => {
    if (!v.job) return 0;
    let job;
    try { job = await api(`/api/jobs/${encodeURIComponent(v.job)}?log=200`); } catch (err) { if (err.status === 404) { v.job = null; return 0; } throw err; }
    go.disabled = ["running", "starting", "stopping"].includes(job.state);
    paint(out, job);
    if (job.state === "failed" && !v.reported) { v.reported = true; toast(job.error || t("state.failed"), "error"); }
    return go.disabled ? 1500 : 0;
  });
}

function paint(out, job) {
  const r = job.result || {};
  const rows = r.rows || [];
  const o = r.outputs || {};
  const diff = r.output_diff;
  const blocks = [h("div", { class: "row between" }, h("h3", { text: t("infer.output") }), badge(job.state))];
  if (job.state !== "finished") blocks.push(logBox(job.log));
  if (o.lazy || o.full_dram) {
    blocks.push(h("div", { class: "grid two" },
      card(t("infer.lazy_output"), h("div", { class: "output", text: o.lazy || "" })),
      card(t("infer.full_output"), h("div", { class: "output", text: o.full_dram || "" }))));
  } else if (o.single) blocks.push(h("div", { class: "output", text: o.single }));
  if (diff) {
    blocks.push(h("p", { class: diff.same ? "note" : "error",
      text: `${t(diff.same ? "infer.same" : "infer.diff")} · ${t("infer.similarity")} ${pct(diff.similarity)} · ${t("infer.common_prefix", { n: int(diff.common_prefix_chars) })}` }));
  }
  if (r.backend) blocks.push(h("p", { class: "note", text: `${t("ui.backend")}: ${r.backend}${r.backend_note ? ` — ${r.backend_note}` : ""}` }));
  if (rows.length) {
    const used = COLS.filter(([key]) => rows.some((row) => row[key] !== undefined && row[key] !== null));
    blocks.push(card(t("infer.stats"), table([{ label: t("infer.mode"), fmt: (row) => modeLabel(row.mode) },
      ...used.map(([key, fmt]) => ({ label: colLabel(key), num: true, fmt: (row) => (row[key] === undefined || row[key] === null ? "–" : fmt(row[key])) }))], rows),
    h("p", { class: "note", text: t("infer.measured_note") })));
    const sweep = rows.filter((row) => row.ram_load_ratio !== undefined && row.ram_load_ratio !== null);
    if (sweep.length > 1) {
      const pts = (key) => sweep.map((row) => [row.ram_load_ratio, row[key]]).filter((p) => p[1] !== null && p[1] !== undefined);
      blocks.push(card(t("infer.sweep"), h("div", { class: "grid two" },
        lineChart({ title: colLabel("tokens_per_sec"), xLabel: t("infer.ram_ratio"), xFmt: (x) => `${num(x, 2)}×`, yFmt: (y) => num(y, 1),
          series: [{ label: colLabel("tokens_per_sec"), color: "var(--s1)", points: pts("tokens_per_sec") }] }),
        lineChart({ title: t("infer.hit_rates"), xLabel: t("infer.ram_ratio"), xFmt: (x) => `${num(x, 2)}×`, yFmt: (y) => pct(y, 0),
          series: [["cache_hit_rate", "var(--s1)"], ["prediction_hit_rate", "var(--s2)"]].map(([key, c]) => ({ label: colLabel(key), color: c, points: pts(key) })) }),
        lineChart({ title: colLabel("rss_delta_gb"), xLabel: t("infer.ram_ratio"), xFmt: (x) => `${num(x, 2)}×`, yFmt: (y) => num(y, 2),
          series: [{ label: colLabel("rss_delta_gb"), color: "var(--s1)", points: pts("rss_delta_gb") }] }))));
    } else if (rows.length > 1) {
      blocks.push(card(null, h("div", { class: "grid two" },
        barChart({ title: colLabel("tokens_per_sec"), categories: rows.map((row) => modeLabel(row.mode)),
          series: [{ label: colLabel("tokens_per_sec"), values: rows.map((row) => Number(row.tokens_per_sec)) }], fmt: (x) => num(x, 1) }),
        barChart({ title: colLabel("rss_delta_gb"), categories: rows.map((row) => modeLabel(row.mode)),
          series: [{ label: colLabel("rss_delta_gb"), values: rows.map((row) => Number(row.rss_delta_gb)) }], fmt: (x) => num(x, 3) }))));
    }
  }
  if (job.error) blocks.push(h("p", { class: "error", text: job.error }));
  out.replaceChildren(card(null, ...blocks));
}
