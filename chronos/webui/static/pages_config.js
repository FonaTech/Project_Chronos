// Model page: presets, every config group, live size / memory estimate, auto plan.
import { state, t, has, h, api, post, card, tile, pageHead, button, busy, params, bytes, num, int,
  toast, reportError, ApiError, table, fieldLabel, clone } from "./core.js";
import { renderGroup, formState, markErrors, assign, submit } from "./forms.js";
import { pathControl } from "./picker.js";

const PRESET_SLUGS = { "Chronos-v3 Tiny (smoke / CI)": "tiny", "Chronos-v3 50M (16k vocab)": "m50",
  "Chronos-v3 200M (32 GB)": "m200", "Chronos-v3 500M (32 GB)": "m500", "Chronos-v3 1B (64 GB)": "b1" };

let presetCache = null;
export async function presets() {
  if (!presetCache) presetCache = await api("/api/presets");
  return presetCache;
}

function schemaDefaults() {
  const out = {};
  for (const f of state.schema.fields) if (f.scope.includes("config")) out[f.key] = clone(f.default);
  return out;
}

// The model config shared by every page; starts from the default preset.
export async function ensureConfig() {
  if (state.config) return state.config;
  const data = await presets();
  const preset = data.presets.find((p) => p.name === data.default);
  state.config = { ...schemaDefaults(), ...clone(preset ? preset.config : {}) };
  state.presetName = preset ? preset.name : "";
  return state.config;
}

export function presetLabel(p) {
  const slug = PRESET_SLUGS[p.name];
  return slug && has(`preset.${slug}`) ? t(`preset.${slug}`) : p.name;
}

let estimateTimer = 0;
function scheduleEstimate(host, form) {
  clearTimeout(estimateTimer);
  estimateTimer = setTimeout(() => refreshEstimate(host, form), 350);
}

async function refreshEstimate(host, form) {
  try {
    const e = await post("/api/config/estimate", { config: state.config });
    markErrors(form, [], { scroll: false });
    const m = e.memory || {};
    host.replaceChildren(h("div", { class: "tiles" },
      tile(t("config.total"), params(e.total_params)),
      tile(t("config.active"), params(e.active_params), `≤ ${params(e.active_union_upper)}`),
      tile(t("config.compute"), params(e.compute_params)),
      tile(t("config.virtual_layers"), int(e.virtual_layers)),
      tile(t("config.vram"), bytes(m.vram_estimate_bytes)),
      tile(t("config.ram"), bytes(m.ram_estimate_bytes)),
      tile(t("config.ssd"), bytes(m.ssd_estimate_bytes)),
      tile(t("config.kv"), bytes(m.kv_cache_bytes)),
      tile(t("config.train_state"), bytes(m.training_state_bytes)),
      tile(t("config.tps"), num(e.decode_tps_estimate, 0))),
    h("p", { class: "note", text: t("config.estimate_note") }));
  } catch (err) {
    if (err instanceof ApiError && err.status === 422) {
      markErrors(form, err.fields, { scroll: false });
      host.replaceChildren(h("p", { class: "error", text: t("ui.invalid") }));
    } else reportError(err);
  }
}
function presetNote(p) {
  const m = (p && p.meta) || {};
  const parts = [];
  if (m.total_params) parts.push(`${t("config.total")} ${params(m.total_params)}`);
  if (m.active_params) parts.push(`${t("config.active")} ${params(m.active_params)}`);
  if (m.training_peak_gb) parts.push(`${t("config.train_peak")} ${num(m.training_peak_gb, 1)} GB (${t("label.estimated")})`);
  const slug = p && PRESET_SLUGS[p.name];
  const summary = slug && has(`preset.${slug}.summary`) ? t(`preset.${slug}.summary`) : "";
  return [summary, parts.join(" · ")].filter(Boolean).join(" — ");
}

export async function renderConfig(main) {
  const data = await presets();
  await ensureConfig();
  const estimate = h("div", null, h("p", { class: "note", text: t("ui.loading") }));
  const form = h("div");
  const redraw = () => {
    form.replaceChildren(...state.schema.config_groups.map((g, i) =>
      renderGroup("config", g, state.config, () => scheduleEstimate(estimate, form), { open: i < 3 })));
  };
  const apply = (cfg, message) => {
    assign(state.config, cfg);
    redraw();
    refreshEstimate(estimate, form);
    if (message) toast(message);
  };

  // presets
  const picker = h("select", { id: "preset" }, data.presets.map((p) => h("option", { value: p.name, text: presetLabel(p) })));
  picker.value = state.presetName || data.default;
  const note = h("p", { class: "note" });
  const showNote = () => { note.textContent = presetNote(data.presets.find((p) => p.name === picker.value)); };
  picker.addEventListener("change", showNote);
  showNote();
  const usePreset = button("ui.apply", () => {
    const p = data.presets.find((x) => x.name === picker.value);
    if (!p) return;
    state.config = { ...schemaDefaults(), ...clone(p.config) };
    state.presetName = p.name;
    apply({}, `${t("config.loaded")}: ${presetLabel(p)}`);
  }, { primary: true });

  // file load / save
  const path = h("input", { type: "text", id: "config-path", spellcheck: "false", value: state.values.configPath || "./out/chronos_config.json" });
  path.addEventListener("input", () => { state.values.configPath = path.value; });
  const saveBtn = button("ui.save", () => busy(saveBtn, async () => {
    const r = await post("/api/config/save", { config: state.config, path: path.value });
    toast(`${t("ui.saved")}: ${r.path}`);
  }));
  const loadBtn = button("ui.load", () => busy(loadBtn, async () => {
    const r = await post("/api/config/load", { path: path.value });
    apply(r.config, `${t("config.loaded")}: ${r.path}`);
    if (r.ignored && r.ignored.length) toast(`${t("config.ignored")}: ${r.ignored.map(fieldLabel).join(", ")}`, "error");
  }));
  const resetBtn = button("ui.reset", () => {
    state.config = null;
    ensureConfig().then(() => { picker.value = state.presetName; showNote(); apply({}); });
  });

  main.append(
    pageHead("nav.config", "config.intro"),
    h("div", { class: "grid two" },
      card(t("config.preset"),
        h("div", { class: "row", tip: t("config.preset_help") }, h("label", { class: "compact", for: "preset", text: t("config.preset") }), picker, usePreset),
        note),
      card(t("config.file"),
        h("div", { class: "field wide", tip: t("config.path_help") }, h("label", { class: "compact", for: "config-path", text: t("ui.path") }),
          pathControl(path, { kind: "config", multiple: false })),
        h("div", { class: "row mt" }, loadBtn, saveBtn, resetBtn))),
    card(t("config.estimate"), estimate),
    card(null, form),
    planCard(apply));
  redraw();
  refreshEstimate(estimate, form);
}
// Auto plan: target size + data + memory → config and per-stage settings (adaptive planner).
function planCard(apply) {
  const values = formState("plan", "planner");
  const root = h("div");
  const out = h("div", { class: "stack mt" });
  const run = button("plan.run", () => submit(run, root, async () => {
    showPlan(out, await post("/api/plan", values), apply);
  }), { primary: true });
  root.append(h("p", { class: "note", text: t("plan.note") }), renderGroup("planner", "planner", values, null, { summary: false }),
    h("div", { class: "row mt" }, run), out);
  return card(t("plan.title"), root);
}

function showPlan(out, plan, apply) {
  const a = (plan.report && plan.report.architecture) || {};
  const sc = (plan.report && plan.report.scaling) || {};
  const mem = (plan.report && plan.report.memory) || {};
  const stages = plan.stage_overrides || {};
  const keys = [...new Set(Object.values(stages).flatMap((row) => Object.keys(row || {})))];
  const use = button("plan.apply", () => apply({ ...plan.config, stage_overrides: plan.stage_overrides }, t("config.loaded")), { primary: true });
  out.replaceChildren(
    h("h4", { text: t("plan.summary") }),
    h("div", { class: "tiles" },
      tile(t("config.total"), params(a.total_params), t("label.measured_meta")),
      tile(t("config.active"), params(a.active_params), `≤ ${params(a.active_union_upper)}`),
      tile(t("config.virtual_layers"), int(a.virtual_layers), `${int(a.physical_layers)} × ${int(a.recurrent_depth)}`),
      tile(fieldLabel("hidden_size"), int(a.hidden_size)),
      tile(fieldLabel("num_experts"), `${int(a.experts)} / top-${int(a.top_k)}`),
      tile(t("plan.tokens"), params(sc.tokens), t("label.estimated")),
      tile(fieldLabel("learning_rate"), num(sc.chosen_lr, 5), t("label.estimated")),
      tile(t("config.train_peak"), mem.training_peak_gb === undefined ? "–" : `${num(mem.training_peak_gb, 1)} GB`, t("label.estimated"))),
    h("h4", { text: t("plan.stages") }),
    table([{ label: t("ui.phase"), fmt: (r) => r.stage }, ...keys.map((k) => ({ label: fieldLabel(k), num: true,
      fmt: (r) => (r[k] === undefined ? "" : (typeof r[k] === "number" ? num(r[k], 6) : String(r[k]))) }))],
    Object.entries(stages).map(([stage, row]) => ({ stage, ...row }))),
    h("div", { class: "row" }, use),
    h("details", null, h("summary", { text: t("ui.raw_json") }), h("pre", { class: "log", text: JSON.stringify(plan, null, 2) })));
}

