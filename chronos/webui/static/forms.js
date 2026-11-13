// Schema-driven forms. Every field carries a hover / focus tooltip in the page
// language: what it does, its default, the recommended value and where that
// recommendation comes from (code default, adaptive planner, measurement, rule).
import { state, t, has, h, num, setTip, ApiError, reportError } from "./core.js";
import { pathControl } from "./picker.js";

const TEXTAREA = new Set(["tok.paths", "plan.data_paths", "train.sample_prompt", "reward_spec"]);
const WIDE = new Set([...TEXTAREA, "train.stages", "probe.ram_ratios", "export.formats", "tune.search",
  "infer.model_path", "train.data_path", "serve.models_dir"]);
const showAdvanced = {};
// Planner validation codes that name a form field.
const ERROR_ALIASES = { target: "plan.target_m", active_experts: "plan.active_value" };

export function fieldsFor(scope, group) {
  return state.schema.fields.filter((f) => f.scope.includes(scope) && (!group || f.group === group));
}

export function defaultsFor(scope) {
  const out = {};
  for (const f of fieldsFor(scope)) out[f.key] = Array.isArray(f.default) ? [...f.default] : f.default;
  return out;
}

// Values of one form, kept in state.values so they survive re-renders.
export function formState(id, scope) {
  if (!state.values[id]) state.values[id] = defaultsFor(scope || id);
  return state.values[id];
}

// Label of a choice value. With the field: its own wording first (the same id can mean different things
// in different fields; "serve.router_scope" shares "router_scope"), stage names for the stage chips and
// field names for the auto-tune search list. Format and device ids (float16, cuda, gguf…) stay as they are.
export function choiceLabel(value, f) {
  if (typeof value === "boolean") return t(value ? "ui.yes" : "ui.no");
  if (value === "" || value === null || value === undefined) return t("ui.from_model");
  if (typeof value === "number") return num(value, 4);
  const s = String(value);
  if (f) {
    for (const key of [f.key, f.key.replace(/^[a-z]+\./, "")]) if (has(`choice.${key}.${s}`)) return t(`choice.${key}.${s}`);
    if (f.key === "train.stages" && has(`stage.${s}`)) return t(`stage.${s}`);
    if (f.key === "tune.search" && has(`field.${s}.label`)) return t(`field.${s}.label`);
  }
  return has(`choice.${s}`) ? t(`choice.${s}`) : s;
}

export function valueText(f, value) {
  if (value === null || value === undefined || value === "") {
    if (f.type === "choice") return choiceLabel("");
    return f.nullable || f.type === "text" || f.type === "path" ? t("tip.auto_empty") : "–";
  }
  if (Array.isArray(value)) return value.map((x) => choiceLabel(x, f)).join(", ");
  if (f.type === "bool") return t(value ? "ui.yes" : "ui.no");
  if (f.type === "choice") return choiceLabel(value, f);
  if (typeof value === "number") return num(value, 6);
  return String(value);
}

export function recommendedText(f) {
  if (f.recommended === undefined) return null;
  if (f.source === "rule" && f.ref && has(`rule.${f.ref}`)) return t(`rule.${f.ref}`);
  return valueText(f, f.recommended);
}

function sourceText(f) {
  if (!f.source) return null;
  const base = t(`source.${f.source}`);
  return f.source === "measured" && f.ref ? `${base} · ${f.ref.split("/").pop()}` : base;
}

export function fieldTip(f) {
  return () => {
    const dl = h("dl");
    const row = (label, value) => { if (value !== null && value !== undefined) dl.append(h("dt", { text: label }), h("dd", { text: value })); };
    row(t("tip.default"), valueText(f, f.default));
    row(t("tip.recommended"), recommendedText(f));
    row(t("tip.source"), sourceText(f));
    if (f.min !== undefined || f.max !== undefined) row(t("tip.range"), `${f.min ?? "−∞"} – ${f.max ?? "∞"}`);
    return h("div", null,
      h("div", { class: "t-title", text: t(`field.${f.key}.label`) }),
      h("div", { class: "t-help", text: t(`field.${f.key}.help`) }),
      dl);
  };
}
const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
const domId = (key) => `f-${key.replace(/[^a-zA-Z0-9]/g, "-")}`;

function control(f, values, set) {
  const id = domId(f.key);
  const v = values[f.key];
  if (f.type === "bool") {
    const el = h("input", { type: "checkbox", id });
    el.checked = !!v;
    el.addEventListener("change", () => set(el.checked));
    return el;
  }
  if (f.type === "choice") {
    const el = h("select", { id }, f.choices.map((c, i) => h("option", { value: String(i), text: choiceLabel(c, f) })));
    const index = f.choices.findIndex((c) => same(c, v));
    el.value = String(index >= 0 ? index : 0);
    el.addEventListener("change", () => set(f.choices[Number(el.value)]));
    return el;
  }
  if (f.type === "multi") {
    const picked = new Set((Array.isArray(v) ? v : []).map((x) => JSON.stringify(x)));
    const box = h("div", { class: "chips", role: "group", id });
    const inputs = f.choices.map((c) => {
      const input = h("input", { type: "checkbox" });
      input.checked = picked.has(JSON.stringify(c));
      input.addEventListener("change", () => set(f.choices.filter((x, j) => inputs[j].checked)));
      box.append(h("label", null, input, choiceLabel(c, f)));
      return input;
    });
    return box;
  }
  if (f.type === "int" || f.type === "float") {
    const el = h("input", { type: "number", id, inputmode: f.type === "int" ? "numeric" : "decimal",
      min: f.min, max: f.max, step: f.step ?? (f.type === "int" ? 1 : "any"),
      placeholder: f.nullable || f.default === null ? t("tip.auto_empty") : null });
    el.value = v === null || v === undefined ? "" : String(v);
    el.addEventListener("input", () => set(el.value.trim() === "" ? null : Number(el.value)));
    return el;
  }
  const paths = !!(f.pick && f.pick.multiple);
  const multiline = TEXTAREA.has(f.key) || paths;
  const el = h(multiline ? "textarea" : "input", { id, type: multiline ? null : "text", spellcheck: "false",
    autocomplete: "off", class: paths ? "paths" : null,
    placeholder: paths ? t("picker.placeholder_multi") : (f.type === "path" ? t("ui.path") : (f.nullable ? t("tip.auto_empty") : null)) });
  el.value = v ?? "";
  const fit = () => { if (paths) el.rows = Math.max(1, Math.min(8, el.value.split("\n").length)); };
  fit();
  el.addEventListener("input", () => { fit(); set(el.value); });
  return el;
}

export function renderField(f, values, onChange) {
  const wide = WIDE.has(f.key) || (f.type === "multi" && f.choices.length > 4) || !!(f.pick && f.pick.multiple);
  const single = !!(f.pick && !f.pick.multiple);
  const wrap = h("div", { class: `field${f.type === "bool" ? " bool" : ""}${wide ? " wide" : ""}${single && !wide ? " path" : ""}`,
    dataset: { key: f.key } });
  const ctl = control(f, values, (x) => {
    values[f.key] = x;
    wrap.classList.toggle("changed", !same(x, f.default));
    wrap.classList.remove("bad");
    ctl.removeAttribute("aria-invalid");
    if (onChange) onChange(f.key, x);
  });
  const rec = recommendedText(f);
  const label = h(f.type === "multi" ? "span" : "label", { class: "label", for: f.type === "multi" ? null : domId(f.key) },
    h("span", { class: "name", text: t(`field.${f.key}.label`) }),
    h("span", { class: "help-dot", "aria-hidden": "true", text: "?" }),
    rec && rec.length <= 18 ? h("span", { class: "rec", text: `✓ ${rec}` }) : null);
  if (f.type === "multi") ctl.setAttribute("aria-label", t(`field.${f.key}.label`));
  wrap.classList.toggle("changed", !same(values[f.key], f.default));
  wrap.append(label, f.pick ? pathControl(ctl, f.pick) : ctl);
  setTip(wrap, fieldTip(f));
  return wrap;
}

// One schema group as a collapsible section; advanced fields behind a toggle.
export function renderGroup(scope, group, values, onChange, { open = true, summary = true } = {}) {
  const all = fieldsFor(scope, group);
  const basic = all.filter((f) => !f.advanced);
  const adv = all.filter((f) => f.advanced);
  const key = `${scope}:${group}`;
  const grid = h("div", { class: "fields" }, basic.map((f) => renderField(f, values, onChange)));
  const advGrid = h("div", { class: "fields", hidden: !showAdvanced[key] }, adv.map((f) => renderField(f, values, onChange)));
  const toggle = adv.length ? h("button", { type: "button", class: "ghost adv-toggle", "aria-expanded": String(!!showAdvanced[key]),
    text: `${t(showAdvanced[key] ? "ui.hide_advanced" : "ui.show_advanced")} (${adv.length})`,
    onclick: () => {
      showAdvanced[key] = !showAdvanced[key];
      advGrid.hidden = !showAdvanced[key];
      toggle.setAttribute("aria-expanded", String(showAdvanced[key]));
      toggle.textContent = `${t(showAdvanced[key] ? "ui.hide_advanced" : "ui.show_advanced")} (${adv.length})`;
    } }) : null;
  if (adv.length) advGrid.classList.add("mt");
  const body = [grid, toggle, advGrid];
  if (!summary) return h("div", null, body);
  return h("details", { class: "group", open }, h("summary", { text: t(`group.${group}`) }), body);
}

export function markErrors(root, keys, { scroll = true } = {}) {
  for (const el of root.querySelectorAll(".field.bad")) el.classList.remove("bad");
  let first = null;
  for (const raw of keys || []) {
    const key = ERROR_ALIASES[raw] || raw;
    const wrap = root.querySelector(`.field[data-key="${CSS.escape(key)}"]`);
    if (!wrap) continue;
    wrap.classList.add("bad");
    const ctl = wrap.querySelector("input, select, textarea, .chips");
    if (ctl) ctl.setAttribute("aria-invalid", "true");
    first = first || wrap;
  }
  if (first && scroll) {
    const details = first.closest("details");
    if (details) details.open = true;
    first.scrollIntoView({ block: "center", behavior: "smooth" });
  }
}

// Replace a form's values (preset / loaded file / plan) without losing unknown keys.
export function assign(values, patch) {
  for (const [key, value] of Object.entries(patch || {})) values[key] = Array.isArray(value) ? [...value] : value;
  return values;
}

// Run a form action: clears old marks, marks the fields a 422 names, reports other errors.
export async function submit(btn, root, fn) {
  if (btn) btn.disabled = true;
  try {
    markErrors(root, [], { scroll: false });
    return await fn();
  } catch (err) {
    if (err instanceof ApiError && err.status === 422 && err.fields.length) markErrors(root, err.fields);
    reportError(err);
    return undefined;
  } finally {
    if (btn) btn.disabled = false;
  }
}
