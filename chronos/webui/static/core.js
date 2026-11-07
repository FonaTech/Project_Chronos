// Shared runtime: API calls, language, translation, DOM helpers, tooltip, toasts.
// All text reaches the page through textContent; no HTML strings are ever parsed.

export const LANGS = ["zh-Hans", "zh-Hant", "en", "ja"];
const LANG_KEY = "chronos.ui.lang";
const THEME_KEY = "chronos.ui.theme";
const HANT = new Set(["zh-tw", "zh-hk", "zh-mo", "zh-hant"]);

export const state = {
  lang: "en", messages: {}, fallback: {}, schema: null, fields: {}, session: null,
  values: {},          // per-form values kept across re-renders (language switch, navigation)
  config: null,        // the model config shared by Config / Train / Inference / Auto-tune
};

// Same mapping as server.resolve_language: first supported tag wins, English otherwise.
export function resolveLanguage(tags) {
  for (const raw of tags || []) {
    const code = String(raw || "").trim().toLowerCase().replace(/_/g, "-");
    if (!code) continue;
    if (HANT.has(code) || code.startsWith("zh-hant")) return "zh-Hant";
    if (code === "zh" || code.startsWith("zh-")) return "zh-Hans";
    if (code === "ja" || code.startsWith("ja-")) return "ja";
    if (code === "en" || code.startsWith("en-")) return "en";
  }
  return "en";
}

export function preferredLanguage() {
  const saved = localStorage.getItem(LANG_KEY);
  if (LANGS.includes(saved)) return saved;
  return resolveLanguage(navigator.languages && navigator.languages.length ? navigator.languages : [navigator.language]);
}

export class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === "string" ? detail : (detail && (detail.message || detail.error)) || `HTTP ${status}`);
    this.status = status;
    this.detail = detail;
    this.fields = (detail && Array.isArray(detail.fields)) ? detail.fields : [];
    this.code = (detail && typeof detail === "object" && detail.error) || (typeof detail === "string" ? detail : "");
  }
}

let onUnauthorized = () => {};
export function setUnauthorizedHandler(fn) { onUnauthorized = fn; }

export async function api(path, { method = "GET", body, signal } = {}) {
  const headers = { "X-Chronos-Request": "1", Accept: "application/json" };
  const init = { method, headers, credentials: "same-origin", signal };
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const res = await fetch(path, init);
  let data = null;
  try { data = await res.json(); } catch { data = null; }
  if (res.status === 401 && !path.startsWith("/api/login")) onUnauthorized();
  if (!res.ok) throw new ApiError(res.status, data && data.detail !== undefined ? data.detail : data);
  return data;
}
export const post = (path, body = {}) => api(path, { method: "POST", body });

// ── translation ─────────────────────────────────────────────────────────
export async function loadLanguage(lang) {
  const code = LANGS.includes(lang) ? lang : "en";
  const [main, en] = await Promise.all([
    api(`/api/i18n/${code}`),
    code === "en" ? null : (Object.keys(state.fallback).length ? { messages: state.fallback } : api("/api/i18n/en")),
  ]);
  state.lang = code;
  state.messages = main.messages || {};
  state.fallback = code === "en" ? state.messages : (en.messages || {});
  document.documentElement.lang = code;
  return code;
}

export function saveLanguage(lang) { localStorage.setItem(LANG_KEY, lang); }

export function has(key) { return key in state.messages || key in state.fallback; }

export function t(key, vars) {
  let text = state.messages[key];
  if (text === undefined) text = state.fallback[key];
  if (text === undefined) text = key;
  if (vars) text = text.replace(/\{(\w+)\}/g, (m, name) => (name in vars ? String(vars[name]) : m));
  return text;
}

export function translateStatic(root = document) {
  for (const node of root.querySelectorAll("[data-i18n]")) node.textContent = t(node.dataset.i18n);
  for (const node of root.querySelectorAll("[data-i18n-label]")) node.setAttribute("aria-label", t(node.dataset.i18nLabel));
  document.title = t("app.title");
}
// ── DOM (text only) ─────────────────────────────────────────────────────
const tips = new WeakMap();

export function h(tag, attrs, ...children) {
  const el = tag === "svg" || attrs?.svg ? document.createElementNS("http://www.w3.org/2000/svg", tag)
    : document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false || key === "svg") continue;
    if (key === "class") el.setAttribute("class", value);
    else if (key === "text") el.textContent = value;
    else if (key === "tip") setTip(el, value);
    else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key in el && !(el instanceof SVGElement) && typeof value !== "string") el[key] = value;
    else el.setAttribute(key, value === true ? "" : String(value));
  }
  append(el, children);
  return el;
}

export function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export function clear(el) { while (el.firstChild) el.firstChild.remove(); return el; }

export function card(title, ...body) { return h("section", { class: "card" }, title ? h("h3", { text: title }) : null, ...body); }

export function badge(stateName) {
  const key = `state.${stateName}`;
  return h("span", { class: `badge ${stateName || ""}`, role: "status", text: has(key) ? t(key) : String(stateName ?? "") });
}

export function table(columns, rows, { caption } = {}) {
  // columns: [{label, num, fmt(row) -> text|Node}]
  const head = h("tr", null, columns.map((c) => h("th", { class: c.num ? "num" : null, scope: "col", text: c.label })));
  const body = rows.map((row) => h("tr", null, columns.map((c) => {
    const value = c.fmt ? c.fmt(row) : row[c.key];
    return h("td", { class: c.num ? "num" : null }, value === null || value === undefined || value === "" ? "–" : value);
  })));
  return h("div", { class: "table-wrap" },
    h("table", null, caption ? h("caption", { class: "skip", text: caption }) : null, h("thead", null, head), h("tbody", null, body)));
}

// ── number / time formatting in the page language ───────────────────────
const nf = (opts) => new Intl.NumberFormat(state.lang, opts);
export function num(v, digits = 2) {
  if (v === null || v === undefined || v === "" || Number.isNaN(Number(v))) return "–";
  const x = Number(v);
  if (x !== 0 && Math.abs(x) < 10 ** -digits) return x.toExponential(1);
  return nf({ maximumFractionDigits: digits }).format(x);
}
export const int = (v) => num(v, 0);
export const pct = (v, digits = 1) => (v === null || v === undefined ? "–" : nf({ style: "percent", maximumFractionDigits: digits }).format(Number(v)));
export function params(v) {
  if (v === null || v === undefined) return "–";
  const x = Number(v);
  if (Math.abs(x) >= 1e9) return `${num(x / 1e9, 2)}B`;
  if (Math.abs(x) >= 1e6) return `${num(x / 1e6, 2)}M`;
  if (Math.abs(x) >= 1e3) return `${num(x / 1e3, 1)}K`;
  return int(x);
}
export function bytes(v) {
  if (v === null || v === undefined) return "–";
  const x = Number(v);
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0; let y = Math.abs(x);
  while (y >= 1024 && i < units.length - 1) { y /= 1024; i += 1; }
  return `${num(Math.sign(x) * y, y >= 100 ? 0 : 2)} ${units[i]}`;
}
export function duration(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(Number(seconds))) return "–";
  let s = Math.max(0, Math.round(Number(seconds)));
  const parts = { hours: Math.floor(s / 3600), minutes: Math.floor((s % 3600) / 60), seconds: s % 60 };
  if (parts.hours) delete parts.seconds;
  if (typeof Intl.DurationFormat === "function") {
    try { return new Intl.DurationFormat(state.lang, { style: "narrow" }).format(parts); } catch { /* fall through */ }
  }
  s = parts.hours ? `${parts.hours}h ${parts.minutes}m` : `${parts.minutes}m ${parts.seconds}s`;
  return s;
}
export function dateTime(v) {
  if (!v) return "–";
  const d = typeof v === "number" ? new Date(v * 1000) : new Date(v);
  return Number.isNaN(d.getTime()) ? String(v) : new Intl.DateTimeFormat(state.lang, { dateStyle: "short", timeStyle: "short" }).format(d);
}
// ── tooltip: one element ────────────────────────────────────────────────
// Hover shows a field's help after the pointer rests on it for HOVER_MS, so moving across a form
// never covers the next fields. The "?" dot is an explicit request and answers after HELP_MS.
// Keyboard focus (Tab) shows it at once; a click, typing, Escape or scrolling hides it.
const HOVER_MS = 2000;
const HELP_MS = 400;
let tipEl = null;
let tipOwner = null;
let tipTimer = 0;
let tipPending = null;
let viaKeyboard = false;
let suppressed = null;

function cancelTip() {
  clearTimeout(tipTimer);
  tipTimer = 0;
  tipPending = null;
}

function scheduleTip(el, ms) {
  if (tipOwner === el || suppressed === el) return;
  if (tipPending === el && ms >= HOVER_MS) return;
  cancelTip();
  tipPending = el;
  tipTimer = setTimeout(() => { tipPending = null; if (el.isConnected) showTip(el); }, ms);
}

export function setTip(el, content) {
  tips.set(el, content);
  if (el.dataset.tipBound) return;
  el.dataset.tipBound = "1";
  el.addEventListener("pointerenter", (e) => { if (e.pointerType !== "touch") scheduleTip(el, HOVER_MS); });
  el.addEventListener("pointerover", (e) => {
    if (e.pointerType !== "touch" && e.target instanceof Element && e.target.closest(".help-dot")) scheduleTip(el, HELP_MS);
  });
  el.addEventListener("pointerleave", () => {
    if (tipPending === el) cancelTip();
    if (suppressed === el) suppressed = null;
    hideTip(el);
  });
  el.addEventListener("pointerdown", (e) => {
    cancelTip();
    if (e.target instanceof Element && e.target.closest(".help-dot")) { suppressed = null; showTip(el); return; }
    suppressed = el;
    hideTip();
  });
  el.addEventListener("focusin", () => { if (viaKeyboard) { cancelTip(); showTip(el); } });
  el.addEventListener("focusout", (e) => { if (!el.contains(e.relatedTarget)) hideTip(el); });
}

function fillTip(content) {
  clear(tipEl);
  const value = typeof content === "function" ? content() : content;
  if (value instanceof Node) tipEl.append(value);
  else tipEl.textContent = String(value ?? "");
}

function showTip(el) {
  tipEl = tipEl || document.getElementById("tooltip");
  const content = tips.get(el);
  if (!tipEl || content === undefined) return;
  tipOwner = el;
  fillTip(content);
  tipEl.hidden = false;
  el.setAttribute("aria-describedby", "tooltip");
  const r = el.getBoundingClientRect();
  const w = tipEl.offsetWidth; const hgt = tipEl.offsetHeight;
  let left = Math.min(window.innerWidth - w - 8, Math.max(8, r.left));
  let top = r.bottom + 8;
  if (top + hgt > window.innerHeight - 8) top = Math.max(8, r.top - hgt - 8);
  if (left < 8) left = 8;
  tipEl.style.left = `${left}px`;
  tipEl.style.top = `${top}px`;
}

export function hideTip(el) {
  if (!el) cancelTip();
  if (!tipEl || (el && el !== tipOwner)) return;
  tipEl.hidden = true;
  if (tipOwner) tipOwner.removeAttribute("aria-describedby");
  tipOwner = null;
}

// Chart crosshair tooltips are positioned at a point instead of an element.
export function showTipAt(x, y, content) {
  tipEl = tipEl || document.getElementById("tooltip");
  if (!tipEl) return;
  cancelTip();
  tipOwner = null;
  fillTip(content);
  tipEl.hidden = false;
  const w = tipEl.offsetWidth; const hgt = tipEl.offsetHeight;
  let left = x + 14; let top = y + 14;
  if (left + w > window.innerWidth - 8) left = x - w - 14;
  if (top + hgt > window.innerHeight - 8) top = y - hgt - 14;
  tipEl.style.left = `${Math.max(8, left)}px`;
  tipEl.style.top = `${Math.max(8, top)}px`;
}

const MODIFIERS = new Set(["Shift", "Control", "Alt", "Meta", "CapsLock"]);
document.addEventListener("keydown", (e) => {
  if (MODIFIERS.has(e.key)) return;
  viaKeyboard = e.key === "Tab";
  if (!viaKeyboard) hideTip();
}, true);
document.addEventListener("pointerdown", () => { viaKeyboard = false; }, true);
document.addEventListener("input", () => hideTip(), true);
window.addEventListener("scroll", () => hideTip(), { passive: true, capture: true });

// ── toasts and errors ───────────────────────────────────────────────────
export function toast(message, kind = "info") {
  const host = document.getElementById("toasts");
  if (!host) return;
  const el = h("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status", text: message });
  host.append(el);
  setTimeout(() => el.remove(), kind === "error" ? 9000 : 4500);
}

export function errorText(err) {
  if (err instanceof ApiError) {
    // Backend detail messages are machine text (English); they are appended, never used as the only text.
    const extra = err.detail && typeof err.detail === "object" && err.detail.message ? ` (${err.detail.message})` : "";
    if (err.code && has(`err.${err.code}`)) return t(`err.${err.code}`) + extra;
    if (err.status === 409) return t("ui.busy");
    if (err.status === 422 && err.fields.length) return t("ui.invalid") + extra;
    if (err.status === 404) return t("err.not_found");
  }
  return `${t("ui.error")}: ${err && err.message ? err.message : String(err)}`;
}

export function reportError(err) {
  if (err instanceof ApiError && err.status === 401) return;
  console.error(err);
  toast(errorText(err), "error");
}

// ── theme ───────────────────────────────────────────────────────────────
export function savedTheme() {
  const v = localStorage.getItem(THEME_KEY);
  return ["light", "dark", "system"].includes(v) ? v : "system";
}
export function applyTheme(value) {
  document.documentElement.dataset.theme = value;
  localStorage.setItem(THEME_KEY, value);
}

// CSS custom property in the active theme (charts read series colors this way).
export function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
export const SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"];

// ── page furniture ──────────────────────────────────────────────────────
export function pageHead(titleKey, introKey, ...actions) {
  return h("div", { class: "page-head" },
    h("div", null, h("h2", { text: t(titleKey) }), introKey ? h("p", { text: t(introKey) }) : null),
    actions.length ? h("div", { class: "row" }, actions) : null);
}

export function tile(label, value, sub, tip) {
  return h("div", { class: "tile", tip }, h("div", { class: "k", text: label }), h("div", { class: "v", text: value ?? "–" }),
    sub ? h("div", { class: "s", text: sub }) : null);
}

export function fieldLabel(key) { return has(`field.${key}.label`) ? t(`field.${key}.label`) : key; }

export function button(labelKey, onclick, { primary = false, danger = false, disabled = false } = {}) {
  return h("button", { type: "button", class: primary ? "primary" : danger ? "danger" : null, disabled, text: t(labelKey), onclick });
}

// Run an async action with the button disabled; errors become toasts.
export async function busy(btn, fn) {
  if (btn) btn.disabled = true;
  try { return await fn(); } catch (err) { reportError(err); return undefined; } finally { if (btn) btn.disabled = false; }
}

export function logBox(lines) {
  const text = Array.isArray(lines) ? lines.join("\n") : String(lines || "");
  const pre = h("pre", { class: "log", tabindex: "0", text: text || t("ui.empty") });
  requestAnimationFrame(() => { pre.scrollTop = pre.scrollHeight; });
  return pre;
}

export function rawJson(value) {
  return h("details", null, h("summary", { text: t("ui.raw_json") }), h("pre", { class: "log", text: JSON.stringify(value, null, 2) }));
}

export const clone = (v) => (v === undefined ? v : JSON.parse(JSON.stringify(v)));
