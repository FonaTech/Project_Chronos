// Entry point: session, language, theme, navigation, page lifecycle and polling.
import { state, t, h, api, post, loadLanguage, saveLanguage, preferredLanguage, translateStatic, applyTheme, savedTheme,
  setUnauthorizedHandler, reportError, hideTip, clear, setTip } from "./core.js";
import { renderConfig } from "./pages_config.js";
import { renderData, renderTokenizer } from "./pages_data.js";
import { renderTrain } from "./pages_train.js";
import { renderMonitor } from "./pages_monitor.js";
import { renderInference } from "./pages_infer.js";
import { renderExport, renderServe, renderBenchmark, renderAutotune, renderIO } from "./pages_ops.js";

const PAGES = [
  ["config", renderConfig], ["data", renderData], ["tokenizer", renderTokenizer], ["train", renderTrain],
  ["monitor", renderMonitor], ["inference", renderInference], ["export", renderExport], ["serve", renderServe],
  ["benchmark", renderBenchmark], ["autotune", renderAutotune], ["io", renderIO],
];
const ROUTES = new Map(PAGES);
let current = null;      // { id, timer, poll, alive }

function route() {
  const id = location.hash.replace(/^#\/?/, "");
  return ROUTES.has(id) ? id : "config";
}

function renderNav(active) {
  const nav = document.getElementById("nav");
  nav.setAttribute("aria-label", t("ui.sections"));
  nav.replaceChildren(...PAGES.map(([id]) => h("a", { href: `#${id}`, "aria-current": id === active ? "page" : null, text: t(`nav.${id}`) })));
}

function stopPage() {
  if (!current) return;
  current.alive = false;
  clearTimeout(current.timer);
  current = null;
}

// Each page may register one poller: it returns the delay to the next call (0 = stop).
function makePage(id) {
  const page = { id, alive: true, timer: 0, fn: null };
  const tick = async () => {
    if (!page.alive || !page.fn) return;
    clearTimeout(page.timer);
    let delay = 0;
    try { delay = await page.fn(); } catch (err) { reportError(err); delay = 10000; }
    if (page.alive && delay > 0) page.timer = setTimeout(tick, document.hidden ? Math.max(delay, 15000) : delay);
  };
  page.poll = (fn) => { page.fn = fn; tick(); };
  page.refresh = () => { if (page.fn) tick(); };
  return page;
}

async function show({ focus = false } = {}) {
  stopPage();
  hideTip();
  const id = route();
  renderNav(id);
  const main = document.getElementById("main");
  clear(main);
  current = makePage(id);
  const page = current;
  try {
    await ROUTES.get(id)(main, page);
  } catch (err) {
    reportError(err);
    main.append(h("div", { class: "empty", text: t("ui.error") }));
  }
  if (focus && page.alive) main.focus({ preventScroll: true });
}

document.addEventListener("chronos:refresh", () => { if (current) current.refresh(); });
document.addEventListener("visibilitychange", () => { if (!document.hidden && current) current.refresh(); });
window.addEventListener("hashchange", () => show({ focus: true }));
// ── session / login ────────────────────────────────────────────────────
function paintSession() {
  const s = state.session || {};
  const badgeEl = document.getElementById("auth-badge");
  // local_only: loopback, LAN off · lan_local: this machine while LAN is on · lan: another machine, signed in
  const mode = s.access === "lan" ? "lan" : (s.lan_enabled ? "lan_local" : "local_only");
  badgeEl.className = `badge ${mode === "lan_local" ? "warning" : (mode === "lan" ? "good" : "")}`;
  badgeEl.textContent = t({ local_only: "ui.local_only", lan_local: "ui.lan_local", lan: "ui.remote_auth" }[mode]);
  setTip(badgeEl, t(`ui.access_tip.${mode}`));
  document.getElementById("logout").hidden = !(s.auth_required && s.authenticated);
}

let loginOpen = false;
function askLogin() {
  const dialog = document.getElementById("login-dialog");
  if (loginOpen) return;
  loginOpen = true;
  stopPage();
  document.getElementById("login-error").hidden = true;
  dialog.showModal();
  document.getElementById("login-key").focus();
}

async function onLogin(event) {
  event.preventDefault();
  const input = document.getElementById("login-key");
  const error = document.getElementById("login-error");
  try {
    await post("/api/login", { key: input.value });
    input.value = "";
    loginOpen = false;
    document.getElementById("login-dialog").close();
    state.session = await api("/api/session");
    paintSession();
    await start();
  } catch (err) {
    error.textContent = err.status === 429 ? t("ui.login_blocked") : t("ui.login_failed");
    error.hidden = false;
  }
}

async function onLogout() {
  try { await post("/api/logout"); } catch (err) { reportError(err); }
  state.session = await api("/api/session");
  paintSession();
  askLogin();
}

// ── language / theme ───────────────────────────────────────────────────
async function setLanguage(lang) {
  await loadLanguage(lang);
  saveLanguage(state.lang);
  document.getElementById("lang-select").value = state.lang;
  translateStatic();
  paintSession();
  if (state.schema) await show();
}

let started = false;
async function start() {
  if (!state.schema) {
    state.schema = await api("/api/schema");
    state.fields = Object.fromEntries(state.schema.fields.map((f) => [f.key, f]));
  }
  started = true;
  await show();
}

async function boot() {
  applyTheme(savedTheme());
  const theme = document.getElementById("theme-select");
  theme.value = savedTheme();
  theme.addEventListener("change", () => applyTheme(theme.value));
  const lang = document.getElementById("lang-select");
  lang.addEventListener("change", () => setLanguage(lang.value).catch(reportError));
  document.getElementById("login-form").addEventListener("submit", onLogin);
  document.getElementById("login-dialog").addEventListener("cancel", (e) => e.preventDefault());
  document.getElementById("logout").addEventListener("click", onLogout);
  setUnauthorizedHandler(askLogin);

  await loadLanguage(preferredLanguage());
  lang.value = state.lang;
  translateStatic();
  state.session = await api("/api/session");
  paintSession();
  if (state.session.auth_required && !state.session.authenticated) { askLogin(); return; }
  await start();
}

boot().catch((err) => {
  reportError(err);
  if (!started) document.getElementById("main").append(h("div", { class: "empty", text: String(err.message || err) }));
});
