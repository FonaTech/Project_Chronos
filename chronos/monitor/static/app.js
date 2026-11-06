/* Chronos training monitor page. No dependencies; strict-CSP safe (no inline
 * script/style, all text through textContent, SVG built with createElementNS). */
"use strict";
(() => {
const SVG = "http://www.w3.org/2000/svg";
const POLL_MS = 15000;
const $ = (id) => document.getElementById(id);

// ---- i18n ------------------------------------------------------------------
const STR = {
  zh: {
    title: "训练监控", run: "运行", logout: "退出登录", theme_light: "浅色", theme_dark: "深色", lang: "English",
    login_title: "登录训练监控", login_hint: "此监控页受 API 密钥保护。会话保存在 HttpOnly Cookie 中，密钥不会写入页面。",
    key: "API 密钥", login: "登录", login_bad: "密钥不正确", login_wait: "失败次数过多，请 5 分钟后再试",
    auth_local: "仅本机 · 无认证", auth_key: "已登录 · API 密钥",
    st_running: "训练中", st_paused: "已暂停", st_evaluating: "评测中", st_checkpointing: "保存检查点", st_starting: "启动中",
    st_finished: "已完成", st_failed: "失败", st_stopped: "已停止", st_pending: "等待中",
    n_stale: "最近 {age} 没有新的日志行（正常间隔约 {int}）。训练进程可能卡住或被挂起。",
    n_derived: "此运行由旧版代码启动，没有 status.json / parameters.jsonl。状态由指标流和进程表推断；张量统计、路由热图和评测探针会在下次以新代码恢复（resume）后出现。",
    n_error: "运行报告错误：", n_paused: "训练进程处于 SIGSTOP 挂起状态，恢复后会继续。", n_none: "没有找到运行目录。用 chronos longrun 启动训练后，这里会自动出现。",
    phase: "阶段", step: "步", of: "/", budget: "预算", remaining: "剩余", eta: "预计完成", wall: "已用", elapsed: "已运行",
    ph_complete: "完成", ph_active: "进行中", ph_pending: "未开始", pretrain: "预训练", sft: "SFT",
    k_loss: "训练损失", k_ce: "主损失 (CE)", k_valid: "最佳验证 CE", k_tps: "吞吐", k_eta: "预计完成", k_mem: "内存",
    k_grad: "梯度范数", k_depth: "平均循环深度", k_lr: "学习率", k_route: "路由熵",
    sub_ppl: "困惑度 {v}", sub_at: "第 {s} 步", sub_proc: "含填充 {v} tok/s", sub_footprint: "进程物理占用 (phys_footprint)", sub_peak: "MLX 峰值 {v} · 活跃 {a}",
    sub_clip: "裁剪阈值 {v}", sub_maxdepth: "最大 {v}", sub_entmax: "上限 {v}", sub_left: "剩余 {v}",
    t_overview: "总览", t_loss: "损失分量", t_recurrent: "循环深度", t_routing: "MoE 路由", t_params: "参数统计",
    t_evals: "评测与样例", t_config: "训练参数", t_log: "日志", t_ckpt: "检查点",
    window: "窗口", w_all: "全部", w_last: "最近 {n} 点", phase_all: "全部阶段", events_on: "显示验证/回滚标记",
    c_loss: "训练损失", c_primary: "主损失 (CE)", c_valid: "验证 CE", c_lr: "学习率", c_grad: "梯度范数（裁剪前）",
    c_tps: "吞吐 (tok/s)", c_step: "单步耗时 (s)", c_mem: "MLX 内存 (GB)", c_tokens: "累计 token",
    peak: "峰值", active: "活跃", cache: "缓存", train: "训练", valid: "验证", seen: "有效", processed: "含填充",
    no_data: "暂无数据", no_telemetry: "这个运行还没有张量遥测（parameters.jsonl）。新代码恢复训练后，每 {m} 分钟写一行。",
    rec_caption: "每组实际循环深度、收缩系数和提前退出比例；深度上限来自模型配置。",
    route_caption: "负载份额 = 每个专家被 top-k 选中的比例；熵越接近上限，路由越均匀。",
    heat_load: "专家负载（最近一次遥测）", heat_importance: "平均路由概率", heat_wrms: "专家权重 RMS", heat_grms: "专家梯度 RMS",
    heat_rows: "行 = 层·迭代，列 = 专家", heat_rows_layer: "行 = 物理层，列 = 专家",
    groups: "参数组", tensors: "张量", filter: "筛选张量名…", sort: "排序", name: "名称", group: "组", shape: "形状", dtype: "类型",
    numel: "参数量", rms: "RMS", mean: "均值", std: "标准差", absmax: "|最大|", grad_rms: "梯度 RMS", grad_norm: "梯度范数",
    update_ratio: "更新比 ‖Δ‖/‖W‖", tensors_n: "张量数", hist_of: "{n} 的历史", hist_hint: "点击表格中的张量查看历史曲线。",
    tel_meta: "第 {s} 步 · {t} · 统计耗时 {ms} ms · {note}", load_tensors: "加载全部 {n} 个张量",
    validations: "验证记录", probes: "评测探针", samples: "评测样例", inspections: "离线检查报告", metric: "指标",
    src: "输入", ref: "参考", hyp: "输出", direction: "方向", set: "数据集", bleu: "BLEU", chrf: "chrF", ppl: "困惑度",
    time: "时间", kind: "事件", detail: "详情", ev_validation: "验证", ev_rollback: "回滚", ev_final: "最终评测",
    ev_phase_start: "阶段开始", ev_phase_end: "阶段结束", ev_resume: "恢复",
    cfg_run: "运行配置", cfg_phases: "阶段配方", cfg_model: "模型结构", cfg_runtime: "运行时与精度", cfg_data: "数据分片", cfg_tok: "分词器",
    cfg_raw: "原始 JSON", cfg_missing: "运行未记录 runtime.json（旧版代码启动）。",
    log_more: "加载更多", log_refresh: "刷新", log_auto: "自动刷新",
    ck_file: "文件", ck_size: "大小", ck_modified: "修改时间", ck_age: "距今", ck_none: "还没有检查点。",
    foot: "只读监控 · 每 {s} 秒刷新 · 数据目录 {root} · 生成于 {t}", paused_poll: "页面不可见，已暂停刷新",
    err_net: "无法连接监控服务：", generated: "更新于",
  },
  en: {
    title: "Training monitor", run: "Run", logout: "Sign out", theme_light: "Light", theme_dark: "Dark", lang: "中文",
    login_title: "Sign in to the training monitor", login_hint: "This monitor is protected by an API key. The session lives in an HttpOnly cookie; the key is never stored in the page.",
    key: "API key", login: "Sign in", login_bad: "Wrong key", login_wait: "Too many failed attempts; wait 5 minutes",
    auth_local: "Local only · no auth", auth_key: "Signed in · API key",
    st_running: "Running", st_paused: "Paused", st_evaluating: "Evaluating", st_checkpointing: "Checkpointing", st_starting: "Starting",
    st_finished: "Finished", st_failed: "Failed", st_stopped: "Stopped", st_pending: "Pending",
    n_stale: "No new log line for {age} (usual interval about {int}). The trainer may be stuck or suspended.",
    n_derived: "This run was started by older code and writes no status.json / parameters.jsonl. Status is derived from the metric stream and the process table; tensor stats, routing heatmaps and evaluation probes appear after the next resume with the new code.",
    n_error: "The run reported an error: ", n_paused: "The trainer is suspended (SIGSTOP) and will continue when resumed.", n_none: "No run directories found. Runs started with chronos longrun show up here.",
    phase: "Phase", step: "step", of: "/", budget: "budget", remaining: "remaining", eta: "ETA", wall: "used", elapsed: "Elapsed",
    ph_complete: "complete", ph_active: "active", ph_pending: "pending", pretrain: "Pretrain", sft: "SFT",
    k_loss: "Training loss", k_ce: "Primary loss (CE)", k_valid: "Best valid CE", k_tps: "Throughput", k_eta: "ETA", k_mem: "Memory",
    k_grad: "Grad norm", k_depth: "Mean recurrent depth", k_lr: "Learning rate", k_route: "Routing entropy",
    sub_ppl: "perplexity {v}", sub_at: "step {s}", sub_proc: "{v} tok/s incl. padding", sub_footprint: "process phys_footprint", sub_peak: "MLX peak {v} · active {a}",
    sub_clip: "clip at {v}", sub_maxdepth: "max {v}", sub_entmax: "max {v}", sub_left: "{v} left",
    t_overview: "Overview", t_loss: "Loss terms", t_recurrent: "Recurrence", t_routing: "MoE routing", t_params: "Parameters",
    t_evals: "Evals & samples", t_config: "Training parameters", t_log: "Log", t_ckpt: "Checkpoints",
    window: "Window", w_all: "All", w_last: "Last {n} points", phase_all: "All phases", events_on: "Show validation/rollback marks",
    c_loss: "Training loss", c_primary: "Primary loss (CE)", c_valid: "Validation CE", c_lr: "Learning rate", c_grad: "Grad norm (pre-clip)",
    c_tps: "Throughput (tok/s)", c_step: "Step time (s)", c_mem: "MLX memory (GB)", c_tokens: "Tokens seen",
    peak: "peak", active: "active", cache: "cache", train: "train", valid: "valid", seen: "counted", processed: "incl. padding",
    no_data: "No data yet", no_telemetry: "No tensor telemetry (parameters.jsonl) for this run yet. After a resume with the new code, one row is written every {m} minutes.",
    rec_caption: "Depth actually used per group, contraction and early-exit share; the depth cap comes from the model config.",
    route_caption: "Load share = fraction of top-k picks per expert; entropy close to the cap means even routing.",
    heat_load: "Expert load (latest telemetry)", heat_importance: "Mean router probability", heat_wrms: "Expert weight RMS", heat_grms: "Expert gradient RMS",
    heat_rows: "rows = layer·iteration, columns = experts", heat_rows_layer: "rows = physical layer, columns = experts",
    groups: "Parameter groups", tensors: "Tensors", filter: "Filter tensor names…", sort: "Sort", name: "Name", group: "Group", shape: "Shape", dtype: "dtype",
    numel: "Params", rms: "RMS", mean: "Mean", std: "Std", absmax: "|max|", grad_rms: "Grad RMS", grad_norm: "Grad norm",
    update_ratio: "Update ratio ‖Δ‖/‖W‖", tensors_n: "Tensors", hist_of: "History of {n}", hist_hint: "Click a tensor row to see its history.",
    tel_meta: "step {s} · {t} · stats took {ms} ms · {note}", load_tensors: "Load all {n} tensors",
    validations: "Validations", probes: "Evaluation probes", samples: "Evaluation samples", inspections: "Offline inspections", metric: "Metric",
    src: "Input", ref: "Reference", hyp: "Output", direction: "Direction", set: "Set", bleu: "BLEU", chrf: "chrF", ppl: "PPL",
    time: "Time", kind: "Event", detail: "Detail", ev_validation: "validation", ev_rollback: "rollback", ev_final: "final eval",
    ev_phase_start: "phase start", ev_phase_end: "phase end", ev_resume: "resume",
    cfg_run: "Run", cfg_phases: "Phase recipes", cfg_model: "Model architecture", cfg_runtime: "Runtime & precision", cfg_data: "Data shards", cfg_tok: "Tokenizer",
    cfg_raw: "Raw JSON", cfg_missing: "No runtime.json for this run (started by older code).",
    log_more: "Load more", log_refresh: "Refresh", log_auto: "Auto refresh",
    ck_file: "File", ck_size: "Size", ck_modified: "Modified", ck_age: "Age", ck_none: "No checkpoints yet.",
    foot: "Read-only monitor · refresh every {s} s · runs dir {root} · generated {t}", paused_poll: "Page hidden; polling paused",
    err_net: "Cannot reach the monitor service: ", generated: "updated",
  },
};
// Metric labels.  recurrent_* = weighted term inside the total loss; rec_* = raw value.
const METRIC = {
  zh: { loss: "总损失", primary_loss: "主损失 (CE)", balance_loss: "负载均衡（加权）", lookahead_loss: "lookahead（加权）",
    lookahead_topk_loss: "lookahead top-k（加权）", lookahead_union_loss: "lookahead 并集（加权）", temporal_loss: "时序一致（加权）",
    locality_loss: "路由局部性（加权）", anchor_loss: "路由锚定（加权）", recurrent_ponder_loss: "ponder（加权）",
    recurrent_distill_loss: "深度蒸馏（加权）", recurrent_stability_loss: "稳定性（加权）", recurrent_contraction_loss: "收缩约束（加权）",
    rec_depth_mean: "平均深度", rec_contraction: "收缩系数", rec_early_exit_ratio: "提前退出比例", rec_ponder_loss: "ponder（原始）",
    rec_distill_loss: "深度蒸馏（原始）", rec_stability_loss: "稳定性（原始）", route_entropy: "路由熵", load_cv: "负载变异系数",
    dead_experts: "空闲专家数", max_expert_load: "最大专家负载（均值倍数）" },
  en: { loss: "Total loss", primary_loss: "Primary loss (CE)", balance_loss: "Load balance (weighted)", lookahead_loss: "Lookahead (weighted)",
    lookahead_topk_loss: "Lookahead top-k (weighted)", lookahead_union_loss: "Lookahead union (weighted)", temporal_loss: "Temporal (weighted)",
    locality_loss: "Router locality (weighted)", anchor_loss: "Router anchor (weighted)", recurrent_ponder_loss: "Ponder (weighted)",
    recurrent_distill_loss: "Depth distill (weighted)", recurrent_stability_loss: "Stability (weighted)", recurrent_contraction_loss: "Contraction (weighted)",
    rec_depth_mean: "Mean depth", rec_contraction: "Contraction", rec_early_exit_ratio: "Early-exit share", rec_ponder_loss: "Ponder (raw)",
    rec_distill_loss: "Depth distill (raw)", rec_stability_loss: "Stability (raw)", route_entropy: "Routing entropy", load_cv: "Load CV",
    dead_experts: "Idle experts", max_expert_load: "Max expert load (× mean)" },
};

// ---- state -------------------------------------------------------------------
const S = {
  lang: localStorage.getItem("chronos.monitor.lang") || ((navigator.language || "").startsWith("zh") ? "zh" : "en"),
  theme: localStorage.getItem("chronos.monitor.theme") || "",
  tab: localStorage.getItem("chronos.monitor.tab") || "overview",
  run: null, snap: null, session: null, timer: null, inflight: false, error: null,
  window: 0, phase: "", showEvents: true,
  params: null, tensors: null, tensorFilter: "", tensorSort: "name", tensorPick: null, tensorHist: null,
  log: null, logBytes: 16000, logAuto: true,
};
const tr = (key, vars) => {
  let s = (STR[S.lang] && STR[S.lang][key]) ?? STR.en[key] ?? key;
  if (vars) for (const [k, v] of Object.entries(vars)) s = s.split(`{${k}}`).join(String(v));
  return s;
};
const mlabel = (key) => (METRIC[S.lang] && METRIC[S.lang][key]) || METRIC.en[key] || key;

// ---- DOM helpers ---------------------------------------------------------------
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return el;
}
function s(tag, attrs) {
  const el = document.createElementNS(SVG, tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, v);
  return el;
}
const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };

// ---- formatting ----------------------------------------------------------------
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
function fmt(v, digits = 4) {
  if (!isNum(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-4 || a >= 1e6)) return v.toExponential(2);
  return Number(v.toPrecision(digits)).toLocaleString(S.lang === "zh" ? "zh-CN" : "en-US", { maximumFractionDigits: 6 });
}
function fmtInt(v) { return isNum(v) ? Math.round(v).toLocaleString(S.lang === "zh" ? "zh-CN" : "en-US") : "—"; }
function fmtCount(v) {
  if (!isNum(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(2) + "B";
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e8 ? 0 : 1) + "M";
  if (a >= 1e3) return (v / 1e3).toFixed(1) + "k";
  return String(Math.round(v));
}
function fmtGB(v) { return isNum(v) ? (v >= 100 ? v.toFixed(0) : v.toFixed(2)) + " GB" : "—"; }
function fmtDur(sec) {
  if (!isNum(sec)) return "—";
  sec = Math.max(0, sec);
  const d = Math.floor(sec / 86400), hh = Math.floor((sec % 86400) / 3600), mm = Math.floor((sec % 3600) / 60);
  if (S.lang === "zh") return d ? `${d}天${hh}时` : hh ? `${hh}时${mm}分` : mm ? `${mm}分` : `${Math.round(sec)}秒`;
  return d ? `${d}d ${hh}h` : hh ? `${hh}h ${mm}m` : mm ? `${mm}m` : `${Math.round(sec)}s`;
}
function fmtTime(stamp) {
  if (!stamp) return "—";
  const d = typeof stamp === "number" ? new Date(stamp * 1000) : new Date(stamp);
  if (Number.isNaN(d.getTime())) return String(stamp);
  return d.toLocaleString(S.lang === "zh" ? "zh-CN" : "en-US", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false });
}
const phaseName = (p) => (p === "pretrain" || p === "sft") ? tr(p) : (p || "—");
// ---- API -------------------------------------------------------------------------
class AuthError extends Error {}
const url = (path, params) => {
  const u = new URL(path, document.baseURI);            // relative: works at / and when mounted at /monitor/
  if (params) for (const [k, v] of Object.entries(params)) if (v != null && v !== "") u.searchParams.set(k, v);
  return u;
};
async function api(path, params, text = false) {
  const res = await fetch(url(path, params), { credentials: "same-origin", headers: { Accept: text ? "text/plain" : "application/json" } });
  if (res.status === 401) throw new AuthError("401");
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); if (j && j.detail) msg = String(j.detail); } catch (_) { /* not JSON */ }
    throw new Error(msg);
  }
  return text ? res.text() : res.json();
}

// ---- session / login ---------------------------------------------------------------
function showLogin(message) {
  stop();
  $("login").hidden = false;
  $("root").setAttribute("inert", "");
  $("login-title").textContent = tr("login_title");
  $("login-hint").textContent = tr("login_hint");
  $("key-label").textContent = tr("key");
  $("login-submit").textContent = tr("login");
  $("login-error").textContent = message || "";
  $("key").focus();
}
function hideLogin() { $("login").hidden = true; $("root").removeAttribute("inert"); }
async function submitLogin(ev) {
  ev.preventDefault();
  const btn = $("login-submit");
  btn.disabled = true;
  try {
    const res = await fetch(url("api/login"), { method: "POST", credentials: "same-origin",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key: $("key").value }) });
    if (res.status === 429) { $("login-error").textContent = tr("login_wait"); return; }
    if (!res.ok) { $("login-error").textContent = tr("login_bad"); return; }
    $("key").value = "";
    hideLogin();
    await start();
  } catch (e) {
    $("login-error").textContent = tr("err_net") + e.message;
  } finally { btn.disabled = false; }
}
async function logout() {
  try { await fetch(url("api/logout"), { method: "POST", credentials: "same-origin" }); } catch (_) { /* offline */ }
  S.snap = S.params = S.tensors = S.tensorHist = S.log = null;
  showLogin("");
}

// ---- polling ---------------------------------------------------------------------
function stop() { clearTimeout(S.timer); S.timer = null; }
function schedule() {
  stop();
  if (!document.hidden) S.timer = setTimeout(async () => { await refresh(); schedule(); }, POLL_MS);
}
async function refresh() {
  if (S.inflight) return;
  S.inflight = true;
  try {
    const snap = await api("api/state", { run: S.run, history: 1 });
    S.snap = snap;
    S.error = null;
    if (snap.run) S.run = snap.run;
    if (S.tab === "log" && S.logAuto) await loadLog();
    renderAll(false);
  } catch (e) {
    if (e instanceof AuthError) showLogin("");
    else { S.error = e.message; renderNotices(); }
  } finally { S.inflight = false; }
}
async function start() {
  try { S.session = await api("api/session"); } catch (e) { S.error = e.message; renderNotices(); return; }
  if (S.session.auth_required && !S.session.authenticated) { renderChrome(); showLogin(""); return; }
  await refresh();
  schedule();
}

// ---- header, notices, progress ------------------------------------------------------
const effectiveDark = () => S.theme ? S.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
function renderChrome() {
  document.documentElement.lang = S.lang === "zh" ? "zh-CN" : "en";
  if (S.theme) document.documentElement.dataset.theme = S.theme; else delete document.documentElement.dataset.theme;
  $("title").textContent = tr("title");
  document.title = `Chronos · ${tr("title")}${S.snap && S.snap.run ? " · " + S.snap.run : ""}`;
  $("run-label").textContent = tr("run");
  $("lang").textContent = tr("lang");
  $("theme").textContent = effectiveDark() ? tr("theme_light") : tr("theme_dark");
  $("logout").textContent = tr("logout");
  $("logout").hidden = !(S.session && S.session.auth_required);
  $("auth-badge").textContent = S.session ? (S.session.auth_required ? tr("auth_key") : tr("auth_local")) : "";
  const sel = $("run");
  const names = ((S.snap && S.snap.runs) || []).map((r) => r.name);
  if (sel.dataset.names !== names.join("\n")) {
    clear(sel);
    for (const n of names) sel.append(h("option", { value: n, text: n }));
    sel.dataset.names = names.join("\n");
  }
  if (S.snap && S.snap.run) sel.value = S.snap.run;
  const st = (S.snap && S.snap.status) || { state: "pending" };
  const pill = clear($("state-pill"));
  pill.className = `pill ${st.state}`;
  pill.append(h("span", { class: "dot", "aria-hidden": "true" }), h("span", { text: tr("st_" + st.state) + (st.stale ? " · !" : "") }));
}
function renderNotices() {
  const box = clear($("notices"));
  const add = (kind, icon, text) => box.append(h("div", { class: `notice ${kind}`, role: kind === "error" ? "alert" : null },
    h("span", { class: "icon", "aria-hidden": "true", text: icon }), h("span", { text })));
  if (S.error) add("error", "✕", tr("err_net") + S.error);
  const snap = S.snap;
  if (!snap) return;
  if (!snap.run) { add("info", "i", tr("n_none")); return; }
  const st = snap.status || {};
  if (st.error) add("error", "✕", tr("n_error") + st.error);
  if (st.state === "paused") add("warn", "‖", tr("n_paused"));
  if (st.stale) add("warn", "!", tr("n_stale", { age: fmtDur(st.last_update_age_s), int: fmtDur(st.log_interval_s || 60) }));
  if (st.source === "derived" && !snap.parameters) add("info", "i", tr("n_derived"));
}
function renderProgress() {
  const box = clear($("progress"));
  const snap = S.snap;
  box.hidden = !(snap && snap.run);
  if (box.hidden) return;
  const p = snap.progress || {};
  const latest = snap.latest || {};
  const pct = isNum(p.phase_fraction) ? ` · ${(p.phase_fraction * 100).toFixed(1)}%` : "";
  const right = [p.eta ? `${tr("eta")} ${fmtTime(p.eta)}` : null,
    isNum(p.run_remaining_s) ? `${tr("remaining")} ${fmtDur(p.run_remaining_s)}` : null,
    isNum(p.run_wall_s) ? `${tr("wall")} ${fmtDur(p.run_wall_s)}` : null,
    isNum(p.budget_hours) && p.budget_hours ? `${tr("budget")} ${p.budget_hours} h` : null].filter(Boolean).join(" · ");
  box.append(h("div", { class: "progress-head" },
    h("div", null, h("strong", { text: `${phaseName(p.current)} · ${tr("step")} ${fmtInt(latest.step)} ${tr("of")} ${fmtInt(latest.total)}` }),
      h("span", { class: "muted num", text: pct })),
    h("div", { class: "muted num", text: right })));
  const phases = h("div", { class: "phases" });
  const list = p.phases || [];
  phases.style.gridTemplateColumns = list.map((ph) => `${Math.max(0.05, ph.budget_h || 1)}fr`).join(" ") || "1fr";
  for (const ph of list) {
    const frac = ph.state === "complete" ? 1 : ph.state === "active"
      ? (isNum(p.phase_fraction) ? p.phase_fraction : (ph.total ? ph.step / ph.total : 0)) : 0;
    const fill = h("div", { class: "fill" });
    fill.style.width = `${(Math.max(0, Math.min(1, frac)) * 100).toFixed(2)}%`;
    phases.append(h("div", { class: `phase ${ph.state}` },
      h("div", { class: "track", role: "progressbar", "aria-label": phaseName(ph.name), "aria-valuemin": "0",
        "aria-valuemax": "100", "aria-valuenow": String(Math.round(frac * 100)) }, fill),
      h("b", { text: phaseName(ph.name) }), ` ${tr("ph_" + ph.state)} · ${ph.budget_h} h`,
      ph.total ? ` · ${fmtInt(ph.step)}/${fmtInt(ph.total)}` : ""));
  }
  box.append(phases);
}

// ---- KPI tiles ---------------------------------------------------------------------
function phaseRows(phase) { return ((S.snap && S.snap.history) || {})[phase] || []; }
function sparkline(rows, key) {
  const pts = rows.slice(-160).map((r) => r[key]).filter(isNum);
  if (pts.length < 2) return null;
  let lo = Math.min(...pts), hi = Math.max(...pts);
  if (lo === hi) { lo -= 1; hi += 1; }
  const d = pts.map((v, i) => `${i ? "L" : "M"}${(i / (pts.length - 1) * 100).toFixed(2)} ${(24 - (v - lo) / (hi - lo) * 22).toFixed(2)}`).join("");
  const svg = s("svg", { viewBox: "0 0 100 26", preserveAspectRatio: "none", "aria-hidden": "true" });
  svg.append(s("path", { d, fill: "none", stroke: "var(--series-1)", "stroke-width": "2", "vector-effect": "non-scaling-stroke",
    "stroke-linejoin": "round", "stroke-linecap": "round" }));
  return svg;
}
function renderTiles() {
  const box = clear($("tiles"));
  const snap = S.snap;
  if (!snap || !snap.run) return;
  const L = snap.latest || {};
  const phase = (snap.progress || {}).current || L.phase;
  const rows = phaseRows(phase);
  const spec = snap.spec || {};
  const cfg = spec.model_config || {};
  const phSpec = (spec.phases || []).find((p) => p.name === phase) || {};
  const best = (snap.best_valid || {})[phase];
  const mem = snap.memory || {};
  const p = snap.progress || {};
  const tile = (label, value, sub, key) => box.append(h("div", { class: "tile card" },
    h("div", { class: "label", text: label }), h("div", { class: "value num", text: value }),
    sub ? h("div", { class: "sub", text: sub }) : null, key ? sparkline(rows, key) : null));
  tile(tr("k_loss"), fmt(L.loss), tr("sub_at", { s: fmtInt(L.step) }), "loss");
  tile(tr("k_ce"), fmt(L.primary_loss), isNum(L.primary_loss) ? tr("sub_ppl", { v: fmt(Math.exp(L.primary_loss), 3) }) : null, "primary_loss");
  tile(tr("k_valid"), best ? fmt(best.valid_ce) : "—", best ? `${tr("sub_at", { s: fmtInt(best.step) })} · ${tr("sub_ppl", { v: fmt(Math.exp(best.valid_ce), 3) })}` : null);
  tile(tr("k_tps"), isNum(L.tokens_per_s) ? `${fmtCount(L.tokens_per_s)} tok/s` : "—",
    isNum(L.processed_tokens_per_s) ? tr("sub_proc", { v: fmtCount(L.processed_tokens_per_s) }) : null, "tokens_per_s");
  tile(tr("k_eta"), p.eta ? fmtTime(p.eta) : "—", isNum(p.run_remaining_s) ? tr("sub_left", { v: fmtDur(p.run_remaining_s) }) : null);
  const osMem = isNum(mem.footprint_gb);            // live process footprint has no history; MLX peak does
  tile(tr("k_mem"), osMem ? fmtGB(mem.footprint_gb) : fmtGB(L.peak_mem_gb),
    `${osMem ? tr("sub_footprint") + " · " : ""}${tr("sub_peak", { v: fmtGB(L.peak_mem_gb), a: fmtGB(L.active_mem_gb) })}`, osMem ? null : "peak_mem_gb");
  tile(tr("k_grad"), fmt(L.grad_norm), isNum(phSpec.grad_clip) ? tr("sub_clip", { v: phSpec.grad_clip }) : null, "grad_norm");
  tile(tr("k_lr"), fmt(L.lr, 3), isNum(phSpec.peak_lr) ? `peak ${fmt(phSpec.peak_lr, 3)}` : null, "lr");
  tile(tr("k_depth"), fmt(L.rec_depth_mean, 3), isNum(cfg.recurrent_depth) ? tr("sub_maxdepth", { v: cfg.recurrent_depth }) : null, "rec_depth_mean");
  tile(tr("k_route"), fmt(L.route_entropy, 3), isNum(L.route_entropy_max) ? `${tr("sub_entmax", { v: fmt(L.route_entropy_max, 3) })} · CV ${fmt(L.load_cv, 3)}` : null, "route_entropy");
}

// ---- tabs ----------------------------------------------------------------------------
const TABS = ["overview", "loss", "recurrent", "routing", "params", "evals", "config", "log", "ckpt"];
function renderTabs() {
  const nav = clear($("tabs"));
  if (!TABS.includes(S.tab)) S.tab = "overview";
  for (const id of TABS) {
    nav.append(h("button", { type: "button", role: "tab", id: `tab-${id}`, "aria-selected": String(S.tab === id),
      "aria-controls": "panel", tabindex: S.tab === id ? "0" : "-1", text: tr("t_" + id),
      onclick: () => selectTab(id), onkeydown: (e) => tabKeys(e, id) }));
  }
  $("panel").setAttribute("aria-labelledby", `tab-${S.tab}`);
}
function tabKeys(e, id) {
  const i = TABS.indexOf(id);
  const next = e.key === "ArrowRight" ? TABS[(i + 1) % TABS.length] : e.key === "ArrowLeft" ? TABS[(i - 1 + TABS.length) % TABS.length] : null;
  if (!next) return;
  e.preventDefault();
  selectTab(next);
  $(`tab-${next}`).focus();
}
function selectTab(id) {
  S.tab = id;
  localStorage.setItem("chronos.monitor.tab", id);
  history.replaceState(null, "", `#tab=${id}`);       // shareable deep link
  renderTabs();
  renderPanel(true);
  if (id === "log") loadLog().then(() => renderPanel(true));
}
// ---- tooltip -------------------------------------------------------------------------
const tip = {
  show(x, y, head, rows) {
    const el = clear($("tooltip"));
    el.append(h("div", { class: "t-head", text: head }));
    for (const r of rows) {
      const sw = h("i", { "aria-hidden": "true" });
      if (r.cls) sw.className = r.cls;
      el.append(h("div", { class: "t-row" }, r.cls ? sw : null, h("span", { text: r.label }), h("b", { text: r.value })));
    }
    el.hidden = false;
    const w = el.offsetWidth, ht = el.offsetHeight;
    el.style.left = `${Math.min(window.innerWidth - w - 8, x + 14)}px`;
    el.style.top = `${Math.max(8, Math.min(window.innerHeight - ht - 8, y - ht - 12))}px`;
  },
  hide() { $("tooltip").hidden = true; },
};

// ---- scales ----------------------------------------------------------------------------
function niceTicks(lo, hi, count = 4) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((st) => st >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}
function bisect(xs, x) {
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] < x) lo = mid; else hi = mid; }
  return Math.abs(xs[lo] - x) <= Math.abs(xs[hi] - x) ? lo : hi;
}

// ---- line chart (small multiple; one y axis; crosshair tooltip) ---------------------------
const pendingDraws = [];
function lineChart({ title, series, events = [], fmtY = fmt, note }) {
  const live = series.map((sr, i) => ({ ...sr, cls: sr.cls || `s${i + 1}`,
    pts: (sr.rows || []).filter((r) => isNum(r[sr.key]) && isNum(r.step)).map((r) => ({ x: r.step, y: r[sr.key], t: r.time })) }))
    .filter((sr) => sr.pts.length);
  const card = h("figure", { class: "chart card", tabindex: live.length ? "0" : null, "aria-label": title });
  const lastPts = live.length ? live[0].pts[live[0].pts.length - 1] : null;
  card.append(h("div", { class: "chart-head" }, h("h3", { text: title }),
    h("span", { class: "last", text: lastPts ? fmtY(lastPts.y) : "" })));
  if (live.length > 1) {
    card.append(h("div", { class: "legend" }, live.map((sr) => h("span", null, h("i", { class: sr.cls, "aria-hidden": "true" }), sr.label))));
  }
  if (!live.length) { card.append(h("div", { class: "empty", text: tr("no_data") })); return card; }
  const svg = s("svg", { role: "img", "aria-label": `${title}: ${lastPts ? fmtY(lastPts.y) : ""}` });
  card.append(svg);
  if (note) card.append(h("figcaption", { class: "sub muted", text: note }));
  pendingDraws.push(() => drawLines(card, svg, live, events, fmtY));
  return card;
}
function drawLines(card, svg, live, events, fmtY) {
  const W = Math.max(240, Math.round(svg.getBoundingClientRect().width || card.clientWidth - 28)), H = 170;
  const m = { l: 52, r: 10, t: 8, b: 22 };
  clear(svg);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const all = live.flatMap((sr) => sr.pts);
  let x0 = Math.min(...all.map((p) => p.x)), x1 = Math.max(...all.map((p) => p.x));
  let y0 = Math.min(...all.map((p) => p.y)), y1 = Math.max(...all.map((p) => p.y));
  if (x0 === x1) { x0 -= 1; x1 += 1; }
  if (y0 === y1) { const d = Math.abs(y0) * 0.05 || 1; y0 -= d; y1 += d; }
  const pad = (y1 - y0) * 0.06; y0 -= pad; y1 += pad;
  const X = (v) => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
  const Y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const grid = s("g", { class: "grid" }), axis = s("g", { class: "axis" });
  for (const v of niceTicks(y0, y1, 4)) {
    grid.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
    const t = s("text", { x: m.l - 6, y: Y(v) + 3, "text-anchor": "end" }); t.textContent = fmtY(v); axis.append(t);
  }
  for (const v of niceTicks(x0, x1, Math.max(3, Math.floor(W / 80)))) {
    const t = s("text", { x: X(v), y: H - 6, "text-anchor": "middle" }); t.textContent = fmtCount(v); axis.append(t);
  }
  svg.append(grid, s("line", { class: "baseline", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }), axis);
  for (const ev of events) {
    if (!isNum(ev.step) || ev.step < x0 || ev.step > x1) continue;
    svg.append(s("line", { class: `event ${ev.event}`, x1: X(ev.step), x2: X(ev.step), y1: m.t, y2: H - m.b }));
  }
  for (const sr of live.slice().reverse()) {
    if (sr.dots) {
      for (const p of sr.pts) svg.append(s("circle", { class: `valid`, cx: X(p.x), cy: Y(p.y), r: 4 }));
      continue;
    }
    svg.append(s("path", { class: `line ${sr.cls}`, d: sr.pts.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)} ${Y(p.y).toFixed(1)}`).join("") }));
  }
  // crosshair layer
  const cross = s("line", { class: "cross", y1: m.t, y2: H - m.b, visibility: "hidden" });
  const marks = live.map((sr) => s("circle", { class: `marker ${sr.dots ? "s2" : sr.cls}`, r: 4, visibility: "hidden" }));
  const hit = s("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: "transparent" });
  svg.append(cross, ...marks, hit);
  const base = live.find((sr) => !sr.dots) || live[0];
  const xs = base.pts.map((p) => p.x);
  let idx = xs.length - 1;
  const place = (i, clientX, clientY) => {
    idx = Math.max(0, Math.min(xs.length - 1, i));
    const step = xs[idx], cx = X(step);
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("visibility", "visible");
    const rows = [];
    live.forEach((sr, k) => {
      const j = bisect(sr.pts.map((p) => p.x), step), p = sr.pts[j];
      const near = sr.dots ? Math.abs(p.x - step) <= (x1 - x0) * 0.02 : true;
      marks[k].setAttribute("visibility", near ? "visible" : "hidden");
      if (!near) return;
      marks[k].setAttribute("cx", X(p.x)); marks[k].setAttribute("cy", Y(p.y));
      rows.push({ label: sr.label, value: fmtY(p.y), cls: live.length > 1 ? (sr.dots ? "s2" : sr.cls) : null });
    });
    const t = base.pts[idx].t;
    const box = svg.getBoundingClientRect();
    tip.show(clientX ?? box.left + cx * box.width / W, clientY ?? box.top + 20,
      `${tr("step")} ${fmtInt(step)}${t ? " · " + fmtTime(t) : ""}`, rows);
  };
  const hide = () => { cross.setAttribute("visibility", "hidden"); marks.forEach((mk) => mk.setAttribute("visibility", "hidden")); tip.hide(); };
  hit.addEventListener("pointermove", (e) => {
    const box = svg.getBoundingClientRect();
    place(bisect(xs, x0 + ((e.clientX - box.left) * W / box.width - m.l) / (W - m.l - m.r) * (x1 - x0)), e.clientX, e.clientY);
  });
  hit.addEventListener("pointerleave", hide);
  card.onkeydown = (e) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") { e.preventDefault(); place(idx + (e.key === "ArrowRight" ? 1 : -1)); }
    else if (e.key === "Escape") hide();
  };
  card.onfocus = () => place(idx);
  card.onblur = hide;
}
function flushDraws() { const fns = pendingDraws.splice(0); requestAnimationFrame(() => fns.forEach((fn) => fn())); }

// ---- heatmap (sequential, binned to theme ramp steps) -------------------------------------
function heatmap({ title, caption, rowLabels, values, colLabel = (j) => String(j), fmtV = (v) => fmt(v, 3), name }) {
  const box = h("div", { class: "box card" }, h("h3", { text: title }), caption ? h("p", { class: "caption", text: caption }) : null);
  const flat = values.flat().filter(isNum);
  if (!flat.length) { box.append(h("div", { class: "empty", text: tr("no_data") })); return box; }
  const lo = Math.min(...flat), hi = Math.max(...flat);
  const cols = Math.max(...values.map((r) => r.length));
  const grid = h("div", { class: "heat", role: "img", "aria-label": `${title}: ${fmtV(lo)} – ${fmtV(hi)}` });
  grid.style.gridTemplateColumns = `max-content repeat(${cols}, minmax(9px, 1fr))`;
  grid.append(h("span"));
  for (let j = 0; j < cols; j++) grid.append(h("span", { class: "collab", text: cols > 32 && j % 4 ? "" : colLabel(j) }));
  values.forEach((row, i) => {
    grid.append(h("span", { class: "rowlab", text: rowLabels[i] ?? String(i) }));
    for (let j = 0; j < cols; j++) {
      const v = row[j];
      const cell = h("span", { class: "cell" });
      if (isNum(v)) {
        const bin = hi > lo ? 1 + Math.min(6, Math.floor((v - lo) / (hi - lo) * 7)) : 4;
        cell.style.background = `var(--seq-${bin})`;
      }
      cell.addEventListener("pointerenter", (e) => tip.show(e.clientX, e.clientY, `${rowLabels[i] ?? i} · ${colLabel(j, true)}`,
        [{ label: name || title, value: fmtV(v) }]));
      cell.addEventListener("pointermove", (e) => tip.show(e.clientX, e.clientY, `${rowLabels[i] ?? i} · ${colLabel(j, true)}`,
        [{ label: name || title, value: fmtV(v) }]));
      cell.addEventListener("pointerleave", tip.hide);
      grid.append(cell);
    }
  });
  const ramp = h("span", { class: "ramp", "aria-hidden": "true" });
  for (let b = 1; b <= 7; b++) { const sw = h("span"); sw.style.background = `var(--seq-${b})`; ramp.append(sw); }
  box.append(grid, h("div", { class: "scale" }, h("span", { class: "num", text: fmtV(lo) }), ramp, h("span", { class: "num", text: fmtV(hi) })));
  return box;
}
// ---- panels: shared bits ---------------------------------------------------------------
function phasesInHistory() { return Object.keys((S.snap && S.snap.history) || {}); }
function chosenPhase() {
  const list = phasesInHistory();
  if (S.phase && list.includes(S.phase)) return S.phase;
  const cur = S.snap && S.snap.progress && S.snap.progress.current;
  return list.includes(cur) ? cur : list[list.length - 1] || "";
}
function windowed(rows) { return S.window > 0 ? rows.slice(-S.window) : rows; }
function chartEvents(phase) {
  if (!S.showEvents) return [];
  return ((S.snap && S.snap.events) || []).filter((e) => e.phase === phase && (e.event === "validation" || e.event === "rollback"));
}
function toolbar() {
  const bar = h("div", { class: "toolbar" });
  const phases = phasesInHistory();
  if (phases.length > 1) {
    const sel = h("select", { "aria-label": tr("phase"), onchange: (e) => { S.phase = e.target.value; renderPanel(true); } },
      phases.map((ph) => h("option", { value: ph, text: phaseName(ph) })));
    sel.value = chosenPhase();
    bar.append(h("label", { class: "field" }, tr("phase"), sel));
  }
  const win = h("select", { "aria-label": tr("window"), onchange: (e) => { S.window = Number(e.target.value); renderPanel(true); } },
    [0, 600, 200, 50].map((n) => h("option", { value: String(n), text: n ? tr("w_last", { n }) : tr("w_all") })));
  win.value = String(S.window);
  const ev = h("input", { type: "checkbox", id: "events-toggle", onchange: (e) => { S.showEvents = e.target.checked; renderPanel(true); } });
  ev.checked = S.showEvents;
  bar.append(h("label", { class: "field" }, tr("window"), win), h("label", { class: "field" }, ev, tr("events_on")));
  return bar;
}
const charts = (...cards) => h("div", { class: "charts" }, cards.flat(Infinity));

// ---- overview ----------------------------------------------------------------------------
function panelOverview() {
  const phase = chosenPhase();
  const rows = windowed(phaseRows(phase));
  const ev = chartEvents(phase);
  const valid = ((S.snap && S.snap.events) || []).filter((e) => e.event === "validation" && e.phase === phase && isNum(e.valid_ce));
  const lo = rows.length ? rows[0].step : 0;
  return [toolbar(), charts(
    lineChart({ title: tr("c_primary"), events: ev, series: [
      { key: "primary_loss", label: tr("train"), rows },
      { key: "valid_ce", label: tr("valid"), rows: valid.filter((e) => e.step >= lo), dots: true, cls: "s2" }] }),
    lineChart({ title: tr("c_loss"), events: ev, series: [{ key: "loss", label: mlabel("loss"), rows }] }),
    lineChart({ title: tr("c_lr"), series: [{ key: "lr", label: tr("k_lr"), rows }], fmtY: (v) => fmt(v, 3) }),
    lineChart({ title: tr("c_grad"), events: ev, series: [{ key: "grad_norm", label: tr("k_grad"), rows }] }),
    lineChart({ title: tr("c_tps"), series: [
      { key: "tokens_per_s", label: tr("seen"), rows }, { key: "processed_tokens_per_s", label: tr("processed"), rows }], fmtY: fmtCount }),
    lineChart({ title: tr("c_step"), series: [{ key: "step_s", label: tr("c_step"), rows }], fmtY: (v) => fmt(v, 3) }),
    lineChart({ title: tr("c_mem"), series: [
      { key: "peak_mem_gb", label: tr("peak"), rows }, { key: "active_mem_gb", label: tr("active"), rows },
      { key: "cache_mem_gb", label: tr("cache"), rows }], fmtY: (v) => fmt(v, 3) }),
    lineChart({ title: tr("c_tokens"), series: [{ key: "tokens_seen", label: tr("c_tokens"), rows }], fmtY: fmtCount }),
  )];
}

// ---- loss components ---------------------------------------------------------------------------
function panelLoss() {
  const phase = chosenPhase();
  const rows = windowed(phaseRows(phase));
  const ev = chartEvents(phase);
  const keys = ((S.snap && S.snap.groups && S.snap.groups.loss) || []).filter((k) => rows.some((r) => isNum(r[k])));
  for (const k of ["recurrent_contraction_loss"]) if (rows.some((r) => isNum(r[k])) && !keys.includes(k)) keys.push(k);
  return [toolbar(), charts(keys.map((k) => lineChart({ title: mlabel(k), events: ev, series: [{ key: k, label: mlabel(k), rows }] })))];
}

// ---- recurrence ------------------------------------------------------------------------------------
function panelRecurrent() {
  const phase = chosenPhase();
  const rows = windowed(phaseRows(phase));
  const L = (S.snap && S.snap.latest) || {};
  const cfg = ((S.snap && S.snap.spec) || {}).model_config || {};
  const out = [toolbar(), h("p", { class: "caption muted", text: tr("rec_caption") })];
  out.push(charts(["rec_depth_mean", "rec_contraction", "rec_early_exit_ratio", "rec_ponder_loss", "rec_distill_loss", "rec_stability_loss"]
    .filter((k) => rows.some((r) => isNum(r[k])))
    .map((k) => lineChart({ title: mlabel(k), series: [{ key: k, label: mlabel(k), rows }], fmtY: (v) => fmt(v, 3) }))));
  const facts = h("dl", { class: "kv" });
  const add = (k, v) => facts.append(h("dt", { text: k }), h("dd", { class: "num", text: v }));
  add("recurrent_depth", String(cfg.recurrent_depth ?? "—"));
  add("min_recurrent_depth", String(cfg.min_recurrent_depth ?? "—"));
  add("recurrent_attention_mode", String(cfg.recurrent_attention_mode ?? "—"));
  add("adaptive_halting", String(cfg.adaptive_halting ?? "—"));
  add("depth_schedule", String(cfg.depth_schedule ?? "—"));
  if (Array.isArray(L.rec_depth_per_group)) add(S.lang === "zh" ? "每组深度（最新）" : "depth per group (latest)", L.rec_depth_per_group.map((v) => fmt(v, 3)).join(" · "));
  if (Array.isArray(L.rec_halt_mass_by_step)) add(S.lang === "zh" ? "累计停止质量 / 迭代" : "cumulative halt mass / iteration", L.rec_halt_mass_by_step.map((v) => fmt(v, 3)).join(" · "));
  out.push(h("div", { class: "grid2" }, h("div", { class: "box card" }, h("h3", { text: tr("cfg_model") }), facts)));
  return out;
}

// ---- routing -------------------------------------------------------------------------------------------
function panelRouting() {
  const phase = chosenPhase();
  const rows = windowed(phaseRows(phase));
  const out = [toolbar(), h("p", { class: "caption muted", text: tr("route_caption") })];
  out.push(charts(["route_entropy", "load_cv", "max_expert_load", "dead_experts"].filter((k) => rows.some((r) => isNum(r[k])))
    .map((k) => lineChart({ title: mlabel(k), series: [{ key: k, label: mlabel(k), rows }], fmtY: (v) => fmt(v, 3) }))));
  const r = S.snap && S.snap.parameters && S.snap.parameters.routing;
  const m = S.snap && S.snap.parameters;
  if (!r) {
    const every = ((S.snap && S.snap.spec) || {}).tensor_stats_every_s || 1800;
    out.push(h("div", { class: "empty", text: tr("no_telemetry", { m: Math.round(every / 60) }) }));
    return out;
  }
  const expertLabel = (j, long) => long ? `${S.lang === "zh" ? "专家" : "expert"} ${j}` : String(j);
  const meta = tr("tel_meta", { s: fmtInt(m.step), t: fmtTime(m.time), ms: fmtInt((m.telemetry_s || 0) * 1000), note: r.source || "" });
  out.push(h("p", { class: "caption muted", text: meta }));
  const ent = r.entropy || [];
  out.push(h("div", { class: "grid2" },
    heatmap({ title: tr("heat_load"), caption: `${tr("heat_rows")} · top-${r.top_k} · ${r.experts} experts`, rowLabels: r.labels,
      values: r.load, colLabel: expertLabel, fmtV: (v) => isNum(v) ? (v * 100).toFixed(1) + "%" : "—", name: tr("heat_load") }),
    heatmap({ title: tr("heat_importance"), caption: tr("heat_rows"), rowLabels: r.labels.map((l, i) => isNum(ent[i]) ? `${l}  H=${fmt(ent[i], 3)}` : l),
      values: r.importance, colLabel: expertLabel, fmtV: (v) => fmt(v, 3), name: tr("heat_importance") })));
  return out;
}
// Extra strings used by the detail panels.
Object.assign(STR.zh, {
  probe_note: "评测指标由训练器在每次验证时写入 metrics.jsonl，页面自动发现并按指标分组；探针样本量通常很小，只作趋势参考。",
  lower_better: "越低越好", higher_better: "越高越好",
  final_eval: "最终评测", all_events: "全部事件", no_events: "暂无事件。", no_samples: "暂无评测样例。",
  tensors_loading: "正在加载张量统计…", shown: "显示 {n} / {t}", h_rms: "权重 RMS", h_grad: "梯度 RMS", h_upd: "更新比",
  stats_total: "全模型", every: "每 {m} 一次", none: "无",
});
Object.assign(STR.en, {
  probe_note: "Evaluation metrics are whatever the trainer writes to metrics.jsonl at each validation; the page discovers and groups them by metric. Probes usually use a small sample, so read them as a trend.",
  lower_better: "lower is better", higher_better: "higher is better",
  final_eval: "Final evaluation", all_events: "All events", no_events: "No events yet.", no_samples: "No evaluation samples yet.",
  tensors_loading: "Loading tensor statistics…", shown: "showing {n} of {t}", h_rms: "Weight RMS", h_grad: "Grad RMS", h_upd: "Update ratio",
  stats_total: "Whole model", every: "every {m}", none: "none",
});

function table(columns, rows, opts = {}) {
  if (opts.hideEmpty) {                      // drop columns with no value in any row (e.g. SFT metrics during pretrain)
    const value = (c, r) => (c.get ? c.get(r) : r[c.key]);
    columns = columns.filter((c) => c.keep || rows.some((r) => { const v = value(c, r); return v != null && v !== ""; }));
  }
  const head = h("tr", null, columns.map((c) => h("th", { class: c.num ? "num" : null, scope: "col", text: c.label })));
  const body = h("tbody");
  for (const r of rows) {
    const tr_ = h("tr", { class: opts.onRow ? "clickable" : null, tabindex: opts.onRow ? "0" : null });
    if (opts.selected && opts.selected(r)) tr_.classList.add("selected");
    for (const c of columns) {
      const v = c.get ? c.get(r) : r[c.key];
      tr_.append(h("td", { class: c.num ? "num" : null, text: c.fmt ? c.fmt(v, r) : (v ?? "—") }));
    }
    if (opts.onRow) {
      tr_.addEventListener("click", () => opts.onRow(r));
      tr_.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); opts.onRow(r); } });
    }
    body.append(tr_);
  }
  return h("div", { class: "table-wrap" }, h("table", null, h("thead", null, head), body));
}

// ---- parameters (per-tensor telemetry) ----------------------------------------------------------
async function loadTensors() {
  if (S.tensorsLoading) return;
  S.tensorsLoading = true;
  try { S.tensors = await api("api/tensors", { run: S.run }); }
  catch (e) { if (e instanceof AuthError) showLogin(""); else S.error = e.message; }
  finally { S.tensorsLoading = false; }
}
async function pickTensor(name) {
  S.tensorPick = name;
  try { S.tensorHist = (await api("api/tensor", { run: S.run, name })).history || []; }
  catch (e) { if (e instanceof AuthError) { showLogin(""); return; } S.tensorHist = []; S.error = e.message; }
  renderPanel(true);
}
function tensorTable(container) {
  clear(container);
  const all = (S.tensors && S.tensors.tensors) || [];
  const q = S.tensorFilter.trim().toLowerCase();
  let rows = q ? all.filter((t) => t.name.toLowerCase().includes(q) || (t.group || "").includes(q)) : all.slice();
  const key = S.tensorSort;
  if (key !== "name") rows.sort((a, b) => (isNum(b[key]) ? b[key] : -Infinity) - (isNum(a[key]) ? a[key] : -Infinity));
  container.append(h("p", { class: "caption muted", text: tr("shown", { n: Math.min(rows.length, 400), t: all.length }) }));
  const numel = (t) => (t.shape || []).reduce((a, b) => a * b, 1);
  container.append(table([
    { label: tr("name"), key: "name" }, { label: tr("group"), key: "group" },
    { label: tr("shape"), get: (t) => (t.shape || []).join("×") }, { label: tr("dtype"), key: "dtype" },
    { label: tr("numel"), get: numel, num: true, fmt: fmtCount },
    { label: tr("rms"), key: "rms", num: true, fmt: (v) => fmt(v, 3) }, { label: tr("std"), key: "std", num: true, fmt: (v) => fmt(v, 3) },
    { label: tr("absmax"), key: "absmax", num: true, fmt: (v) => fmt(v, 3) },
    { label: tr("grad_rms"), key: "grad_rms", num: true, fmt: (v) => fmt(v, 3) },
    { label: tr("update_ratio"), key: "update_ratio", num: true, fmt: (v) => fmt(v, 3) },
  ], rows.slice(0, 400), { onRow: (t) => pickTensor(t.name), selected: (t) => t.name === S.tensorPick }));
}
function panelParams() {
  const m = S.snap && S.snap.parameters;
  const every = ((S.snap && S.snap.spec) || {}).tensor_stats_every_s || 1800;
  if (!m) return [h("div", { class: "empty", text: tr("no_telemetry", { m: Math.round(every / 60) }) })];
  const out = [h("p", { class: "caption muted", text: `${phaseName(m.phase)} · ${tr("tel_meta", { s: fmtInt(m.step), t: fmtTime(m.time),
    ms: fmtInt((m.telemetry_s || 0) * 1000), note: m.grad_note || "" })} · ${tr("every", { m: fmtDur(every) })}` })];
  const groups = Object.entries(m.groups || {}).map(([name, g]) => ({ name, ...g }));
  if (m.stats) groups.push({ name: tr("stats_total"), ...m.stats, tensors: m.tensor_count });
  out.push(h("div", { class: "box card" }, h("h3", { text: tr("groups") }), table([
    { label: tr("group"), key: "name" }, { label: tr("tensors_n"), key: "tensors", num: true, fmt: fmtInt },
    { label: tr("numel"), key: "numel", num: true, fmt: fmtCount }, { label: tr("rms"), key: "rms", num: true, fmt: (v) => fmt(v, 3) },
    { label: tr("grad_norm"), key: "grad_norm", num: true, fmt: (v) => fmt(v, 3) },
    { label: tr("update_ratio"), key: "update_ratio", num: true, fmt: (v) => fmt(v, 3) }], groups)));
  const ex = m.experts || {};
  if (ex.layers && ex.layers.length) {
    const rowsL = ex.layers.map((l) => `L${l}`);
    const expertLabel = (j, long) => long ? `${S.lang === "zh" ? "专家" : "expert"} ${j}` : String(j);
    out.push(h("div", { class: "grid2" },
      heatmap({ title: tr("heat_wrms"), caption: tr("heat_rows_layer"), rowLabels: rowsL, values: ex.weight_rms, colLabel: expertLabel }),
      heatmap({ title: tr("heat_grms"), caption: tr("heat_rows_layer"), rowLabels: rowsL, values: ex.grad_rms, colLabel: expertLabel })));
  }
  if (S.tensorPick && S.tensorHist) {
    const rows = S.tensorHist;
    out.push(h("div", { class: "box card" }, h("h3", { text: tr("hist_of", { n: S.tensorPick }) }), charts(
      lineChart({ title: tr("h_rms"), series: [{ key: "rms", label: tr("h_rms"), rows }], fmtY: (v) => fmt(v, 3) }),
      lineChart({ title: tr("h_grad"), series: [{ key: "grad_rms", label: tr("h_grad"), rows }], fmtY: (v) => fmt(v, 3) }),
      lineChart({ title: tr("h_upd"), series: [{ key: "update_ratio", label: tr("h_upd"), rows }], fmtY: (v) => fmt(v, 3) }))));
  }
  const box = h("div", { class: "box card" }, h("h3", { text: tr("tensors") }));
  const holder = h("div");
  if (!S.tensors || S.tensors.step !== m.step) {
    holder.append(h("div", { class: "empty", text: tr("tensors_loading") }));
    if (S.tensorsFor !== m.step) {             // one attempt per telemetry step (no retry loop on errors)
      S.tensorsFor = m.step;
      loadTensors().then(() => renderPanel(true));
    }
  } else {
    const filter = h("input", { type: "search", placeholder: tr("filter"), "aria-label": tr("filter"), value: S.tensorFilter,
      oninput: (e) => { S.tensorFilter = e.target.value; tensorTable(holder); } });
    const sort = h("select", { "aria-label": tr("sort"), onchange: (e) => { S.tensorSort = e.target.value; tensorTable(holder); } },
      ["name", "rms", "absmax", "grad_rms", "update_ratio"].map((k) => h("option", { value: k, text: tr(k) })));
    sort.value = S.tensorSort;
    box.append(h("div", { class: "toolbar" }, filter, h("label", { class: "field" }, tr("sort"), sort)),
      h("p", { class: "caption muted", text: tr("hist_hint") }));
    tensorTable(holder);
  }
  box.append(holder);
  out.push(box);
  return out;
}
// ---- evaluations -------------------------------------------------------------------------------------
function flatten(obj, prefix = "", out = [], depth = 0) {
  if (obj == null) return out;
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v) && depth < 3) flatten(v, key, out, depth + 1);
    else if (Array.isArray(v)) {
      const prim = v.every((x) => x == null || typeof x !== "object");
      out.push([key, prim && v.length <= 24 ? v.map((x) => isNum(x) ? fmt(x, 5) : String(x)).join(", ") : `[${v.length}]`]);
    } else if (v && typeof v === "object") out.push([key, JSON.stringify(v).slice(0, 240)]);
    else out.push([key, isNum(v) ? (Number.isInteger(v) ? fmtInt(v) : fmt(v, 6)) : String(v)]);
  }
  return out;
}
function kvBox(title, obj, extra) {
  const dl = h("dl", { class: "kv" });
  for (const [k, v] of flatten(obj)) dl.append(h("dt", { text: k }), h("dd", { text: v }));
  return h("div", { class: "box card" }, h("h3", { text: title }), extra || null, obj ? dl : h("div", { class: "empty", text: tr("none") }));
}
// Samples come from samples.jsonl through chronos/monitor/evals.py:sample_view —
// input / reference / output roles plus free-form tags, whatever the task.
function sampleList(samples) {
  if (!samples.length) return h("div", { class: "empty", text: tr("no_samples") });
  const line = (lab, v) => (v == null || v === "" ? null : h("div", null, h("span", { class: "lab", text: lab }), String(v)));
  return h("div", null, samples.map((x) => {
    const tags = Object.entries(x.tags || {}).map(([k, v]) => `${k}=${isNum(v) ? fmt(v, 4) : v}`);
    return h("div", { class: "sample" },
      h("div", { class: "meta", text: [phaseName(x.phase), x.step != null ? `${tr("step")} ${fmtInt(x.step)}` : null,
        ...tags, fmtTime(x.time)].filter(Boolean).join(" · ") }),
      line(tr("src"), x.input), line(tr("ref"), x.reference), line(tr("hyp"), x.output));
  }));
}
const METRIC_NAMES = { bleu: "BLEU", chrf: "chrF", ter: "TER", rouge1: "ROUGE-1", rouge2: "ROUGE-2", rougel: "ROUGE-L",
  meteor: "METEOR", comet: "COMET", bertscore: "BERTScore", acc: "Acc", accuracy: "Accuracy", f1: "F1", em: "EM",
  auc: "AUC", wer: "WER", cer: "CER", ppl: "PPL", ce: "CE", nll: "NLL", bpb: "BPB", bpc: "BPC", kl: "KL" };
const metricName = (m) => METRIC_NAMES[m] || m;
// One chart per metric family and phase (steps restart at each phase), at most three series each.
function evalCharts(valid, families) {
  const cards = [];
  const phases = [...new Set(valid.map((e) => e.phase))];
  for (const fam of families) {
    for (const ph of phases) {
      const rows = valid.filter((e) => e.phase === ph);
      const live = fam.series.filter((sr) => rows.some((e) => isNum(e[sr.key])));
      for (let i = 0; i < live.length; i += 3) {
        const part = live.slice(i, i + 3);
        const title = `${metricName(fam.metric)} · ${phaseName(ph)}${live.length > 3 ? ` (${i / 3 + 1})` : ""}`;
        cards.push(lineChart({ title, series: part.map((sr) => ({ key: sr.key, label: sr.label, rows })),
          fmtY: (v) => fmt(v, 4), note: fam.direction ? tr(fam.direction + "_better") : undefined }));
      }
    }
  }
  return cards;
}
function panelEvals() {
  const snap = S.snap || {};
  const events = snap.events || [];
  const evals = snap.evals || { families: [], keys: [], context: [] };
  const out = [];
  const final = events.filter((e) => e.event === "final").pop();
  if (final) out.push(kvBox(tr("final_eval"), Object.fromEntries(Object.entries(final).filter(([k]) => !["event", "samples"].includes(k)))));
  const valid = events.filter((e) => e.event === "validation");
  const cards = evalCharts(valid, evals.families || []);
  if (cards.length) out.push(h("p", { class: "caption muted", text: tr("probe_note") }), charts(cards));
  const f4 = (v) => fmt(v, 4);
  const extra = (evals.keys || []).filter((k) => k !== "valid_ce" && k !== "valid_ppl");
  out.push(h("div", { class: "box card" }, h("h3", { text: tr("validations") }), valid.length ? table([
    { label: tr("time"), key: "time", fmt: fmtTime }, { label: tr("phase"), key: "phase", fmt: phaseName },
    { label: tr("step"), key: "step", num: true, fmt: fmtInt }, { label: "valid CE", key: "valid_ce", num: true, fmt: f4 },
    { label: tr("ppl"), key: "valid_ppl", num: true, fmt: f4 },
    ...extra.map((k) => ({ label: k, key: k, num: true, fmt: f4 })),
    ...(evals.context || []).map((k) => ({ label: k, key: k, num: true, fmt: (v) => fmt(v, 4) })),
  ], valid.slice().reverse(), { hideEmpty: true }) : h("div", { class: "empty", text: tr("no_events") })));
  const ins = snap.inspections || [];
  if (ins.length) {
    const keys = [...new Set(ins.flatMap((r) => Object.keys(r.metrics || {})))];
    const shown = ins.slice().reverse();
    out.push(h("div", { class: "box card" }, h("h3", { text: tr("inspections") }),
      h("p", { class: "caption muted", text: shown.map((r) => `${r.file}: ${phaseName(r.phase)} ${tr("step")} ${fmtInt(r.step)} · ${r.device || "?"} · ${r.compute_dtype || "?"}`).join("   ") }),
      table([{ label: tr("metric"), get: (k) => k },
        ...shown.map((r) => ({ label: `${tr("step")} ${fmtInt(r.step)}`, get: (k) => (r.metrics || {})[k], num: true, fmt: f4 }))],
      keys)));
  }
  out.push(h("div", { class: "box card" }, h("h3", { text: tr("samples") }), sampleList((snap.samples || []).slice().reverse().slice(0, 12))));
  const other = events.filter((e) => e.event !== "validation");
  out.push(h("div", { class: "box card" }, h("h3", { text: tr("all_events") }), other.length ? table([
    { label: tr("time"), key: "time", fmt: fmtTime }, { label: tr("kind"), key: "event", fmt: (v) => tr("ev_" + v) },
    { label: tr("phase"), key: "phase", fmt: phaseName }, { label: tr("step"), key: "step", num: true, fmt: fmtInt },
    { label: tr("detail"), get: (e) => Object.entries(e).filter(([k]) => !["time", "event", "phase", "step"].includes(k))
      .map(([k, v]) => `${k}=${isNum(v) ? fmt(v, 5) : typeof v === "object" ? JSON.stringify(v) : v}`).join("  ") },
  ], other.slice().reverse()) : h("div", { class: "empty", text: tr("no_events") })));
  return out;
}
// ---- training parameters ---------------------------------------------------------------------------
async function loadParams() {
  try { S.params = await api("api/parameters", { run: S.run }); }
  catch (e) { if (e instanceof AuthError) { showLogin(""); return; } S.params = {}; S.error = e.message; }
}
// Every scalar top-level spec field (whatever the recipe defines), plus ``extra``.
const RUN_SKIP = new Set(["phases", "model_config", "extra"]);
function panelConfig() {
  const run = S.snap && S.snap.run;
  if (!S.params || S.params.run !== run) {
    if (S.paramsFor !== run) { S.paramsFor = run; loadParams().then(() => renderPanel(true)); }
    return [h("div", { class: "empty", text: "…" })];
  }
  const spec = S.params.spec || {};
  const rt = S.params.runtime;
  const phases = (spec.phases || []).map((ph) => {
    const r = ((rt && rt.phases) || []).find((x) => x.name === ph.name) || {};
    return { ...ph, accum: r.accum ?? (ph.micro_tokens ? Math.max(1, Math.floor(ph.tokens_per_step / ph.micro_tokens)) : null) };
  });
  const out = [];
  out.push(h("div", { class: "box card" }, h("h3", { text: tr("cfg_phases") }), table([
    { label: tr("phase"), key: "name", fmt: phaseName }, { label: "budget", key: "budget_fraction", num: true, fmt: (v) => fmt(v, 3) },
    { label: "peak_lr", key: "peak_lr", num: true, fmt: (v) => fmt(v, 3) }, { label: "tokens/step", key: "tokens_per_step", num: true, fmt: fmtInt },
    { label: "micro_tokens", key: "micro_tokens", num: true, fmt: fmtInt }, { label: "accum", key: "accum", num: true, fmt: fmtInt },
    { label: "seq_len", key: "seq_len", num: true, fmt: fmtInt }, { label: "warmup", key: "warmup_fraction", num: true, fmt: (v) => fmt(v, 3) },
    { label: "min_lr_ratio", key: "min_lr_ratio", num: true, fmt: (v) => fmt(v, 3) }, { label: "weight_decay", key: "weight_decay", num: true, fmt: (v) => fmt(v, 3) },
    { label: "grad_clip", key: "grad_clip", num: true, fmt: (v) => fmt(v, 3) }], phases)));
  const runObj = Object.fromEntries(Object.entries(spec).filter(([k, v]) => !RUN_SKIP.has(k) && (v == null || typeof v !== "object")));
  if (spec.extra) runObj.extra = spec.extra;
  const grid = h("div", { class: "grid2" });
  grid.append(kvBox(tr("cfg_run"), runObj), kvBox(tr("cfg_model"), spec.model_config || null));
  if (rt) {
    const { phases: _p, ...rest } = rt;
    grid.append(kvBox(tr("cfg_runtime"), rest));
  } else {
    const dl = h("dl", { class: "kv" }, h("dt", { text: "dtype" }), h("dd", { text: String(spec.dtype ?? "—") }));
    grid.append(h("div", { class: "box card" }, h("h3", { text: tr("cfg_runtime") }), h("p", { class: "caption", text: tr("cfg_missing") }), dl));
  }
  if (S.params.data) grid.append(kvBox(tr("cfg_data"), S.params.data));
  if (S.params.tokenizer) grid.append(kvBox(tr("cfg_tok"), S.params.tokenizer));
  out.push(grid);
  out.push(h("details", null, h("summary", { text: tr("cfg_raw") }), h("pre", { text: JSON.stringify(S.params, null, 1) })));
  return out;
}

// ---- log ----------------------------------------------------------------------------------------------
async function loadLog() {
  try { S.log = await api("api/log", { run: S.run, bytes: S.logBytes }, true); }
  catch (e) { if (e instanceof AuthError) showLogin(""); else S.log = tr("err_net") + e.message; }
}
function panelLog() {
  const pre = h("pre", { class: "log", tabindex: "0", text: S.log ?? "…" });
  const auto = h("input", { type: "checkbox", onchange: (e) => { S.logAuto = e.target.checked; } });
  auto.checked = S.logAuto;
  pendingDraws.push(() => { if (S.logAuto) pre.scrollTop = pre.scrollHeight; });
  return [h("div", { class: "toolbar" },
    h("button", { type: "button", text: tr("log_refresh"), onclick: () => loadLog().then(() => renderPanel(true)) }),
    h("button", { type: "button", text: tr("log_more"), disabled: S.logBytes >= 400000 ? true : null,
      onclick: () => { S.logBytes = Math.min(400000, S.logBytes * 4); loadLog().then(() => renderPanel(true)); } }),
    h("label", { class: "field" }, auto, tr("log_auto"))), pre];
}

// ---- checkpoints -----------------------------------------------------------------------------------------
function panelCkpt() {
  const list = (S.snap && S.snap.checkpoints) || [];
  if (!list.length) return [h("div", { class: "empty", text: tr("ck_none") })];
  return [h("div", { class: "box card" }, table([
    { label: tr("ck_file"), key: "file" }, { label: tr("ck_size"), key: "size_gb", num: true, fmt: fmtGB },
    { label: tr("ck_modified"), key: "modified", fmt: fmtTime }, { label: tr("ck_age"), key: "age_s", num: true, fmt: fmtDur }], list))];
}

// ---- render orchestration -----------------------------------------------------------------------------------
const PANELS = { overview: panelOverview, loss: panelLoss, recurrent: panelRecurrent, routing: panelRouting, params: panelParams,
  evals: panelEvals, config: panelConfig, log: panelLog, ckpt: panelCkpt };
function panelKey() {
  const snap = S.snap || {};
  if (S.tab === "params") return [snap.run, (snap.parameters || {}).step, S.tensors && S.tensors.step, S.tensorPick, (S.tensorHist || []).length, S.lang].join("|");
  if (S.tab === "config") return [snap.run, S.params ? S.params.run : "", S.lang].join("|");
  return null;                                   // charts, events and the log redraw on every poll
}
function renderPanel(force) {
  const panel = $("panel");
  if (!S.snap || !S.snap.run) { panel.replaceChildren(); return; }
  const key = panelKey();
  if (!force && key !== null && key === S.panelKey) return;   // keep filter focus / table scroll between polls
  S.panelKey = key;
  tip.hide();
  pendingDraws.length = 0;
  let content;
  try { content = PANELS[S.tab](); }
  catch (e) { content = [h("div", { class: "notice error", role: "alert" }, h("span", { class: "icon", text: "✕" }), h("span", { text: String(e && e.stack || e) }))]; }
  panel.replaceChildren(...[content].flat().filter(Boolean));
  flushDraws();
}
function renderFoot() {
  const snap = S.snap;
  const foot = clear($("foot"));
  if (!snap) return;
  const st = snap.status || {};
  const bits = [tr("foot", { s: POLL_MS / 1000, root: snap.root || "—", t: fmtTime(snap.generated) })];
  if (snap.run) bits.push(`status: ${st.source || "?"}${st.pid ? ` · pid ${st.pid}` : ""}${st.ps_state ? ` (${st.ps_state})` : ""}`);
  if (snap.runtime && snap.runtime.code) bits.push(`code ${snap.runtime.code.sha256}`);
  foot.textContent = bits.join(" · ");
}
function renderAll(force) { renderChrome(); renderNotices(); renderProgress(); renderTiles(); renderPanel(force); renderFoot(); }

// ---- wiring --------------------------------------------------------------------------------------------------
function resetRunCaches() { S.params = S.tensors = S.tensorHist = S.log = null; S.tensorPick = null; S.paramsFor = S.tensorsFor = undefined; S.panelKey = null; S.phase = ""; }
function init() {
  $("run").addEventListener("change", (e) => { S.run = e.target.value; resetRunCaches(); refresh(); });
  $("lang").addEventListener("click", () => {
    S.lang = S.lang === "zh" ? "en" : "zh";
    localStorage.setItem("chronos.monitor.lang", S.lang);
    renderTabs(); renderAll(true);
    if (!$("login").hidden) showLogin($("login-error").textContent);
  });
  $("theme").addEventListener("click", () => {
    S.theme = effectiveDark() ? "light" : "dark";
    localStorage.setItem("chronos.monitor.theme", S.theme);
    renderChrome();
  });
  $("logout").addEventListener("click", logout);
  $("login-form").addEventListener("submit", submitLogin);
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); else if ($("login").hidden) refresh().then(schedule); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderChrome);
  let resizeT;
  window.addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(() => renderPanel(true), 200); });
  const fromHash = /(?:^|[#&])tab=([a-z]+)/.exec(location.hash);
  if (fromHash && TABS.includes(fromHash[1])) S.tab = fromHash[1];
  window.addEventListener("hashchange", () => {
    const m = /(?:^|[#&])tab=([a-z]+)/.exec(location.hash);
    if (m && TABS.includes(m[1]) && m[1] !== S.tab) selectTab(m[1]);
  });
  renderChrome();
  renderTabs();
  start();
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
