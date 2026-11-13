// Export, Ollama/OpenAI server, benchmark, auto-tune and IO metrics pages.
import { state, t, h, api, post, card, pageHead, button, num, int, table, logBox, badge, toast, fieldLabel, dateTime } from "./core.js";
import { renderGroup, formState, submit, assign } from "./forms.js";
import { lineChart, barChart } from "./charts.js";
import { ensureConfig } from "./pages_config.js";

const active = (s) => ["running", "starting", "stopping", "loading"].includes(s);

// ── export ─────────────────────────────────────────────────────────────
export async function renderExport(main, page) {
  const values = formState("export", "export");
  const v = state.values.exportPage || (state.values.exportPage = { job: null });
  const root = h("div");
  const out = h("div");
  const run = button("export.run", () => submit(run, root, async () => {
    v.job = (await post("/api/export", values)).job.id;
    page.refresh();
  }), { primary: true });
  root.append(h("p", { class: "note", text: t("export.note") }), renderGroup("export", "export", values, null, { summary: false }),
    h("div", { class: "row end mt" }, run));
  main.append(pageHead("export.title", "export.intro"), card(null, root), out);
  page.poll(async () => {
    if (!v.job) return 0;
    const job = await api(`/api/jobs/${encodeURIComponent(v.job)}?log=300`);
    run.disabled = active(job.state);
    out.replaceChildren(card(null, h("div", { class: "row between" }, h("h3", { text: t("ui.result") }), badge(job.state)),
      Array.isArray(job.result) ? table([
        { label: t("export.format"), fmt: (r) => r.format }, { label: t("ui.path"), fmt: (r) => h("span", { class: "mono", text: r.path }) },
        { label: t("export.size"), num: true, fmt: (r) => num(r.size_mb, 2) }, { label: t("export.tensors"), num: true, fmt: (r) => int(r.tensors) },
      ], job.result) : null,
      job.error ? h("p", { class: "error", text: job.error }) : null, logBox(job.log)));
    return run.disabled ? 1500 : 0;
  });
}

// ── serve ──────────────────────────────────────────────────────────────
export async function renderServe(main, page) {
  const values = formState("serve", "serve");
  const root = h("div");
  const status = h("div");
  const start = button("serve.start", () => submit(start, root, async () => { paint(await post("/api/serve/start", values)); }), { primary: true });
  const stop = button("serve.stop", async () => { paint(await post("/api/serve/stop")); toast(t("serve.stopped")); }, { danger: true });
  const paint = (s) => {
    start.disabled = s.running;
    stop.disabled = !s.running;
    status.replaceChildren(card(null,
      h("div", { class: "row between" }, h("h3", { text: s.running ? t("serve.running", { url: s.url }) : t("serve.stopped") }),
        badge(s.running ? "running" : "stopped")),
      s.running ? h("pre", { class: "log", text: t("serve.hint", { addr: s.addr }) }) : null,
      s.running ? h("p", { class: "note", text: s.auth ? t("serve.auth_on") : t("serve.auth_off") }) : null,
      logBox(s.log)));
  };
  root.append(h("p", { class: "note", text: t("serve.remote_warning") }), renderGroup("serve", "serve", values, null, { summary: false }),
    h("div", { class: "row end mt" }, stop, start));
  main.append(pageHead("serve.title", "serve.intro"), card(null, root), status);
  page.poll(async () => { const s = await api("/api/serve/state"); paint(s); return s.running ? 3000 : 10000; });
}

// ── benchmark ──────────────────────────────────────────────────────────
export async function renderBenchmark(main, page) {
  const v = state.values.benchPage || (state.values.benchPage = { job: null });
  const out = h("div", { class: "stack" });
  const run = button("bench.run", async () => {
    run.disabled = true;
    try { v.job = (await post("/api/benchmark/start")).job.id; page.refresh(); } catch (err) { run.disabled = false; throw err; }
  }, { primary: true });
  main.append(pageHead("bench.title", "bench.intro", run), out);
  page.poll(async () => {
    const res = await api("/api/benchmark/results");
    let job = null;
    if (v.job) { try { job = await api(`/api/jobs/${encodeURIComponent(v.job)}?log=200`); } catch { v.job = null; } }
    run.disabled = !res.available || (job && active(job.state));
    const results = res.results || {};
    const names = Object.keys(results);
    const charts = res.metrics.filter((m) => names.some((n) => Number.isFinite(results[n][m.key]))).map((m) => card(null, barChart({
      title: t(m.label), categories: names.map((n) => results[n].name || n), fmt: (x) => num(x, 2),
      series: [{ label: t(m.label), values: names.map((n) => Number(results[n][m.key])) }],
      note: t(m.lower_is_better ? "bench.lower" : "bench.higher") })));
    out.replaceChildren(
      !res.available ? h("p", { class: "note", text: t("bench.unavailable") }) : null,
      names.length ? h("div", { class: "grid two" }, charts) : h("div", { class: "empty", text: t("bench.none") }),
      names.length ? h("p", { class: "note", text: t("bench.measured_note") }) : null,
      job ? card(null, h("div", { class: "row between" }, h("h3", { text: t("ui.log") }), badge(job.state)), logBox(job.log),
        job.error ? h("p", { class: "error", text: job.error }) : null) : null);
    return job && active(job.state) ? 2000 : 0;
  });
}
// ── auto-tune ──────────────────────────────────────────────────────────
export async function renderAutotune(main, page) {
  await ensureConfig();
  const values = formState("tune", "autotune");
  const root = h("div");
  const out = h("div", { class: "stack" });
  const start = button("tune.start", () => submit(start, root, async () => { paint(await post("/api/autotune/start", { tune: values, config: state.config })); }), { primary: true });
  const stop = button("tune.stop", async () => paint(await post("/api/autotune/stop")), { danger: true });
  const paint = (s) => {
    start.disabled = active(s.state);
    stop.disabled = !active(s.state);
    const trials = s.trials || [];
    const best = s.best || {};
    const ok = trials.filter((tr) => tr.status === "complete" && Number.isFinite(tr.loss));
    const keys = [...new Set(trials.flatMap((tr) => Object.keys(tr.params || {})))];
    out.replaceChildren(
      card(null, h("div", { class: "row between" }, h("h3", { text: t("tune.trials") }), badge(s.state)),
        ok.length ? lineChart({ title: t("tune.loss"), xLabel: t("tune.trial"), yFmt: (y) => num(y, 4),
          series: [{ label: t("tune.loss"), color: "var(--s1)", points: ok.map((tr) => [tr.trial, tr.loss]) }] }) : null,
        trials.length ? table([{ label: t("tune.trial"), num: true, fmt: (tr) => int(tr.trial) },
          { label: t("tune.loss"), num: true, fmt: (tr) => num(tr.loss, 4) },
          { label: t("ui.state"), fmt: (tr) => (tr.status && t(`trial.${tr.status}`)) || tr.status },
          { label: t("tune.duration"), num: true, fmt: (tr) => `${num(tr.duration_s, 1)} s` },
          ...keys.map((k) => ({ label: fieldLabel(k), num: true, fmt: (tr) => ((tr.params || {})[k] === undefined ? "" : num(tr.params[k], 6)) }))],
        trials.slice().reverse()) : null,
        h("p", { class: "note", text: t("tune.note") })),
      Object.keys(best).length ? card(t("tune.best"), table([{ label: t("ui.value"), fmt: (r) => fieldLabel(r[0]) }, { label: "", num: true, fmt: (r) => num(r[1], 6) }],
        Object.entries(best)), h("div", { class: "row end mt" }, button("tune.apply", () => {
        assign(state.config, best);
        toast(t("config.loaded"));
      }, { primary: true }))) : null,
      card(t("ui.log"), logBox(s.log)));
  };
  root.append(renderGroup("autotune", "autotune", values, null, { summary: false }), h("div", { class: "row end mt" }, stop, start));
  main.append(pageHead("tune.title", "tune.intro"), card(null, root), out);
  page.poll(async () => { const s = await api("/api/autotune/state"); paint(s); return active(s.state) ? 2000 : 10000; });
}

// ── IO / cache metrics of this server process ──────────────────────────
export async function renderIO(main, page) {
  const out = h("div", { class: "stack" });
  main.append(pageHead("io.title", "io.intro"), out);
  page.poll(async () => {
    const r = await api("/api/io?samples=240");
    const names = Object.keys(r.series || {});
    if (!names.length) { out.replaceChildren(h("div", { class: "empty", text: t("io.empty") })); return 4000; }
    const t0 = Math.min(...names.map((n) => (r.series[n][0] || [Infinity])[0]));
    out.replaceChildren(
      card(t("io.latest"), table([{ label: t("io.metric"), fmt: (n) => h("span", { class: "mono", text: n }) },
        { label: t("io.latest"), num: true, fmt: (n) => num(r.series[n][r.series[n].length - 1][1], 4) },
        { label: t("io.samples"), num: true, fmt: (n) => int(r.series[n].length) },
        { label: t("ui.time"), fmt: (n) => dateTime(r.series[n][r.series[n].length - 1][0]) }], names)),
      h("div", { class: "grid two" }, names.slice(0, 12).map((n) => card(null, lineChart({ title: n, xLabel: t("io.seconds"),
        series: [{ label: n, color: "var(--s1)", points: r.series[n].map(([ts, val]) => [ts - t0, val]) }], yFmt: (y) => num(y, 3) })))));
    return 3000;
  });
}
