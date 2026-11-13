// Train page: job form, lazy-expert probe sweep (runs after every validation), job list.
import { state, t, h, has, api, post, card, pageHead, button, badge, dateTime, logBox, toast, table, int } from "./core.js";
import { renderGroup, formState, submit } from "./forms.js";
import { ensureConfig } from "./pages_config.js";

let previewTimer = 0;
const STAGE_ORDER = ["pretrain", "sft", "dpo", "orpo", "grpo", "distill"];
const stageName = (m) => (has(`stage.${m}`) ? t(`stage.${m}`) : m);

// One data row per selected stage after the first (canonical order); the first stage uses train.data_path.
function paintStageData(host, train) {
  const later = STAGE_ORDER.filter((m) => train["train.stages"].includes(m)).slice(1);
  host.hidden = !later.length;
  for (const el of host.querySelectorAll(".field[data-key^='train.data_']")) {
    el.hidden = !later.includes(el.dataset.key.slice("train.data_".length));
  }
}

async function preview(out, values) {
  try {
    const p = await post("/api/probe/preview", values);
    const text = p.requested > p.variants ? t("train.variants_capped", { n: int(p.requested), cap: int(p.variants) })
      : t("train.variants", { n: int(p.variants) });
    out.replaceChildren(h("p", { class: "note", text: values["probe.enabled"] ? text : t("train.probe_off") }),
      values["probe.enabled"] && p.labels.length ? h("details", null, h("summary", { text: t("probe.variant") }),
        h("ul", { class: "mono" }, p.labels.map((l) => h("li", { text: l })))) : null);
  } catch {
    out.replaceChildren(h("p", { class: "error", text: t("ui.invalid") }));
  }
}

export async function renderTrain(main, page) {
  await ensureConfig();
  const train = formState("train", "train");
  const probe = formState("probe", "probe");
  const root = h("div");
  const variants = h("div");
  const jobs = h("div");
  const stageData = card(t("group.stage_data"), h("p", { class: "note", text: t("train.stage_data_note") }),
    renderGroup("train", "stage_data", train, null, { summary: false }));
  const onTrain = (key) => { if (key === "train.stages") paintStageData(stageData, train); };
  const onProbe = () => { clearTimeout(previewTimer); previewTimer = setTimeout(() => preview(variants, probe), 300); };
  const start = button("train.start", () => submit(start, root, async () => {
    const r = await post("/api/train/start", { config: state.config, train, probe });
    state.values.monitorRun = r.run;
    toast(`${t("train.started")}: ${r.run}`);
    page.refresh();
  }), { primary: true });
  root.append(
    card(t("group.train"), h("p", { class: "note", text: t("train.config_note") }), renderGroup("train", "train", train, onTrain, { summary: false })),
    stageData,
    card(t("train.probe_title"), h("p", { class: "note", text: t("train.probe_note") }),
      renderGroup("probe", "probe", probe, onProbe, { summary: false }), variants),
    h("div", { class: "row end mt" }, start));
  main.append(pageHead("train.title", "train.intro"), root, card(t("train.jobs"), jobs));
  paintStageData(stageData, train);
  preview(variants, probe);
  page.poll(async () => {
    const r = await api("/api/jobs?kind=train");
    paintJobs(jobs, r.jobs || [], page);
    return (r.jobs || []).some((j) => ["running", "starting", "stopping"].includes(j.state)) ? 2000 : 8000;
  });
}

function paintJobs(host, list, page) {
  if (!list.length) { host.replaceChildren(h("p", { class: "note", text: t("ui.empty") })); return; }
  host.replaceChildren(table([
    { label: t("train.run_dir"), fmt: (j) => (j.meta && j.meta.run) || j.id },
    { label: t("ui.phase"), fmt: (j) => ((j.meta && j.meta.stages) || []).map(stageName).join(" → ") || j.title },
    { label: t("ui.state"), fmt: (j) => badge(j.state) },
    { label: t("ui.started"), fmt: (j) => dateTime(j.started) },
    { label: "", fmt: (j) => h("div", { class: "row" },
      h("a", { href: "#monitor", text: t("train.open_monitor"), onclick: () => { state.values.monitorRun = j.meta && j.meta.run; } }),
      ["running", "starting"].includes(j.state) ? button("train.stop", async (e) => {
        e.target.disabled = true;
        try { await post(`/api/jobs/${encodeURIComponent(j.id)}/stop`); } finally { page.refresh(); }
      }, { danger: true }) : null,
      button("ui.log", async () => {
        const full = await api(`/api/jobs/${encodeURIComponent(j.id)}?log=400`);
        host.append(card(`${t("ui.log")} · ${(j.meta && j.meta.run) || j.id}`, logBox(full.log),
          full.error ? h("p", { class: "error", text: full.error }) : null));
      })) },
  ], list.slice().reverse()));
}
