// Data inspector and Chronos tokenizer builder.
import { state, t, h, post, api, card, tile, pageHead, button, busy, int, num, pct, table, logBox, badge, fieldLabel } from "./core.js";
import { renderGroup, formState, submit } from "./forms.js";
import { histogram } from "./charts.js";
import { pathControl } from "./picker.js";

function pathsInput(id, value, onInput) {
  const el = h("textarea", { id, class: "paths", spellcheck: "false", autocomplete: "off", placeholder: t("picker.placeholder_multi") });
  el.value = value ?? "";
  const fit = () => { el.rows = Math.max(1, Math.min(8, el.value.split("\n").length)); };
  fit();
  el.addEventListener("input", () => { fit(); onInput(el.value); });
  return el;
}

export async function renderData(main) {
  const v = state.values.data || (state.values.data = { path: "", max: 50000 });
  const out = h("div", { class: "stack" });
  const path = pathsInput("data-path", v.path, (x) => { v.path = x; });
  const max = h("input", { type: "number", id: "data-max", min: 100, max: 500000, step: 100 });
  max.value = String(v.max);
  max.addEventListener("input", () => { v.max = Number(max.value) || 50000; });
  const scan = button("ui.scan", () => busy(scan, async () => {
    v.result = await post("/api/data/scan", { path: v.path, max_records: v.max });
    show();
  }), { primary: true });
  const show = () => {
    const r = v.result;
    if (!r) { out.replaceChildren(); return; }
    const sm = r.summary || {};
    const fields = Object.entries(sm.fields || {});
    out.replaceChildren(
      h("div", { class: "tiles" },
        tile(t("data.files"), int(sm.files), sm.size_mb !== undefined ? `${num(sm.size_mb, 1)} MB` : null),
        tile(t("data.scanned"), int(sm.scanned), sm.note === "full scan" ? t("data.full_scan") : t("data.sampled", { n: int(v.max) })),
        tile(t("data.median"), int(sm.char_median), t("data.chars")),
        tile(t("data.p95"), int(sm.char_p95), t("data.chars")),
        tile(t("data.dups"), pct(sm.dup_ratio))),
      card(null, histogram({ title: t("data.hist"), edges: r.histogram.edges, counts: r.histogram.counts, xLabel: t("data.chars") })),
      h("div", { class: "grid two" },
        card(t("data.schema"), table([{ label: t("data.schema"), key: "k" }, { label: t("data.records"), key: "n", num: true }],
          Object.entries(sm.schemas || {}).map(([k, n]) => ({ k, n: int(n) })))),
        card(t("data.fields"), table([{ label: t("data.fields"), key: "k" }, { label: t("data.type"), key: "type" },
          { label: t("data.present"), num: true, fmt: (row) => pct(row.present) }],
        fields.map(([k, f]) => ({ k, type: f.type, present: f.present }))))),
      Object.keys(sm.scripts || {}).length ? card(t("data.scripts"), table([{ label: t("data.script"), key: "k" },
        { label: t("data.share"), num: true, fmt: (row) => pct(row.share) }], Object.entries(sm.scripts).map(([k, share]) => ({ k, share })))) : null,
      card(t("data.samples"), ...(r.samples || []).map((x) => h("div", { class: "output mt", text: x }))),
      sm.read_errors ? card(t("ui.error"), logBox(sm.read_errors)) : null);
  };
  main.append(pageHead("data.title", "data.intro"),
    card(null, h("div", { class: "fields" },
      h("div", { class: "field wide", tip: t("data.path_help") }, h("label", { class: "label", for: "data-path", text: t("ui.path") }),
        pathControl(path, { kind: "data", multiple: true })),
      h("div", { class: "field", tip: t("data.max_help") }, h("label", { class: "label", for: "data-max", text: t("data.max_records") }), max)),
    h("div", { class: "row mt" }, scan)),
    out);
  show();
}

// ── tokenizer builder ──────────────────────────────────────────────────
export async function renderTokenizer(main, page) {
  const values = formState("tok", "tokenizer");
  const root = h("div");
  const status = h("div", { class: "stack" });
  const start = button("ui.start", () => submit(start, root, async () => { paint(await post("/api/tokenizer/start", values)); }), { primary: true });
  const paint = (s) => {
    start.disabled = s.state === "running";
    const sm = s.summary || {};
    status.replaceChildren(
      h("div", { class: "row between" }, h("h3", { text: t("tok.progress") }), badge(s.state)),
      s.error ? h("p", { class: "error", text: s.error }) : null,
      logBox(s.log),
      s.state === "done" ? h("div", { class: "tiles" },
        tile(fieldLabel("tok.vocab_size"), int(sm.vocab_size)),
        tile(t("tok.vocab_use"), pct(sm.vocab_utilization)),
        tile(t("tok.algorithm"), sm.algorithm || "–"),
        tile(t("tok.fingerprint"), sm.fingerprint ? String(sm.fingerprint).slice(0, 12) : "–"),
        tile(t("tok.elapsed"), sm.elapsed_s === undefined ? "–" : `${num(sm.elapsed_s, 1)} s`)) : null,
      s.rows && s.rows.length ? h("div", null, h("h4", { text: t("tok.report") }), table([
        { label: t("tok.bucket"), fmt: (r) => r[0] }, { label: t("tok.texts"), num: true, fmt: (r) => int(r[1]) },
        { label: t("tok.compression"), num: true, fmt: (r) => num(r[2], 2) }, { label: t("tok.chars_per_token"), num: true, fmt: (r) => num(r[3], 2) },
        { label: t("tok.fertility"), num: true, fmt: (r) => num(r[4], 2) }, { label: t("tok.byte_fallback"), num: true, fmt: (r) => pct(r[5], 2) },
        { label: t("tok.roundtrip"), num: true, fmt: (r) => (r[6] === 0 ? t("ui.yes") : `${t("ui.no")} (${int(r[6])})`) }], s.rows),
      h("p", { class: "note", text: t("tok.report_note") })) : null,
      s.out_dir ? h("p", { class: "note", text: `${t("tok.out_dir_now")}: ${s.out_dir}` }) : null);
  };
  root.append(renderGroup("tokenizer", "tokenizer", values, null, { summary: false }), h("div", { class: "row mt" }, start));
  main.append(pageHead("tok.title", "tok.intro"), card(null, root), card(null, status));
  const poll = async () => { const s = await api("/api/tokenizer/state"); paint(s); return s.state === "running" ? 1500 : 6000; };
  page.poll(poll);
}

