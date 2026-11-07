// File / folder picking for path fields. Two ways in, both labelled in the user's language:
//   "Browse…"  lists the server's folders through /api/fs/list (in-page dialog).
//   "Upload…"  uses the system file manager. On the server's own machine the server opens the native
//              dialog and the real path is used in place; from another machine (LAN access) the files
//              are uploaded into <workdir>/uploads/… and that server path is used.
// kind: dir | data | checkpoint | config | any. multiple: several files/folders, one per line.
import { state, t, has, h, api, post, clear, bytes, dateTime, reportError, toast, errorText, ApiError } from "./core.js";

const lastDir = {};                         // per kind, for this browser session

function splitPaths(text) {
  return String(text || "").split(/\r?\n/).map((x) => x.trim()).filter(Boolean);
}

// Opens the picker; resolves to a list of paths, or null when cancelled.
export function openPicker({ kind = "any", multiple = false, start = "", title } = {}) {
  return new Promise((resolve) => {
    const picked = new Set();
    let view = null;
    let cursor = -1;
    const dlg = h("dialog", { class: "picker", "aria-labelledby": "picker-title" });
    const where = h("input", { type: "text", spellcheck: "false", autocomplete: "off", "aria-label": t("ui.path") });
    const list = h("ul", { class: "picker-list", role: "listbox", tabindex: "0", "aria-multiselectable": multiple ? "true" : null,
      "aria-label": t("picker.entries") });
    const roots = h("div", { class: "picker-roots", role: "group", "aria-label": t("picker.places") });
    const status = h("p", { class: "note", "aria-live": "polite" });
    const hidden = h("input", { type: "checkbox", id: "picker-hidden" });
    const choose = h("button", { type: "button", class: "primary" });
    const thisFolder = kind === "dir" || kind === "data" || kind === "any";
    const finish = (value) => { dlg.close(); dlg.remove(); resolve(value); };

    let seq = 0;                            // only the newest folder request may repaint
    const load = async (path) => {
      const mine = ++seq;
      let next;
      try {
        next = await api(`/api/fs/list?${new URLSearchParams({ path, kind, hidden: hidden.checked ? "true" : "false" })}`);
      } catch (err) {
        if (mine !== seq) return;
        if (err.status === 404 && path) return load("");
        reportError(err);
        return;
      }
      if (mine !== seq || !dlg.isConnected) return;
      view = next;
      lastDir[kind] = view.path;
      where.value = view.path;
      cursor = view.entries.length ? 0 : -1;
      paint();
    };
    const entryAt = (i) => (view && i >= 0 ? view.entries[i] : null);
    const selectable = (e) => e && (!e.dir || thisFolder);
    const toggle = (e) => {
      if (!selectable(e)) return;
      if (!multiple) picked.clear();
      if (picked.has(e.path)) picked.delete(e.path); else picked.add(e.path);
      mark();
    };
    const open = (e) => { if (e && e.dir) load(e.path); else if (e && !multiple) finish([e.path]); };
    const result = () => (picked.size ? [...picked] : (thisFolder && view ? [view.path] : []));

    // The list is rebuilt only when the folder changes; selection updates attributes in place, so a
    // double-click lands on the same element as its first click.
    function paint() {
      clear(list);
      view.entries.forEach((e, i) => {
        let before = false;                      // selection before the first click of a double-click
        const li = h("li", { role: "option", id: `pick-${i}`, class: `picker-item${e.dir ? " dir" : ""}` },
          icon(e.dir), h("span", { class: "picker-name", text: e.name }),
          h("span", { class: "picker-meta", text: e.dir ? "" : bytes(e.size) }),
          h("span", { class: "picker-meta", text: e.mtime ? dateTime(e.mtime) : "" }));
        li.addEventListener("click", (ev) => {
          cursor = i;
          if (ev.detail > 1) return;
          before = picked.has(e.path);
          if (selectable(e)) toggle(e); else mark();
        });
        li.addEventListener("dblclick", () => {
          if (e.dir) { if (!before) picked.delete(e.path); load(e.path); return; }
          if (!multiple) finish([e.path]);
        });
        list.append(li);
      });
      if (!view.entries.length) list.append(h("li", { class: "picker-empty", text: t("picker.empty") }));
      if (view.truncated) list.append(h("li", { class: "picker-empty", text: t("picker.truncated", { n: view.entries.length }) }));
      if (!roots.childElementCount) {
        roots.append(...view.roots.map((r) => h("button", { type: "button", title: r.path, onclick: () => load(r.path),
          text: r.id && has(`picker.place.${r.id}`) ? t(`picker.place.${r.id}`) : r.name })));
      }
      up.disabled = !view.parent;
      mark();
    }
    function mark() {
      list.querySelectorAll(".picker-item").forEach((li, i) => {
        const e = view.entries[i];
        li.setAttribute("aria-selected", picked.has(e.path) ? "true" : "false");
        li.classList.toggle("cursor", i === cursor);
      });
      list.setAttribute("aria-activedescendant", cursor >= 0 ? `pick-${cursor}` : "");
      list.querySelector(".cursor")?.scrollIntoView({ block: "nearest" });
      const out = result();
      choose.disabled = !out.length;
      status.textContent = picked.size
        ? (multiple ? t("picker.n_selected", { n: picked.size }) : [...picked][0])
        : (thisFolder && view ? t("picker.this_folder", { path: view.path }) : t("picker.pick_file"));
    }

    const up = h("button", { type: "button", text: "↑", "aria-label": t("picker.up"), title: t("picker.up"),
      onclick: () => view && view.parent && load(view.parent) });
    const go = h("button", { type: "button", text: t("picker.go"), onclick: () => load(where.value) });
    const cancel = h("button", { type: "button", text: t("ui.cancel"), onclick: () => finish(null) });
    choose.textContent = multiple ? t("picker.add") : t("picker.select");
    choose.addEventListener("click", () => { const out = result(); if (out.length) finish(out); });
    where.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); load(where.value); } });
    hidden.addEventListener("change", () => view && load(view.path));
    list.addEventListener("keydown", (ev) => {
      if (!view || !view.entries.length) return;
      const n = view.entries.length;
      const move = { ArrowDown: 1, ArrowUp: -1, PageDown: 10, PageUp: -10 }[ev.key];
      if (move) cursor = Math.max(0, Math.min(n - 1, cursor + move));
      else if (ev.key === "Home") cursor = 0;
      else if (ev.key === "End") cursor = n - 1;
      else if (ev.key === " ") toggle(entryAt(cursor));
      else if (ev.key === "Enter") { const e = entryAt(cursor); if (e && e.dir) open(e); else if (e) { toggle(e); if (!multiple) open(e); } }
      else if (ev.key === "Backspace") { if (view.parent) load(view.parent); }
      else return;
      ev.preventDefault();
      mark();
    });
    dlg.addEventListener("cancel", (e) => { e.preventDefault(); finish(null); });
    dlg.append(
      h("div", { class: "picker-head" }, h("h3", { id: "picker-title", text: title || t(`picker.title.${kind}`) }),
        h("button", { type: "button", class: "ghost", text: "×", "aria-label": t("ui.close"), onclick: () => finish(null) })),
      h("div", { class: "picker-bar" }, up, where, go),
      roots, list,
      h("div", { class: "picker-foot" },
        h("label", { class: "compact", for: "picker-hidden" }, hidden, t("picker.hidden")), status,
        h("div", { class: "row end" }, cancel, choose)));
    document.body.append(dlg);
    dlg.showModal();
    load(start || lastDir[kind] || "").then(() => list.focus());
  });
}

function icon(dir) {
  return h("svg", { viewBox: "0 0 16 16", width: 16, height: 16, class: "picker-icon", "aria-hidden": "true" },
    h("path", { svg: true, d: dir ? "M1.5 3.5h4.5l1.5 1.5h7v8.5h-13z" : "M3.5 1.5h6l3 3v10h-9zM9.5 1.5v3h3" }));
}

// ── shared field helpers ───────────────────────────────────────────────
const lastLine = (field) => { const c = splitPaths(field.value); return c[c.length - 1] || ""; };

// Multiple fields append (no repeats); single fields take the first path.
function putPaths(field, got, multiple, onChange) {
  const current = splitPaths(field.value);
  field.value = multiple ? [...new Set([...current, ...got])].join("\n") : got[0];
  field.dispatchEvent(new Event("input", { bubbles: true }));
  if (onChange) onChange(field.value);
  field.focus();
}

// A "Browse…" button: the in-page list of the server's folders.
export function browseButton(field, { kind = "any", multiple = false, onChange } = {}) {
  const btn = h("button", { type: "button", class: "browse", text: t(multiple ? "picker.add_button" : "ui.browse"),
    tip: t("picker.browse_tip") });
  btn.addEventListener("click", async () => {
    const got = await openPicker({ kind, multiple, start: lastLine(field) || lastDir[kind] || "" });
    if (got && got.length) putPaths(field, got, multiple, onChange);
  });
  return btn;
}

// ── "Upload…": system file manager ─────────────────────────────────────
// Mirrors chronos/workbench/files.py (KINDS, BLOCKED_SUFFIXES); the server checks again.
const ACCEPT = {
  data: [".jsonl", ".json", ".parquet", ".csv", ".tsv", ".txt"],
  checkpoint: [".pth", ".pt", ".safetensors", ".npz", ".gguf", ".json"],   // + the .config.json sidecar
  config: [".json"],
};
const WEIGHTS = [".pth", ".pt", ".safetensors", ".npz", ".gguf"];
const BLOCKED = [".py", ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe", ".bat", ".cmd", ".com", ".sh", ".command",
  ".ps1", ".app", ".pkg", ".dmg", ".jar"];
const ends = (name, list) => { const low = name.toLowerCase(); return list.some((s) => low.endsWith(s)); };
const accepts = (kind, name) => !ends(name, BLOCKED) && (!ACCEPT[kind] || ends(name, ACCEPT[kind]));
const baseName = (p) => String(p).split(/[\\/]/).filter(Boolean).pop() || String(p);

function uploadMode() {
  const s = state.session || {};
  return s.access === "local" && s.native_dialog ? "native" : "upload";
}

function batchId() {                        // [A-Za-z0-9_-]{6,48}; works on plain-HTTP LAN pages too
  const b = new Uint8Array(9);
  crypto.getRandomValues(b);
  return `${Date.now().toString(36)}-${Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("")}`;
}
// The server opens the operating system's dialog on its own desktop; the chosen paths are used in place.
async function nativePick(field, { kind, multiple, folder }) {
  const got = await post("/api/fs/native", { kind, multiple, folder, start: lastLine(field) || lastDir[kind] || "",
    title: t(`picker.title.${folder ? "dir" : kind}`) });
  return got && got.paths && got.paths.length ? got.paths : null;
}

// One file as a streamed PUT body (XHR, because fetch has no upload progress).
function putFile(file, query, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", `/api/fs/upload?${new URLSearchParams(query)}`);
    xhr.withCredentials = true;
    xhr.setRequestHeader("X-Chronos-Request", "1");
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => onProgress(e.loaded);
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch { data = null; }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new ApiError(xhr.status, data && data.detail !== undefined ? data.detail : data));
    };
    xhr.onerror = () => reject(new ApiError(0, { error: "upload_failed" }));
    xhr.onabort = () => reject(new ApiError(0, { error: "upload_failed" }));
    xhr.send(file);
  });
}

// Uploads the picked files into one server batch folder; resolves to the server paths a field should use.
async function uploadFiles(files, { kind, folder, note }) {
  const all = [...files];
  const list = all.filter((f) => accepts(kind, f.webkitRelativePath || f.name)
    && !(f.webkitRelativePath || f.name).split("/").some((p) => p.startsWith(".")));
  if (!list.length) throw new ApiError(422, { error: "upload_type" });
  const limit = ((state.session && state.session.max_upload_mb) || 20480) * 2 ** 20;
  const big = list.find((f) => f.size > limit);
  if (big) throw new ApiError(413, { error: "upload_too_large", message: big.name });
  const total = list.reduce((s, f) => s + f.size, 0);
  const batch = batchId();
  let done = 0;
  const show = (n) => { note.textContent = t("picker.uploading", { done: bytes(n), total: bytes(total) }); };
  note.hidden = false;
  show(0);
  const saved = [];
  for (const f of list) {
    const got = await putFile(f, { kind, batch, name: f.webkitRelativePath || f.name }, (n) => show(done + n));
    done += f.size;
    saved.push(got);
  }
  note.textContent = all.length > list.length
    ? t("picker.uploaded_skipped", { n: list.length, skipped: all.length - list.length })
    : t("picker.uploaded", { n: list.length });
  if (folder) {                               // the uploaded folder itself: <batch root>/<top folder>
    const top = (list[0].webkitRelativePath || "").split("/")[0];
    return [top ? `${saved[0].root}/${top}` : saved[0].root];
  }
  if (kind === "checkpoint") {                // weights + sidecar: the field takes the weights file
    const weights = saved.filter((s) => ends(s.path, WEIGHTS));
    return (weights.length ? weights : saved).map((s) => s.path);
  }
  return saved.map((s) => s.path);
}
// Which "Upload" buttons a field gets: folders only for dir, files + folders for datasets.
function uploadKinds(kind) {
  if (kind === "dir") return [true];
  if (kind === "data" || kind === "any") return [false, true];
  return [false];
}

// "Upload…" button(s) for a path field, plus the progress note they share.
export function uploadButtons(field, { kind = "any", multiple = false, onChange } = {}, note) {
  const mode = uploadMode();
  const folders = uploadKinds(kind);
  return folders.map((folder) => {
    const label = folders.length > 1 ? (folder ? "picker.upload_folder" : "picker.upload_files") : "ui.upload";
    const btn = h("button", { type: "button", class: "upload", text: t(label),
      tip: t(mode === "native" ? "picker.native_tip" : "picker.upload_tip") });
    let input = null;
    const run = async (files) => {
      btn.disabled = true;
      try {
        const got = await uploadFiles(files, { kind, folder, note });
        if (got.length) putPaths(field, got, multiple, onChange);
      } catch (err) {
        note.hidden = false;
        note.textContent = errorText(err);
        reportError(err);
      } finally {
        btn.disabled = false;
        if (input) input.value = "";
      }
    };
    btn.addEventListener("click", async () => {
      if (uploadMode() === "native") {
        btn.disabled = true;
        try {
          const got = await nativePick(field, { kind, multiple, folder });
          if (got) putPaths(field, got, multiple, onChange);
          return;
        } catch (err) {
          if (!(err instanceof ApiError) || err.code !== "native_unavailable") { reportError(err); return; }
          state.session.native_dialog = false;     // no dialog on this machine: fall back to uploading
          toast(t("picker.native_fallback"));
        } finally {
          btn.disabled = false;
        }
      }
      if (!input) {
        input = h("input", { type: "file", hidden: true, tabindex: "-1", "aria-hidden": "true",
          multiple: multiple || folder || kind === "checkpoint" ? true : null,
          accept: folder || !ACCEPT[kind] ? null : ACCEPT[kind].join(",") });
        if (folder) input.webkitdirectory = true;
        input.addEventListener("change", () => { if (input.files && input.files.length) run(input.files); });
        btn.after(input);
      }
      input.click();
    });
    return btn;
  });
}

// Input/textarea + Browse + Upload in one row. ``pick.merge`` fields say when several datasets will be
// merged into one training file; the upload note reports progress and where the files went.
export function pathControl(field, pick) {
  const status = h("p", { class: "note upload-note", "aria-live": "polite", hidden: true });
  const row = h("div", { class: "path-row" }, field, browseButton(field, pick), uploadButtons(field, pick, status));
  const box = h("div", { class: "path-box" }, row, status);
  if (!pick.merge) return box;
  const note = h("p", { class: "note path-note", "aria-live": "polite" });
  const update = () => {
    const n = splitPaths(field.value).length;
    note.textContent = n > 1 ? t("picker.will_merge", { n }) : t("picker.multi_hint");
  };
  field.addEventListener("input", update);
  update();
  box.append(note);
  return box;
}
