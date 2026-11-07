// Inline-SVG charts: line (crosshair + tooltip listing every series), horizontal
// bars, histogram. Each chart has a legend (>= 2 series) and a table view.
// Colors come from CSS custom properties, so light and dark are both selected.
import { h, t, num, clear, showTipAt, hideTip, SERIES } from "./core.js";

const NS = "http://www.w3.org/2000/svg";
export function s(tag, attrs, ...kids) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null) continue;
    // var() colors go through CSSOM: presentation attributes do not resolve custom properties everywhere,
    // and the CSP forbids style attributes.
    if ((k === "fill" || k === "stroke") && String(v).startsWith("var(")) el.style[k] = v;
    else el.setAttribute(k, String(v));
  }
  for (const kid of kids) if (kid !== null && kid !== undefined) el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return el;
}
export const color = (i) => `var(${SERIES[((i % SERIES.length) + SERIES.length) % SERIES.length]})`;

function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const n = raw / mag;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * mag;
}
export function ticks(lo, hi, count = 4) {
  if (!(hi > lo)) { const pad = Math.abs(lo) * 0.1 || 1; lo -= pad; hi += pad; }
  const step = niceStep(hi - lo, count);
  const a = Math.floor(lo / step) * step; const b = Math.ceil(hi / step) * step;
  const values = [];
  for (let v = a; v <= b + step * 1e-9; v += step) values.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { lo: a, hi: b, values };
}
const extent = (vals) => [Math.min(...vals), Math.max(...vals)];
const textWidth = (str) => [...String(str)].reduce((w, ch) => w + (ch.charCodeAt(0) > 0x2e7f ? 11 : 6.4), 0);

// Frame shared by every chart: title, legend, chart/table toggle, resize.
function frame({ title, legend, draw, tableOf, note }) {
  const body = h("div", { class: "chart" });
  const head = h("div", { class: "chart-head" });
  const view = { table: false };
  const toggle = h("button", { type: "button", class: "ghost", text: t("ui.table_view"),
    onclick: () => { view.table = !view.table; toggle.textContent = t(view.table ? "ui.chart_view" : "ui.table_view"); paint(); } });
  head.append(h("h4", { text: title || "" }),
    h("div", { class: "row" }, legend && legend.length >= 2 ? h("div", { class: "legend" }, legend.map((l) => {
      const sw = h("i", { class: l.box ? "box" : null }); sw.style.background = l.color;
      return h("span", null, sw, l.label);
    })) : null, toggle));
  const wrap = h("div", { class: "chart-block" }, head, body, note ? h("p", { class: "note", text: note }) : null);
  let lastWidth = 0;
  const paint = () => {
    clear(body);
    if (view.table) { body.append(tableOf()); return; }
    const width = Math.max(260, Math.floor(body.clientWidth || wrap.clientWidth || 640));
    lastWidth = width;
    body.append(draw(width));
  };
  const ro = new ResizeObserver(() => {
    const w = Math.floor(body.clientWidth);
    if (!view.table && w && Math.abs(w - lastWidth) > 4) paint();
  });
  ro.observe(body);
  requestAnimationFrame(paint);
  return wrap;
}

export function emptyState(text) { return h("div", { class: "empty", text: text || t("ui.empty") }); }
function tipBody(title, rows) {
  return h("div", null, h("div", { class: "t-title", text: title }), rows.map((r) => {
    const key = h("i", { class: "key" }); key.style.background = r.color;
    return h("div", { class: "t-row" }, key, h("span", { text: r.label }), h("b", { text: r.value }));
  }));
}

// series: [{label, color?, points: [[x, y], ...]}]; colors follow the entity (index given by caller).
export function lineChart({ title, series, xLabel, yFmt = (v) => num(v, 3), xFmt = (v) => num(v, 0), height = 220, note, empty }) {
  const live = (series || []).map((sr, i) => ({ ...sr, color: sr.color || color(i),
    points: (sr.points || []).filter((p) => p && Number.isFinite(p[0]) && Number.isFinite(p[1])).sort((a, b) => a[0] - b[0]) }))
    .filter((sr) => sr.points.length);
  if (!live.length) return h("div", { class: "chart-block" }, h("h4", { text: title || "" }), emptyState(empty));
  const xs = [...new Set(live.flatMap((sr) => sr.points.map((p) => p[0])))].sort((a, b) => a - b);
  const [y0, y1] = extent(live.flatMap((sr) => sr.points.map((p) => p[1])));
  const yt = ticks(y0, y1, 4);
  const at = (sr, x) => {           // nearest point of one series to x
    let lo = 0; let hi = sr.points.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (sr.points[mid][0] < x) lo = mid; else hi = mid; }
    return Math.abs(sr.points[lo][0] - x) <= Math.abs(sr.points[hi][0] - x) ? sr.points[lo] : sr.points[hi];
  };
  const draw = (width) => {
    const left = Math.max(...yt.values.map((v) => textWidth(yFmt(v)))) + 12;
    const m = { l: left, r: 14, t: 10, b: xLabel ? 40 : 26 };
    const W = width - m.l - m.r; const H = height - m.t - m.b;
    const [xa, xb] = xs.length > 1 ? [xs[0], xs[xs.length - 1]] : [xs[0] - 1, xs[0] + 1];
    const X = (x) => m.l + ((x - xa) / (xb - xa)) * W;
    const Y = (y) => m.t + H - ((y - yt.lo) / (yt.hi - yt.lo)) * H;
    const svg = s("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": title || "" });
    for (const v of yt.values) {
      svg.append(s("line", { class: "grid-line", x1: m.l, x2: m.l + W, y1: Y(v), y2: Y(v) }),
        s("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, yFmt(v)));
    }
    svg.append(s("line", { class: "axis-line", x1: m.l, x2: m.l + W, y1: m.t + H, y2: m.t + H }));
    const xt = ticks(xa, xb, Math.max(2, Math.floor(W / 110))).values.filter((v) => v >= xa && v <= xb);
    for (const v of xt) svg.append(s("text", { x: X(v), y: m.t + H + 16, "text-anchor": "middle" }, xFmt(v)));
    if (xLabel) svg.append(s("text", { x: m.l + W / 2, y: height - 6, "text-anchor": "middle" }, xLabel));
    for (const sr of live) {
      const d = sr.points.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
      svg.append(s("path", { class: "series", d, stroke: sr.color }));
      const last = sr.points[sr.points.length - 1];
      svg.append(s("circle", { class: "end-dot", cx: X(last[0]), cy: Y(last[1]), r: 4, fill: sr.color }));
    }
    const cross = s("line", { class: "crosshair", y1: m.t, y2: m.t + H, visibility: "hidden" });
    const dots = live.map((sr) => s("circle", { class: "end-dot", r: 4, fill: sr.color, visibility: "hidden" }));
    svg.append(cross, ...dots);
    const overlay = s("rect", { x: m.l, y: m.t, width: W, height: H, fill: "transparent", tabindex: 0,
      "aria-label": `${title || ""}: ${t("ui.table_view")}` });
    let index = xs.length - 1;
    const show = (i, cx, cy) => {
      index = Math.max(0, Math.min(xs.length - 1, i));
      const x = xs[index];
      cross.setAttribute("x1", X(x)); cross.setAttribute("x2", X(x)); cross.setAttribute("visibility", "visible");
      const rows = live.map((sr, j) => {
        const p = at(sr, x);
        dots[j].setAttribute("cx", X(p[0])); dots[j].setAttribute("cy", Y(p[1])); dots[j].setAttribute("visibility", "visible");
        return { color: sr.color, label: sr.label, value: yFmt(p[1]) };
      });
      const box = svg.getBoundingClientRect();
      showTipAt(cx ?? box.left + X(x), cy ?? box.top + m.t, tipBody(`${xLabel ? `${xLabel} ` : ""}${xFmt(x)}`, rows));
    };
    const hide = () => { cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); hideTip(); };
    overlay.addEventListener("pointermove", (e) => {
      const box = svg.getBoundingClientRect();
      const x = xa + ((e.clientX - box.left - m.l) / W) * (xb - xa);
      let best = 0;
      for (let i = 1; i < xs.length; i += 1) if (Math.abs(xs[i] - x) < Math.abs(xs[best] - x)) best = i;
      show(best, e.clientX, e.clientY);
    });
    overlay.addEventListener("pointerleave", hide);
    overlay.addEventListener("blur", hide);
    overlay.addEventListener("focus", () => show(index));
    overlay.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft") { show(index - 1); e.preventDefault(); }
      if (e.key === "ArrowRight") { show(index + 1); e.preventDefault(); }
    });
    svg.append(overlay);
    return svg;
  };
  const tableOf = () => {
    const rows = xs.slice(-500).map((x) => [x, ...live.map((sr) => { const p = at(sr, x); return p[0] === x ? p[1] : null; })]);
    return h("div", { class: "table-wrap" }, h("table", null,
      h("thead", null, h("tr", null, h("th", { class: "num", text: xLabel || "x" }), live.map((sr) => h("th", { class: "num", text: sr.label })))),
      h("tbody", null, rows.map((r) => h("tr", null, h("td", { class: "num", text: xFmt(r[0]) }),
        r.slice(1).map((v) => h("td", { class: "num", text: v === null ? "–" : yFmt(v) })))))));
  };
  return frame({ title, legend: live.map((sr) => ({ label: sr.label, color: sr.color })), draw, tableOf, note });
}
// Bar with a 4px rounded data end; the baseline end stays square.
function barPath(x, y, w, hgt, horizontal) {
  const r = Math.min(4, Math.abs(horizontal ? w : hgt) / 2, (horizontal ? hgt : w) / 2);
  if (horizontal) {
    if (w <= 0) return "";
    return `M${x},${y}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${hgt - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-(w - r)}z`;
  }
  if (hgt <= 0) return "";
  return `M${x},${y + hgt}v${-(hgt - r)}a${r},${r} 0 0 1 ${r},${-r}h${w - 2 * r}a${r},${r} 0 0 1 ${r},${r}v${hgt - r}z`;
}

function markTip(el, content) {
  el.setAttribute("tabindex", "0");
  const show = (e) => {
    const box = el.getBoundingClientRect();
    showTipAt(e && e.clientX !== undefined ? e.clientX : box.right, e && e.clientY !== undefined ? e.clientY : box.top, content());
  };
  el.addEventListener("pointermove", show);
  el.addEventListener("focus", () => show());
  el.addEventListener("pointerleave", () => hideTip());
  el.addEventListener("blur", () => hideTip());
}

// Horizontal grouped bars: categories down the side, one bar per series.
export function barChart({ title, categories, series, fmt = (v) => num(v, 2), note, empty }) {
  const live = (series || []).map((sr, i) => ({ ...sr, color: sr.color || color(i) }));
  const values = live.flatMap((sr) => sr.values.filter((v) => Number.isFinite(v)));
  if (!values.length) return h("div", { class: "chart-block" }, h("h4", { text: title || "" }), emptyState(empty));
  const bar = live.length > 1 ? 14 : 18; const gap = 2; const groupGap = 14;
  const groupH = live.length * bar + (live.length - 1) * gap;
  const height = categories.length * (groupH + groupGap) + 24;
  const max = Math.max(0, ...values); const min = Math.min(0, ...values);
  const xt = ticks(min, max, 4);
  const draw = (width) => {
    const left = Math.min(220, Math.max(...categories.map(textWidth)) + 14);
    const right = Math.max(...values.map((v) => textWidth(fmt(v)))) + 14;
    const W = width - left - right;
    const X = (v) => left + ((v - xt.lo) / (xt.hi - xt.lo)) * W;
    const svg = s("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": title || "" });
    for (const v of xt.values) svg.append(s("line", { class: "grid-line", x1: X(v), x2: X(v), y1: 0, y2: height - 20 }),
      s("text", { x: X(v), y: height - 6, "text-anchor": "middle" }, fmt(v)));
    categories.forEach((cat, ci) => {
      const gy = ci * (groupH + groupGap) + 4;
      svg.append(s("text", { x: left - 10, y: gy + groupH / 2 + 4, "text-anchor": "end" }, cat));
      live.forEach((sr, si) => {
        const v = sr.values[ci];
        if (!Number.isFinite(v)) return;
        const y = gy + si * (bar + gap);
        const x0 = X(Math.max(0, xt.lo)); const x1 = X(v);
        const el = s("path", { class: "bar", d: barPath(Math.min(x0, x1), y, Math.max(1, Math.abs(x1 - x0)), bar, true), fill: sr.color });
        markTip(el, () => tipBody(cat, [{ color: sr.color, label: sr.label, value: fmt(v) }]));
        svg.append(el, s("text", { x: Math.max(x0, x1) + 6, y: y + bar / 2 + 4 }, fmt(v)));
      });
    });
    svg.append(s("line", { class: "axis-line", x1: X(Math.max(0, xt.lo)), x2: X(Math.max(0, xt.lo)), y1: 0, y2: height - 20 }));
    return svg;
  };
  const tableOf = () => h("div", { class: "table-wrap" }, h("table", null,
    h("thead", null, h("tr", null, h("th", { text: "" }), live.map((sr) => h("th", { class: "num", text: sr.label })))),
    h("tbody", null, categories.map((cat, ci) => h("tr", null, h("td", { text: cat }),
      live.map((sr) => h("td", { class: "num", text: Number.isFinite(sr.values[ci]) ? fmt(sr.values[ci]) : "–" })))))));
  return frame({ title, legend: live.map((sr) => ({ label: sr.label, color: sr.color, box: true })), draw, tableOf, note });
}

// Histogram over (possibly log-spaced) bin edges; one hue, per-bar tooltip.
export function histogram({ title, edges, counts, xLabel, height = 200, note, empty }) {
  if (!counts || !counts.length) return h("div", { class: "chart-block" }, h("h4", { text: title || "" }), emptyState(empty));
  const yt = ticks(0, Math.max(...counts), 4);
  const draw = (width) => {
    const left = Math.max(...yt.values.map((v) => textWidth(num(v, 0)))) + 12;
    const m = { l: left, r: 10, t: 8, b: xLabel ? 40 : 26 };
    const W = width - m.l - m.r; const H = height - m.t - m.b;
    const bw = W / counts.length;
    const Y = (v) => m.t + H - (v / yt.hi) * H;
    const svg = s("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": title || "" });
    for (const v of yt.values) svg.append(s("line", { class: "grid-line", x1: m.l, x2: m.l + W, y1: Y(v), y2: Y(v) }),
      s("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, num(v, 0)));
    counts.forEach((c, i) => {
      const x = m.l + i * bw + 1; const w = Math.max(1, bw - 2);
      const el = s("path", { class: "bar", d: barPath(x, Y(c), w, m.t + H - Y(c), false) || `M${x},${m.t + H}h${w}v-1h${-w}z`, fill: "var(--s1)" });
      markTip(el, () => tipBody(`${num(edges[i], 0)} – ${num(edges[i + 1], 0)}`, [{ color: "var(--s1)", label: t("ui.value"), value: num(c, 0) }]));
      svg.append(el);
    });
    svg.append(s("line", { class: "axis-line", x1: m.l, x2: m.l + W, y1: m.t + H, y2: m.t + H }));
    const every = Math.max(1, Math.ceil(counts.length / Math.max(2, Math.floor(W / 70))));
    edges.forEach((e, i) => { if (i % every === 0 || i === edges.length - 1) svg.append(s("text", { x: m.l + i * bw, y: m.t + H + 16, "text-anchor": "middle" }, num(e, 0))); });
    if (xLabel) svg.append(s("text", { x: m.l + W / 2, y: height - 6, "text-anchor": "middle" }, xLabel));
    return svg;
  };
  const tableOf = () => h("div", { class: "table-wrap" }, h("table", null,
    h("thead", null, h("tr", null, h("th", { class: "num", text: xLabel || "" }), h("th", { class: "num", text: t("ui.value") }))),
    h("tbody", null, counts.map((c, i) => h("tr", null, h("td", { class: "num", text: `${num(edges[i], 0)} – ${num(edges[i + 1], 0)}` }), h("td", { class: "num", text: num(c, 0) }))))));
  return frame({ title, legend: null, draw, tableOf, note });
}
