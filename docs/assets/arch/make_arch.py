#!/usr/bin/env python3
"""Chronos v3 architecture artwork, in physical millimetres.

Usage:
    python3 make_arch.py
    python3 make_arch.py --outdir ./figures --dpi 600
    python3 make_arch.py --config architecture_config.json --formats svg pdf

The forward figure contains its block interiors. Input and output use separate
lanes. Solid arrows carry data, orange dashes carry expert gates, and the
violet return carries recurrent state. Expert branches are parallel.

Architecture is traced to Project_Chronos-0.3.3a1 source. Defaults are symbolic.
The four cards are supported configuration variants, not a fixed ABCE schedule.
The SSD kernel diagram is generated entirely by this code.
Requires Python >= 3.9 and matplotlib >= 3.6. No network, TeX or image assets.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

# Avoid creating a font cache in a read-only home directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "chronos-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

MM = 1 / 25.4
OUT = Path(__file__).resolve().parent

# Colours encode operations, not a presumed stage order. Attention scope and
# dense, routed, shared computation use the same colour roles in every figure.
COLORS = {
    "ink": "#253243", "muted": "#617080", "rule": "#CFD7DF",
    "paper": "#FFFFFF", "panel": "#F7F9FB", "io": "#F1F4F7",
    "norm": "#F4EBD7", "norm_edge": "#A28D60",
    "attn": "#DBE9F8", "attn_edge": "#4977A7",
    "local": "#D9EDE8", "local_edge": "#367D73",
    "ffn": "#E8E1F3", "ffn_edge": "#8A73AA",
    "moe": "#F8E4D8", "moe_edge": "#B67C56",
    "router": "#F5DDCB", "router_edge": "#B67C56",
    "expert": "#F8E8DC", "expert_edge": "#B67C56",
    "shared": "#E4EDD7", "shared_edge": "#7C9663",
    "recur": "#7557A4", "recur_bg": "#F4F0F8",
}


@dataclass(frozen=True)
class Architecture:
    """Symbolic default. Field names match the actual Chronos config."""
    model_name: str = "Chronos v3"
    hidden_size: object = "d"
    prelude_layers: object = "P"
    recurrent_groups: object = "G"
    coda_layers: object = "C"
    recurrent_depth: object = "R"
    num_experts: object = "E"
    num_experts_per_tok: object = "k"
    num_shared_experts: object = "S"
    sliding_window_size: object = "W"
    intermediate_size: object = r"d_{\mathrm{ff}}"
    moe_intermediate_size: object = r"d_{\mathrm{e}}"
    lookahead_steps: object = "K"
    tie_word_embeddings: object = None
    use_moe: object = None
    recurrent_order: object = None

    def __post_init__(self):
        if isinstance(self.num_experts_per_tok, int) and isinstance(self.num_experts, int):
            if not 0 < self.num_experts_per_tok <= self.num_experts:
                raise ValueError("num_experts_per_tok must be in [1, num_experts]")

    @classmethod
    def from_mapping(cls, values):
        from dataclasses import fields
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in names})


@dataclass(frozen=True)
class BlockSpec:
    key: str
    title: str
    color: str
    attention: str
    moe: bool


# Supported configuration combinations, NOT four serial stage identities.
# Actual physical parity selects MLA/SWA; use_moe selects the MLP globally.
BLOCKS = (
    BlockSpec("I", "MLA + Dense", "attn", "MLA", False),
    BlockSpec("II", "MLA + MoE", "attn", "MLA", True),
    BlockSpec("III", "SWA + Dense", "local", "SWA", False),
    BlockSpec("IV", "SWA + MoE", "local", "SWA", True),
)


def _style():
    rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 6.5,
        "mathtext.fontset": "dejavusans", "text.color": COLORS["ink"],
        "axes.linewidth": 0.6, "lines.linewidth": 0.7,
        "svg.fonttype": "none",    # SVG labels remain editable text.
        "pdf.fonttype": 42,        # Embed TrueType fonts in PDF.
        "ps.fonttype": 42, "savefig.facecolor": "white",
        "figure.facecolor": "white", "path.simplify": False,
    })


class Drawing:
    """One millimetre in data space is one millimetre on the exported page.

    The y axis runs down the page. This avoids square-axis distortion and
    makes typography, gutters and corner radii independent of aspect ratio.
    All labels are registered for bounds and node-containment checks.
    """

    def __init__(self, width: float, height: float, name: str):
        _style()
        self.width, self.height, self.name = width, height, name
        self.fig = plt.figure(figsize=(width * MM, height * MM), dpi=160)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
        self.ax.axis("off")
        self.texts = []
        self.nodes = []
        self.routes = []

    def text(self, x, y, value, size=6.5, *, weight="normal", color="ink",
             ha="center", va="center", parent=None, rotation=0):
        size = max(6.0, size)  # Minimum at the actual 180 mm print width.
        artist = self.ax.text(x, y, value, ha=ha, va=va, fontsize=size,
                              fontweight=weight, color=COLORS.get(color, color),
                              linespacing=1.18, rotation=rotation, zorder=8)
        self.texts.append((artist, parent))
        return artist

    def frame(self, x, y, w, h, *, fill="paper", edge="rule", radius=1.25,
              lw=0.6, dashed=False):
        patch = FancyBboxPatch((x, y), w, h,
                              boxstyle=f"round,pad=0,rounding_size={radius}",
                              facecolor=COLORS.get(fill, fill),
                              edgecolor=COLORS.get(edge, edge), linewidth=lw,
                              linestyle=(0, (3.0, 2.1)) if dashed else "-",
                              zorder=1)
        self.ax.add_patch(patch)
        return patch

    def node(self, x, y, w, h, title, *, kind="io", sub=None, size=6.25,
             subsize=5.6, weight="normal", radius=0.85):
        parent = (x, y, w, h, title)
        self.nodes.append(parent)
        self.frame(x, y, w, h, fill=kind,
                   edge=kind + "_edge" if kind + "_edge" in COLORS else "rule",
                   radius=radius, lw=0.62)
        if sub is None:
            self.text(x+w/2, y+h/2, title, size, weight=weight, parent=parent)
        else:
            self.text(x+w/2, y+h*0.34, title, size, weight=weight, parent=parent)
            self.text(x+w/2, y+h*0.74, sub, subsize, color="muted", parent=parent)

    def route(self, points: Sequence[tuple[float, float]], *, color="ink",
              lw=0.72, arrow=True, dashed=False, zorder=4):
        """Draw an orthogonal polyline with one consistent terminal arrow."""
        if len(points) < 2:
            raise ValueError("A route needs at least two points")
        for p, q in zip(points, points[1:]):
            if p[0] != q[0] and p[1] != q[1]:
                raise ValueError(f"Non-orthogonal route segment: {p} -> {q}")
        self.routes.append(tuple(points))
        col = COLORS.get(color, color)
        if len(points) > 2:
            self.ax.plot(*zip(*points[:-1]), color=col, lw=lw,
                         linestyle=(0, (3.0, 2.1)) if dashed else "-",
                         solid_capstyle="round", solid_joinstyle="round", zorder=zorder)
        if arrow:
            self.ax.add_patch(FancyArrowPatch(points[-2], points[-1],
                              arrowstyle="-|>", mutation_scale=5.7,
                              shrinkA=0, shrinkB=0, color=col, linewidth=lw,
                              linestyle=(0, (3.0, 2.1)) if dashed else "-",
                              zorder=zorder))
        else:
            self.ax.plot(*zip(*points[-2:]), color=col, lw=lw,
                         linestyle=(0, (3.0, 2.1)) if dashed else "-", zorder=zorder)

    def dot(self, x, y, *, color="ink", radius=0.42):
        self.ax.add_patch(Circle((x, y), radius, fc=COLORS[color], ec="none", zorder=6))

    def add(self, x, y, *, r=1.05, sigma=False):
        self.ax.add_patch(Circle((x, y), r, fc="white", ec=COLORS["ink"],
                                  lw=0.65, zorder=6))
        if sigma:
            self.text(x, y+0.02, r"$\Sigma$", 6.0)
        else:
            self.ax.plot([x-r*.52, x+r*.52], [y, y], lw=.65, color=COLORS["ink"], zorder=7)
            self.ax.plot([x, x], [y-r*.52, y+r*.52], lw=.65, color=COLORS["ink"], zorder=7)

    def audit(self):
        """Renderer-based checks in physical space, without tight-crop masking."""
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        inverse = self.ax.transData.inverted()
        issues = []
        min_size = float("inf")
        text_bounds = []
        for artist, parent in self.texts:
            box = artist.get_window_extent(renderer=renderer)
            pts = inverse.transform(((box.x0, box.y0), (box.x1, box.y1)))
            x0, x1 = min(pts[:, 0]), max(pts[:, 0])
            y0, y1 = min(pts[:, 1]), max(pts[:, 1])
            min_size = min(min_size, artist.get_fontsize())
            text_bounds.append((artist.get_text(), (x0, y0, x1, y1)))
            if x0 < -0.05 or x1 > self.width+.05 or y0 < -.05 or y1 > self.height+.05:
                issues.append({"kind": "page_overflow", "text": artist.get_text()})
            if parent:
                x, y, w, h, title = parent
                # 0.18 mm clearance also catches cramped labels before export.
                pad = .18
                if x0 < x+pad or x1 > x+w-pad or y0 < y+pad or y1 > y+h-pad:
                    issues.append({"kind": "node_text_overflow", "text": artist.get_text(),
                                   "node": title, "text_bounds_mm": [x0, y0, x1, y1]})
        for i, (name_a, a) in enumerate(text_bounds):
            for name_b, b in text_bounds[i+1:]:
                if min(a[2], b[2])-max(a[0], b[0]) > .12 and min(a[3], b[3])-max(a[1], b[1]) > .12:
                    issues.append({"kind": "text_overlap", "texts": [name_a, name_b]})
        # Wires may touch node boundaries, but may not cut through a node.
        # Test the interior of every orthogonal segment against every box.
        for route in self.routes:
            for a, b in zip(route, route[1:]):
                for x, y, w, h, title in self.nodes:
                    pad = .12
                    if a[0] == b[0]:
                        crosses = (x+pad < a[0] < x+w-pad and
                                   min(max(a[1], b[1]), y+h-pad) > max(min(a[1], b[1]), y+pad))
                    else:
                        crosses = (y+pad < a[1] < y+h-pad and
                                   min(max(a[0], b[0]), x+w-pad) > max(min(a[0], b[0]), x+pad))
                    if crosses:
                        issues.append({"kind": "wire_crosses_node", "node": title,
                                       "segment": [a, b]})
        return {"figure": self.name, "size_mm": [self.width, self.height],
                "text_count": len(self.texts), "node_count": len(self.nodes),
                "route_count": len(self.routes),
                "minimum_font_pt": min_size, "issues": issues}

    def save(self, outdir: Path, formats=("png", "svg", "pdf"), dpi=600):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        report = self.audit()
        # A failed layout is not published as a seemingly successful export.
        if report["issues"]:
            plt.close(self.fig)
            raise RuntimeError(json.dumps(report, indent=2, ensure_ascii=False))
        for ext in formats:
            path = outdir / f"{self.name}.{ext}"
            metadata = {"Creator": "Chronos architecture renderer"} if ext == "pdf" else None
            self.fig.savefig(path, dpi=dpi, bbox_inches=None, pad_inches=0, metadata=metadata)
            print(path)
            if ext == "svg":
                # Portable publication copy: no recipient-side font lookup.
                # Keep the editable text SVG above for author revisions.
                outline_path = outdir / f"{self.name}_outlined.svg"
                with plt.rc_context({"svg.fonttype": "path"}):
                    self.fig.savefig(outline_path, bbox_inches=None, pad_inches=0)
                print(outline_path)
        plt.close(self.fig)
        return report


def _header(d, cfg, subtitle):
    d.text(4, 6.8, cfg.model_name, 13.2, weight="bold", ha="left")
    d.text(176, 6.8, subtitle, 7.2, color="muted", ha="right")
    d.route([(4, 13.3), (176, 13.3)], arrow=False, color="ink", lw=.85)


def _block_interior(d, x, y, w, block, cfg, *, height=81):
    """True ChronosBlock: norm-attention-residual, norm-MLP-residual.

    All four supported variants contain attention. In one model, use_moe
    chooses one MLP family globally; physical layer parity picks MLA/SWA.
    """
    X = lambda v: x+v*w/30
    Y = lambda v: y+v*height/81
    c, left = X(17), X(2.5)
    font, small = 6.7, 6.0
    d.frame(x, y, w, height, edge=block.color+"_edge")
    d.text(X(15), Y(4), block.key+"   "+block.title, 7.0, weight="bold")
    subtitle = "Global · latent KV" if block.attention == "MLA" else f"Local · window {cfg.sliding_window_size}"
    d.text(X(15), Y(8), subtitle, small, color="muted")
    d.route([(X(2), Y(11)), (X(28), Y(11))], arrow=False, color="rule", lw=.5)
    d.text(c, Y(14), r"$h$", font)
    d.route([(c,Y(15.3)),(c,Y(18))])
    d.dot(c,Y(16.4))
    d.node(X(5.5),Y(18),23*w/30,5*height/81,"RMSNorm",kind="norm",size=font)
    d.route([(c,Y(23)),(c,Y(26))])
    d.node(X(5.5),Y(26),23*w/30,7*height/81,
           "MLA" if block.attention=="MLA" else "Sliding-window",kind=block.color,
           sub="full causal" if block.attention=="MLA" else "attention",size=font,subsize=small)
    d.route([(c,Y(33)),(c,Y(36.5)-1.05)])
    d.route([(c,Y(16.4)),(left,Y(16.4)),(left,Y(36.5)),(c-1.05,Y(36.5))])
    d.add(c,Y(36.5))
    d.route([(c,Y(36.5)+1.05),(c,Y(40))])
    d.dot(c,Y(38.6))
    d.node(X(5.5),Y(40),23*w/30,5*height/81,"RMSNorm",kind="norm",size=font)
    if block.moe:
        sc, rc = X(10),X(22.5)
        d.route([(c,Y(45)),(c,Y(48))],arrow=False)
        d.dot(c,Y(48))
        d.route([(c,Y(48)),(rc,Y(48)),(rc,Y(50))])
        d.node(X(16.5),Y(50),12*w/30,7*height/81,"Router",kind="router",
               sub="FP32 softmax",size=6.5,subsize=6)
        d.route([(rc,Y(57)),(rc,Y(60))],color="router_edge",dashed=True)
        d.dot(rc,Y(48))
        d.route([(rc,Y(48)),(X(29.3),Y(48)),(X(29.3),Y(64)),(X(28.5),Y(64))])
        d.node(X(16.5),Y(60),12*w/30,7*height/81,f"Top-{cfg.num_experts_per_tok}",
               kind="expert",sub=f"of {cfg.num_experts}",size=6.5,subsize=6)
        d.route([(c,Y(48)),(sc,Y(48)),(sc,Y(60))])
        d.node(X(5.5),Y(60),9*w/30,7*height/81,"Shared",kind="shared",
               sub=f"mean ×{cfg.num_shared_experts}",size=6,subsize=6)
        d.route([(sc,Y(67)),(sc,Y(70)),(c-1.05,Y(70))])
        d.route([(rc,Y(67)),(rc,Y(70)),(c+1.05,Y(70))])
        d.add(c,Y(70),sigma=True)
        d.route([(c,Y(70)+1.05),(c,Y(74)-1.05)])
    else:
        d.route([(c,Y(45)),(c,Y(52))])
        d.node(X(5.5),Y(52),23*w/30,14*height/81,"Gated FFN",kind="ffn",
               sub=rf"$d_{{\mathrm{{ff}}}}={cfg.intermediate_size}$" if isinstance(cfg.intermediate_size,int) else r"$d_{\mathrm{ff}}$",
               size=font,subsize=small)
        d.route([(c,Y(66)),(c,Y(74)-1.05)])
    d.route([(c,Y(38.6)),(left,Y(38.6)),(left,Y(74)),(c-1.05,Y(74))])
    d.add(c,Y(74))
    d.route([(c,Y(74)+1.05),(c,Y(76.8))])
    d.text(c,Y(79),r"$F_{\ell}(h)$",font)


def fig_forward_stack(outdir=OUT, *, config=None, formats=("png","svg","pdf"), dpi=600):
    cfg=config or Architecture()
    d=Drawing(180,225,"chronos_forward_stack")
    _header(d,cfg,"Source-aligned configurable architecture")
    d.frame(4,19,74,24,fill="panel")
    d.text(7,22.7,"a   INPUT EMBEDDING",7,weight="bold",ha="left")
    d.text(75,22.7,rf"$d={cfg.hidden_size}$" if isinstance(cfg.hidden_size,int) else "Width d",6,ha="right",color="muted")
    d.node(7,28,26,10,"Token IDs",sub=r"$B\times T$",size=6.5,subsize=6)
    d.node(39,28,35,10,"Input embedding",sub=r"$W_{\mathrm{in}}$",size=6.5,subsize=6)
    d.route([(33,33),(39,33)])
    d.frame(84,19,92,24,fill="panel")
    d.text(87,22.7,"c   OUTPUT EMBEDDING",7,weight="bold",ha="left")
    d.node(87,28,24,10,"Final norm",kind="norm",sub="RMSNorm",size=6.5,subsize=6)
    d.node(116,28,28,10,"Output embedding",kind="attn",sub="LM head · linear",size=6.1,subsize=6)
    d.node(149,28,24,10,"Logits",sub=r"$B\times T\times V$",size=6.5,subsize=6)
    d.route([(111,33),(116,33)])
    d.route([(144,33),(149,33)])
    d.route([(56.5,38),(56.5,47),(2,47),(2,67),(7,67)])
    d.text(34,45.2,r"$h_0\in\mathbb{R}^{B\times T\times d}$",6,color="muted")
    d.route([(173,67),(178,67),(178,47),(99,47),(99,38)])
    d.text(135,45.2,"Final hidden state",6,color="muted")
    d.text(4,54.2,"b   FORWARD STACK WITH EMBEDDED BLOCK INTERIORS",7,weight="bold",ha="left")
    d.frame(4,59,172,124,fill="panel")
    d.node(7,63,26,8,f"Prelude ×{cfg.prelude_layers}",size=6.5,weight="bold")
    d.node(49,63,82,8,f"{cfg.recurrent_groups} recurrent physical blocks ×{cfg.recurrent_depth}",
           kind="recur_bg",size=7,weight="bold")
    d.node(147,63,26,8,f"Coda ×{cfg.coda_layers}",size=6.5,weight="bold")
    d.route([(33,67),(49,67)]); d.route([(131,67),(147,67)])
    d.dot(136,67,color="recur")
    d.route([(136,67),(136,76),(44,76),(44,67),(49,67)],color="recur",dashed=True,lw=.85)
    d.dot(38,67)
    d.route([(38,67),(38,83),(33,83)],color="attn_edge",dashed=True)
    d.node(7,79.5,26,7,"Lookahead",kind="attn",size=6.3)
    d.text(45,83,"Post-prelude observation (R > 1) → future expert prefetch",6.2,color="attn_edge",ha="left")
    d.text(8,92,"SUPPORTED BLOCK VARIANTS",6.4,weight="bold",ha="left")
    d.text(172,92,"Variants are alternatives, not a fixed serial schedule.",6,color="muted",ha="right")
    for i,b in enumerate(BLOCKS):
        _block_interior(d,5+43*i,97,39.6,b,cfg,height=82)
    d.frame(4,188,172,27,fill="recur_bg")
    d.text(7,191.5,"RECURRENCE / CACHE",6.1,weight="bold",color="recur",ha="left")
    d.text(7,197,r"$h^*=h+e_r+e_{\ell};\quad h' = h+\Delta t_{\ell}[F_{\ell}(h^*)-h]$",7.2,ha="left")
    d.text(7,203,"block order: repeat the physical stack; layer order: repeat each layer.",6,ha="left",color="muted")
    d.text(7,208,"shared_first option: first iteration writes KV; later iterations use fresh Q.",6,ha="left",color="muted")
    d.text(7,212.2,"Adaptive halting optional; KV modes also include per_iteration / first_only.",6,ha="left",color="muted")
    d.route([(113,191),(113,212)],arrow=False,color="rule",lw=.6)
    d.text(116,191.5,"CONFIGURATION",6.1,weight="bold",ha="left")
    d.text(116,197,"Even layers: MLA; odd layers: SWA.",6,ha="left")
    d.text(116,202,"use_moe selects Dense / MoE globally.",6,ha="left",color="muted")
    d.text(116,207,"Weight tying is optional (same matrix).",6,ha="left",color="muted")
    d.text(116,212,"Hybrid off: global GQA in all layers.",6,ha="left",color="muted")
    _legend(d,220)
    return d.save(Path(outdir),formats,dpi)


def _legend(d,y):
    for x,kind,name in ((4,"norm","Norm"),(23,"attn","Global attention"),(59,"local","Local attention"),
                         (93,"ffn","Dense FFN"),(121,"moe","Routed"),(142,"shared","Shared")):
        d.frame(x,y-1.4,3,2.8,fill=kind,edge=kind+"_edge" if kind+"_edge" in COLORS else "rule",radius=.35,lw=.5)
        d.text(x+4.2,y,name,6,ha="left",color="muted")
    d.route([(163,y),(169,y)],color="recur",dashed=True,lw=.7)
    d.text(170.5,y,"Loop",6,ha="left",color="muted")


def fig_block_interior(outdir=OUT, *, config=None, formats=("png","svg","pdf"), dpi=600):
    cfg=config or Architecture(); d=Drawing(180,126,"chronos_block_interior")
    _header(d,cfg,"Four supported block configurations")
    for i,b in enumerate(BLOCKS):
        _block_interior(d,4+44*i,20,39.6,b,cfg,height=81)
    d.route([(4,106),(176,106)],arrow=False,color="rule",lw=.6)
    d.text(4,110,"Full block path: attention + MLP, two residuals. In first_only, later iterations bypass attention.",6.2,ha="left")
    d.text(4,115,"Solid: hidden states. Orange dashes: gates. MoE output = routed sum + shared mean (if present). SiLU is the default activation.",6,ha="left",color="muted")
    d.text(4,120,"Hybrid parity selects MLA / SWA; use_moe chooses Dense / MoE globally. These cards are configuration alternatives.",6,ha="left",color="muted")
    return d.save(Path(outdir),formats,dpi)


def fig_ssd_kernel(outdir=OUT, *, config=None, formats=("png","svg","pdf"), dpi=600):
    """Code-only runtime figure: alternative layouts, selective paging, policies."""
    cfg=config or Architecture(); d=Drawing(180,188,"chronos_ssd_kernel")
    _header(d,cfg,"Cluster-aware lazy expert runtime")
    d.text(4,20,"a   EXPERT PAGE LAYOUT · alternative routes",7,weight="bold",ha="left")
    d.frame(4,25,172,35,fill="panel")
    d.node(8,29,40,10,"Training partition",kind="shared",sub="Fixed expert-ID groups",size=6.5,subsize=6)
    d.node(8,45,40,10,"Calibration traces",kind="attn",sub="Expert coactivation",size=6.5,subsize=6)
    d.node(58,45,44,10,"Louvain / greedy",kind="attn",sub="Optional offline repacking",size=6.5,subsize=6)
    d.node(120,35,50,13,"Clustered SSD layout",kind="io",sub=".ctsr files + manifest",size=7,subsize=6)
    d.route([(48,34),(110,34),(110,39),(120,39)])
    d.route([(48,50),(58,50)])
    d.route([(102,50),(110,50),(110,44),(120,44)])
    d.text(78,29,"Locality-trained auto paging",6,color="shared_edge")
    d.text(4,67,"b   ASYNC PREDICTION AND WEIGHT DATA PATH · Torch / CUDA",7,weight="bold",ha="left")
    d.frame(4,72,172,53,fill="panel")
    d.node(8,77,31,11,"Lookahead",kind="attn",sub="Post-prelude (R > 1)",size=6.5,subsize=6)
    d.node(49,77,33,11,"Bounded queue",kind="attn",sub="Future expert IDs",size=6.5,subsize=6)
    d.node(93,77,36,11,"Manifest / index",kind="io",sub="Expert → cluster ID",size=6.5,subsize=6)
    d.node(140,77,32,11,"Page selection",kind="io",sub="Capacity-limited",size=6.5,subsize=6)
    for a,b in ((39,49),(82,93),(129,140)):
        d.route([(a,82.5),(b,82.5)],color="attn_edge",dashed=True)
    d.node(8,105,32,13,"SSD pages",kind="io",sub="Selective mmap reads",size=6.5,subsize=6)
    d.node(51,105,36,13,"RAM cache",kind="norm",sub="Pinned host weights",size=6.5,subsize=6)
    d.node(100,105,35,13,"Compute memory",kind="local",sub="Materialized experts",size=6.5,subsize=6)
    d.node(146,105,26,13,"Expert compute",kind="moe",sub="Selected top-k",size=6,subsize=6)
    d.route([(40,111.5),(51,111.5)])
    d.route([(87,111.5),(100,111.5)])
    d.route([(135,111.5),(146,111.5)])
    d.text(93.5,101.5,"H2D",6,color="muted")
    d.route([(156,88),(156,97),(24,97),(24,105)],color="attn_edge",dashed=True)
    d.text(90,94,"Read requested pages + budgeted cluster neighbours",6,color="muted")
    d.text(4,132,"c   COLD-MISS POLICY · conditional, not an unconditional replacement",7,weight="bold",ha="left")
    d.frame(4,137,172,31,fill="panel")
    d.node(8,146,30,12,"Selected expert",kind="moe",sub="Cold SSD miss",size=6.5,subsize=6)
    d.node(55,140,53,10,"on_demand",kind="shared",sub="Queue exact page; shared fallback now",size=6.5,subsize=6)
    d.node(55,155,53,10,"sync_on_demand",kind="norm",sub="Load / wait / execute exact expert",size=6.5,subsize=6)
    d.node(127,145,44,15,"Residency policy",kind="io",sub="LRU / protected window\nnext-use option (MLX)",size=6.5,subsize=6)
    d.route([(38,152),(46,152),(46,145),(55,145)],color="router_edge",dashed=True)
    d.route([(46,152),(46,160),(55,160)],color="muted",dashed=True)
    d.route([(171,152.5),(174,152.5),(174,123),(117.5,123),(117.5,118)],color="attn_edge",dashed=True)
    d.text(4,174,"Solid black: expert weight movement. Blue dashes: requests / control. SSD misses are not claimed to be free.",6,ha="left",color="muted")
    d.text(4,179,"MLX: whole-cluster load / evaluation can precede bounded admission; unified memory, no CUDA H2D.",6,ha="left",color="muted")
    d.text(4,184,"Locality and packed layout can reduce page demand; improvement requires measurement on the target workload.",6,ha="left",color="muted")
    return d.save(Path(outdir),formats,dpi)


def fig_expert_communities(outdir=OUT, *, config=None, formats=("png","svg","pdf"), dpi=600):
    """Mechanistic schematic; the node positions are not a measured projection."""
    cfg=config or Architecture(); d=Drawing(180,153,"chronos_expert_communities")
    _header(d,cfg,"Routing locality and expert communities")
    d.frame(4,20,108,102,fill="panel")
    d.text(8,24,"a   SPARSE ROUTING WITH SHARED RESIDUAL",7,weight="bold",ha="left")
    d.node(40,108,39,9,"Hidden states",sub=None,size=6.5)
    d.node(43,87,33,10,"Router",kind="router",sub="FP32 softmax → top-k",size=6.5,subsize=6)
    d.route([(59.5,108),(59.5,97)])
    # Illustrative, pre-defined cluster enclosures, never a t-SNE data plot.
    for x,kind,name in ((10,"attn","Cluster A"),(45,"attn","Cluster B"),(80,"attn","Cluster C")):
        d.frame(x,47,27,28,fill="paper",edge="attn_edge",dashed=True)
        d.text(x+13.5,51,name,6.3,weight="bold")
        for dx,dy in ((5,57),(15,57),(5,65),(15,65)):
            d.node(x+dx,dy,7,5,"",kind="expert" if x==45 and dy==57 else "attn",size=6)
    d.node(40,29,39,9,"Weighted output",kind="ffn",size=6.5)
    d.add(59.5,41,sigma=True)
    d.route([(59.5,39.95),(59.5,38)])
    d.route([(59.5,87),(59.5,80)],color="router_edge",dashed=True,arrow=False)
    d.route([(59.5,80),(42,80),(42,43),(46,43)],color="router_edge",dashed=True)
    d.route([(59.5,80),(76,80),(76,43),(71,43)],color="router_edge",dashed=True)
    d.route([(59.5,103),(38.5,103),(38.5,59.5),(50,59.5)])
    d.dot(38.5,77)
    d.route([(38.5,77),(74,77),(74,59.5),(67,59.5)])
    d.dot(59.5,103)
    # Two selected illustrative experts; circles multiply their outputs by gates.
    for xx in (47,70):
        d.ax.add_patch(Circle((xx,43),1,fc="white",ec=COLORS["ink"],lw=.65,zorder=6))
        d.text(xx,43,"×",6)
    d.route([(53.5,57),(53.5,54.5),(47,54.5),(47,44)])
    d.route([(63.5,57),(63.5,54.5),(70,54.5),(70,44)])
    d.route([(47,42),(47,41),(58.45,41)])
    d.route([(70,42),(70,41)],arrow=False)
    d.dot(70,41)
    d.node(84,85,23,13,"Shared",kind="shared",sub="Always on · mean",size=6.5,subsize=6)
    d.route([(59.5,103),(95.5,103),(95.5,98)])
    d.route([(95.5,85),(109,85),(109,41),(60.55,41)])
    d.text(10,43,"Fixed expert groups",6,color="muted",ha="left")
    d.frame(118,20,58,102,fill="recur_bg")
    d.text(122,24,"b   TRAINING OBJECTIVE",7,weight="bold",ha="left")
    d.node(123,33,48,12,"Stream / document",kind="io",sub="Aggregate all dispatch probabilities",size=6.5,subsize=6)
    d.node(123,53,48,17,"Routing-locality objective",kind="shared",sub="Concentrate within a stream\nspread mass across streams",size=6.5,subsize=6)
    d.route([(147,45),(147,53)])
    d.text(147,80,r"$\mathcal{L}_{\mathrm{cl}}=1-\frac{I(w;c)}{\log N_c}$",8)
    d.text(147,93,"Fixed ID partition; learned routing.\nNot physical migration of experts.",6.2,color="muted")
    d.text(147,110,"Enabled only when its coefficient > 0.",6,color="muted")
    d.text(4,129,"Communities here are a conceptual view of routing specialization; positions and colours are not measured embeddings.",6.1,ha="left")
    d.text(4,135,"Offline alternative: activation traces → coactivation graph → Louvain / greedy partition → cluster-page packing.",6.1,ha="left",color="muted")
    d.text(4,141,"Training partition and offline graph partition are distinct mechanisms; neither implies spontaneous expert motion.",6.1,ha="left",color="muted")
    d.text(4,147,"The locality objective encourages reuse of expert pages across tokens and recurrent dispatches; no speedup is asserted.",6.1,ha="left",color="muted")
    return d.save(Path(outdir),formats,dpi)


def _main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir",type=Path,default=OUT)
    parser.add_argument("--config",type=Path,help="Optional symbolic config or real Chronos config.json")
    parser.add_argument("--formats",nargs="+",choices=("png","svg","pdf"),default=("png","svg","pdf"))
    parser.add_argument("--layout-report",type=Path,help="Optional path for the layout audit JSON")
    parser.add_argument("--figures",choices=("all","forward","blocks","kernel","communities"),default="all")
    parser.add_argument("--dpi",type=int,default=600)
    args=parser.parse_args()
    cfg=Architecture.from_mapping(json.loads(args.config.read_text())) if args.config else Architecture()
    if not 72<=args.dpi<=2400: parser.error("dpi must be between 72 and 2400")
    args.outdir.mkdir(parents=True,exist_ok=True)
    reports=[]
    functions=(("forward",fig_forward_stack),("blocks",fig_block_interior),("kernel",fig_ssd_kernel),("communities",fig_expert_communities))
    for name,fn in functions:
        if args.figures in ("all",name): reports.append(fn(args.outdir,config=cfg,formats=args.formats,dpi=args.dpi))
    report={"architecture":asdict(cfg),"source":"Project_Chronos-0.3.3a1","mode":"configuration parameters" if args.config else "symbolic configurable architecture",
            "supported_block_variants":[asdict(b) for b in BLOCKS],"dpi":args.dpi,"formats":list(args.formats),"figures":reports,
            "construction_contracts":["All block variants include attention and two residuals.",
             "use_moe is global; cards are supported alternatives, not a fixed ABCE sequence.",
             "Input token embedding and output linear projection are separate modules; weight tying is optional.",
             "Kernel artwork is entirely code-generated.","No measured high-dimensional embedding geometry or performance measurements are depicted."]}
    if args.layout_report:
        args.layout_report.parent.mkdir(parents=True,exist_ok=True)
        args.layout_report.write_text(json.dumps(report,indent=2)+"\n")
    print("Layout audit passed: text fits, text does not overlap, and wires avoid node interiors.")


if __name__=="__main__":
    _main()
