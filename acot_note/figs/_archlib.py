"""Shared drawing helpers of the architecture figures (fig_architecture_v3 / fig_arch_overview / fig_arch_ablations /
fig_arch_baselines). Palette as in fig_architecture_v1/v2 (validate_palette passed): slot1 blue = TAHRelM, slot2 orange =
PGIF, slot3 aqua = attribute opening (and the NATURAL opening setting); violet = the FORCED shared opening of the
mechanism setting (1006 §3.4); frozen = neutral grey; changed / method-specific blocks = ink dashed border."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

_f = os.path.expanduser("~/.local/share/fonts/SimHei.ttf")
if os.path.exists(_f):
    fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 8.5})

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TAH, TAH_BG = "#2a78d6", "#dbe8f8"
PG, PG_BG = "#eb6834", "#fde3d7"
OP, OP_BG = "#1baf7a", "#d9f2e8"
SH, SH_BG = "#7a5bc7", "#ece6f8"     # forced shared opening (mechanism setting)
FROZ, FROZ_E, INP = "#e6e5e1", "#b9b8b2", "#ffffff"
CHG = (0, (4, 2))


def _t(s):
    """SimHei has no U+2212 (minus sign) glyph → ASCII hyphen."""
    return s.replace("\u2212", "-") if isinstance(s, str) else s


class Canvas:
    """a 100 × 100 data-coordinate canvas on one figure."""

    def __init__(self, w, h):
        self.fig = plt.figure(figsize=(w, h), facecolor=SURF)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.axis("off")

    def box(self, x, y, w, h, title="", sub="", fc=FROZ, ec=FROZ_E, lw=1.0, fs=9.2, bold=False, ls="-", sfs=7.8,
            align="center"):
        a, title, sub = self.ax, _t(title), _t(sub)
        a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=lw,
                                   linestyle=ls, zorder=2))
        tx, ha = (x + w / 2, "center") if align == "center" else (x + 0.9, "left")
        if title and sub:
            a.text(tx, y + h - 0.9, title, ha=ha, va="top", fontsize=fs, color=INK,
                   fontweight="bold" if bold else "normal", zorder=3)
            a.text(tx, y + h - 1.25 - fs * 0.2, sub, ha=ha, va="top", fontsize=sfs, color=INK2, zorder=3,
                   linespacing=1.4)
        elif sub:
            a.text(tx, y + h / 2, sub, ha=ha, va="center", fontsize=sfs, color=INK2, zorder=3, linespacing=1.5)
        elif title:
            a.text(tx, y + h / 2, title, ha=ha, va="center", fontsize=fs, color=INK,
                   fontweight="bold" if bold else "normal", zorder=3)

    def group(self, x, y, w, h, title, fc, ec, sub=None, lw=2.2, ls="-"):
        a = self.ax
        a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0", fc=fc, ec=ec, lw=lw,
                                   linestyle=ls, zorder=1.5))
        a.text(x + w / 2, y + h - 0.3, title, fontsize=10, fontweight="bold", color=INK, ha="center", va="top")
        if sub:
            a.text(x + w / 2, y + h - 2.1, sub, fontsize=8, color=INK2, ha="center", va="top")

    def badge(self, x, y, text, fc=INK):
        self.ax.text(x, y, text, fontsize=7.8, color=SURF, fontweight="bold", ha="right", va="center", zorder=5,
                     bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec="none"))

    def tag(self, x, y, text, fc=OP, fs=7.8):
        """left-aligned filled label (setting tags: natural opening = OP, forced shared opening = SH)."""
        self.ax.text(x, y, _t(text), fontsize=fs, color="#ffffff", fontweight="bold", ha="left", va="center", zorder=5,
                     bbox=dict(boxstyle="round,pad=0.28", fc=fc, ec="none"))

    def arrow(self, x1, y1, x2, y2, col=INK2, lw=1.2, ls="-", rad=0.0, label=None, lx=None, ly=None, fs=7.8):
        self.ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=col, lw=lw,
                                          linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=1))
        if label:
            self.ax.text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2, _t(label),
                         fontsize=fs, color=INK2, ha="center", va="center", zorder=4,
                         bbox=dict(fc=SURF, ec="none", pad=0.4))

    def line(self, xs, ys, col=INK2, lw=1.2, ls="-"):
        self.ax.plot(xs, ys, color=col, lw=lw, ls=ls, zorder=1)

    def text(self, x, y, s, fs=8.2, col=INK2, bold=False, ha="left", va="top", ls=1.45, **kw):
        self.ax.text(x, y, _t(s), fontsize=fs, color=col, fontweight="bold" if bold else "normal", ha=ha, va=va,
                     linespacing=ls, **kw)

    def table(self, x0, y0, cols, rows, fs=8.0, dy=1.9, hl=None, hl_fc=OP_BG, x1=99.2, header_fs=None):
        """cols = [(x, header)]; rows = [[cell, ...]]; hl = set of highlighted row indices."""
        for x, t in cols:
            self.ax.text(x, y0, t, fontsize=header_fs or fs, color=INK2, fontweight="bold", va="center")
        self.ax.plot([x0, x1], [y0 - dy / 2, y0 - dy / 2], color=GRID, lw=0.8)
        for i, row in enumerate(rows):
            y = y0 - dy * (i + 1)
            if hl and i in hl:
                self.ax.add_patch(FancyBboxPatch((x0, y - dy * 0.45), x1 - x0, dy * 0.9,
                                                 boxstyle="round,pad=0.05,rounding_size=0.3", fc=hl_fc, ec="none",
                                                 zorder=0.5))
            for (x, _), t in zip(cols, row):
                self.ax.text(x, y, _t(t), fontsize=fs, color=INK, va="center", zorder=2)

    def save(self, name):
        for ext in ("png", "pdf", "svg"):
            self.fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=150 if ext == "png" else None, facecolor=SURF)
        print(f"saved {name}.{{png,pdf,svg}}")
