"""§9 e/f figures from runs/abl_allactor/runs_post (eval_post.py) + noise_decomp_L2.json.
  fig_e_abstain_rules.png : SU re-entry vs SU absent-pick trade-off per decision-rule family (small multiples, shared axes)
  fig_f_world_noise.png   : SU visible-frame acc / SU re-entry vs world-coordinate noise level + L2 error-source split
Palette: dataviz reference categorical slots 1-3 (validated all-pairs light: CVD ΔE 9.2, normal 24.0) — entity colours fixed
across both figures (TAH-X blue, rnw orange, third series aqua); TAHRelM = muted dashed reference. Aqua < 3:1 contrast →
every series is direct-labelled. Static PNGs for the note → values are printed as labels.
  python3 acot_note/figs/fig_post_ef.py"""
import contextlib, io, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = "/nvidia/hque/code/cyhe/uav-acot-track"
sys.path.insert(0, os.path.join(REPO, "runs/abl_allactor"))
with contextlib.redirect_stdout(io.StringIO()):
    import summarize_post as S                                            # pooled test metrics (same as summary_post.txt)

font_manager.fontManager.addfont("/home/hque/.local/share/fonts/SimHei.ttf")
plt.rcParams["font.family"] = "SimHei"; plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
HERE = os.path.dirname(os.path.abspath(__file__))


def P(s, key, m):
    return S.pooled(S.per_episode(S.BY[s], S.fixed(key), m))


def style(ax, xlabel, ylabel):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10.5); ax.set_ylabel(ylabel, color=INK2, fontsize=10.5)


# ------------------------------------------------------------------ e: trade-off small multiples
E_SER = [("x_target", BLUE, "TAH-X"), ("rnw_target", ORANGE, "rnw"), ("x_all", AQUA, "TAH-X 全车")]
FAM = [("按缺席时长下调:null - β·log1p(Δt)", [("argmax", "0"), ("gap_0.5", "0.5"), ("gap_1.0", "1"), ("gap_2.0", "2"), ("gap_4.0", "4")], "β"),
       ("常数下调:null - b", [("argmax", "0"), ("bias_0.5", "0.5"), ("bias_1.0", "1"), ("bias_2.0", "2"), ("bias_3.0", "3"), ("bias_5.0", "5")], "b"),
       ("重捕获模式:Δt ≥ T 且候选置信 ≥ q 时强制选", [("argmax", "原"), ("reacq_3.0_0.95", "3s/.95"), ("reacq_3.0_0.8", "3s/.8"),
                                         ("reacq_3.0_0.5", "3s/.5"), ("reacq_1.0_0.95", "1s/.95"), ("reacq_1.0_0.8", "1s/.8"), ("reacq_1.0_0.5", "1s/.5")], "T/q")]
fig, axs = plt.subplots(1, 3, figsize=(18, 6.4), dpi=150, sharex=True, sharey=True)
fig.patch.set_facecolor(SURF)
for ax, (title, rules, pname) in zip(axs, FAM):
    style(ax, "缺席期乱选率(自更新;越低越好)", "重新入画首帧认对 ≥ 1 s(自更新;越高越好)" if ax is axs[0] else "")
    ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=10)
    for s, c, lab in E_SER:
        xs = [P(s, f"su|{r}|L0", "absent_pick_rate") for r, _ in rules]
        ys = [P(s, f"su|{r}|L0", "reentry@1s") for r, _ in rules]
        ax.plot(xs, ys, color=c, lw=1.6, zorder=2, alpha=0.9)
        ax.plot(xs[1:], ys[1:], "o", ms=6, color=c, mec=SURF, mew=1.2, zorder=3)
        ax.plot(xs[0], ys[0], "o", ms=9, mfc=SURF, mec=c, mew=2.0, zorder=4)              # hollow = argmax (as trained)
        for j, ((r, t), x, y) in enumerate(zip(rules, xs, ys)):
            if r != "argmax" and (pname != "T/q" or s == "x_target"):
                ax.annotate(t, (x, y), xytext=(4, 4 if j % 2 else -11), textcoords="offset points", fontsize=8.2,
                            color=MUTED)
        ax.annotate(lab, (xs[0], ys[0]), xytext=(-8, 8 if s != "rnw_target" else -14), textcoords="offset points",
                    fontsize=9.8, color=INK, ha="right", weight="bold")
    ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.62)
axs[0].annotate("β = 0.5:重入 +0.186*、锁死 -0.207*、乱选 +0.053*\n(测试集事后观察;验证集预设准则未选中)", xy=(P("x_target", "su|gap_0.5|L0", "absent_pick_rate"), P("x_target", "su|gap_0.5|L0", "reentry@1s")),
                xytext=(0.30, 0.57), fontsize=9, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
h = [plt.Line2D([], [], color=c, lw=1.6, marker="o", ms=6, mec=SURF, label=l) for _, c, l in E_SER]
h.append(plt.Line2D([], [], ls="", marker="o", ms=9, mfc=SURF, mec=INK2, mew=2, label="空心 = 原规则 argmax"))
fig.legend(handles=h, loc="upper right", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.99, 1.0), labelcolor=INK2)
fig.suptitle("e · 弃权决策规则扫描(真值世界坐标,64 集 × 2 种子;点旁数字为规则参数)", x=0.01, ha="left", fontsize=14,
             color=INK, weight="bold", y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(HERE, "fig_e_abstain_rules.png"), facecolor=SURF, bbox_inches="tight")

# ------------------------------------------------------------------ f: noise curves + decomposition
F_SER = [("x_target", BLUE, "TAH-X"), ("rnw_target", ORANGE, "rnw(世界系)"), ("rn_target", AQUA, "rn(图像系)")]
XT = ["L0\n真值", "L1\n0.8 m", "L2\n1.8 m", "L3\n3.5 m", "L4\n6.9 m"]
fig = plt.figure(figsize=(18, 6.4), dpi=150); fig.patch.set_facecolor(SURF)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.28)
for k, (m, yl, ttl) in enumerate([("acc", "自更新·可见帧准确率", "部署可见帧准确率"),
                                  ("reentry@1s", "自更新·重新入画首帧认对 ≥ 1 s", "部署重新入画")]):
    ax = fig.add_subplot(gs[0, k]); style(ax, "噪声档(世界坐标误差中位数)", yl)
    ax.set_title(ttl, fontsize=11.5, color=INK, loc="left", pad=10)
    ref = [P("relm_target", f"su|argmax|L{L}", m) for L in range(5)]
    ax.plot(range(5), ref, color=MUTED, lw=1.4, ls="--", zorder=1)
    ends = [(ref[-1], f"TAHRelM {ref[-1]:.2f}", MUTED)]
    for s, c, lab in F_SER:
        ys = [P(s, f"su|argmax|L{L}", m) for L in range(5)]
        ax.plot(range(5), ys, color=c, lw=1.8, marker="o", ms=6, mec=SURF, mew=1.2, zorder=3)
        ends.append((ys[-1], f"{lab} {ys[-1]:.2f}", INK))
        ax.annotate(f"{ys[0]:.2f}", (0, ys[0]), xytext=(-6, 0), textcoords="offset points", fontsize=8.8,
                    color=INK2, ha="right", va="center")
    top = 1.0 if m == "acc" else 0.5; gapy = 0.045 * top               # end-label repel (min vertical gap)
    ends.sort(key=lambda e: e[0]); placed = []
    for y, t, c in ends:
        y = max(y, placed[-1] + gapy) if placed else max(y, 0.0)
        placed.append(y); ax.text(4.15, y, t, fontsize=9.4, color=c, va="center")
    ax.set_xticks(range(5)); ax.set_xticklabels(XT); ax.set_xlim(-0.6, 5.3)
    ax.set_ylim(0, 1.0 if m == "acc" else 0.5)
D = json.load(open(os.path.join(REPO, "runs/abl_allactor/noise_decomp_L2.json")))["rnw"]
order = [("L0 GT", "无误差(真值)"), ("L2 px only", "仅框中心误差"), ("L2 pose drift only", "仅位姿漂移"),
         ("L2 depth only", "仅深度误差(逐帧 5%)"), ("L2 full", "L2 全部误差")]
ax = fig.add_subplot(gs[0, 2]); style(ax, "自更新·可见帧准确率", "")
ax.set_title("rnw · L2 误差来源分解", fontsize=11.5, color=INK, loc="left", pad=10)
vals = [D[k]["su_acc"] for k, _ in order]
ax.barh(range(len(order))[::-1], vals, color=ORANGE, height=0.62, edgecolor=SURF, linewidth=2)
for i, v in zip(range(len(order))[::-1], vals):
    ax.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=9.6, color=INK)
ax.set_yticks(range(len(order))[::-1]); ax.set_yticklabels([t for _, t in order], fontsize=10, color=INK2)
ax.set_xlim(0, 1.05); ax.grid(axis="y", visible=False)
fig.suptitle("f · 世界坐标估计误差下的部署性能(框中心 + 深度 + 位姿反投影;同一噪声样本所有模型共用)", x=0.01, ha="left",
             fontsize=14, color=INK, weight="bold", y=1.02)
fig.savefig(os.path.join(HERE, "fig_f_world_noise.png"), facecolor=SURF, bbox_inches="tight")
print("saved fig_e_abstain_rules.png, fig_f_world_noise.png")
