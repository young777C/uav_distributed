"""TAHRelM vs TAH-X v0: implementation differences row by row, grouped by the §9 component-ablation step that adds
them (acot_note/1003_event-centric-dataset-review.md §9.3), with each group's measured contribution (paired Δ, 95% CI,
episode bootstrap over 64 test episodes × 2 seeds; runs/abl_allactor/summary3.txt) as an interval plot.
Palette: dataviz reference categorical slots 1-2 (validated light, all-pairs) for the two metrics; neutrals from the
reference instance (gridline #e1e0d9, baseline #c3c2b7). Static PNG for the note → values are printed as labels.
  python3 acot_note/figs/fig_tahrelm_vs_tahx.py  → acot_note/figs/fig_tahrelm_vs_tahx.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

font_manager.fontManager.addfont("/home/hque/.local/share/fonts/SimHei.ttf")
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE, BAND = "#e1e0d9", "#c3c2b7", "#f3f2ee"
BLUE, ORANGE = "#2a78d6", "#eb6834"                        # metric 1 · metric 2

# (group, [(dimension, TAHRelM, TAH-X), ...], effects {metric: (Δ, lo, hi)}, side note)
GROUPS = [
    (("A · 计划采样", "relm → relmS"),
     [("训练时写记忆", "始终用真值目标写入(teacher forcing)",
       "真值写入;以概率 p_ss 改写自身选择\n(计划采样:p_ss 在前 60% 步数内 0 → 0.5)")],
     {"acc": (-0.003, -0.030, 0.020), "re": (0.000, 0.000, 0.000)},
     "缺席乱认 ±0 · 真值重入 +0.029*"),
    (("B · null(目标不在)输出", "relmS → rn"),
     [("输出", "n 个候选的 logits,argmax 必须选一个", "n+1 个 logits,最后一维 = null(目标不在)"),
      ("训练损失", "只在目标可见帧上算 CE", "所有帧算 CE(缺席帧标签 = null)\n+ 0.5·BCE(写门:选中的是否为目标)"),
      ("部署写入", "每帧写入所选候选\n(目标缺席时也写 → 记忆被干扰车污染)", "选 null 不写;SUG 仅在写门 > 0.5 时写")],
     {"acc": (-0.071, -0.116, -0.028), "re": (0.107, 0.037, 0.250)},
     "缺席乱认 -0.284* · 可见时误判不在 +0.119"),
    (("C · 世界系运动 + Δt", "rn → rnw"),
     [("运动坐标系", "图像平面 (u/W, v/H):随无人机运动漂移", "世界坐标(米):与相机运动无关"),
      ("时间", "无时间概念:pred = pos + vel(单步外推)",
       "显式 Δt = t_now - t_last:pred = pos + vel·Δt\n(vel 单位 m/s,EMA 0.5,限幅 ±30)"),
      ("匹配范围", "软运动门 msoft = exp(-‖relp‖ / r)\nr ≈ 0.10·W,固定不随时间变化",
       "d = (wpos - pred) / σ,σ = 2 + 3·Δt 米\n(缺席越久越宽);另输入 log1p(Δt)")],
     {"acc": (0.334, 0.257, 0.414), "re": (0.186, 0.068, 0.388)},
     "缺席乱认 -0.380* · 真值重入 +0.086*"),
    (("D · 高容量部件", "rnw → TAH-X"),
     [("外观度量", "冻结 DINOv2 余弦(不可学)", "可学投影 384→256→128 + L2;另保留原始 DINO 余弦"),
      ("外观记忆", "1 个 EMA 模板(g = σ(gate) ≈ 0.30)",
       "K = 8 个多视角槽(间隔 ≥ 1 s 建槽,cos > 0.98 替换,\n先进先出)+ 原始 DINO 模板(EMA 0.7 / 0.3)"),
      ("候选特征", "7 维:cos, relp_x, relp_y, depth,\nrank, dmot, msoft",
       "11 维:smax, smean, craw, rank, d_x, d_y, ‖d‖,\nlog1p(Δt), u, v, depth"),
      ("打分方式", "每个候选独立过 MLP 7→64→64→1",
       "token 化 → 2 层注意力块(交叉注意力读记忆槽,\n自注意力让候选互相比较)→ 线性头"),
      ("写入置信度", "conf = σ(top1 - top2)(分差)", "写门头 wgate:σ(wgate) > 0.5 才算可写"),
      ("记忆更新", "3 个可学标量:gate、vbeta、mgate", "无参规则(速度 EMA 0.5、模板 EMA 0.7 / 0.3、槽管理)"),
      ("参数量", "4,740", "564,610(注意力块 397,568 · 投影 131,456)")],
     {"acc": (0.002, -0.048, 0.049), "re": (0.071, -0.167, 0.311)},
     "缺席乱认 -0.212* · 真值重入 -0.071*"),
]
METRICS = [("acc", BLUE, "自更新·可见帧准确率"), ("re", ORANGE, "自更新·重新入画 ≥ 1 s")]

# ---- layout (units: 10 per inch) ----
LINE, PAD, GMIN = 2.35, 1.7, 11.5
X_G, X_D, X_R, X_X = 1.5, 27.5, 41.5, 87.0             # group · dimension · TAHRelM · TAH-X text columns
F0, F1, X_V = 140.0, 165.0, 166.5                      # forest axis span, value-label column
VMIN, VMAX = -0.2, 0.45
fx = lambda v: F0 + (v - VMIN) / (VMAX - VMIN) * (F1 - F0)

layout, y = [], 0.0
for g, rows, eff, note in GROUPS:
    hs = [LINE * max(r[1].count("\n"), r[2].count("\n")) + LINE + PAD for r in rows]
    if sum(hs) < GMIN:
        hs[-1] += GMIN - sum(hs)
    layout.append((y, hs)); y += sum(hs)
BODY = y
TOP_H, BOT_H = 17.0, 11.0
H = TOP_H + BODY + BOT_H
fig, ax = plt.subplots(figsize=(18, H / 10), dpi=150)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.set_xlim(0, 180); ax.set_ylim(0, H); ax.axis("off")
Y = lambda d: H - TOP_H - d                             # depth below header → data y

ax.text(1.5, H - 1.2, "TAHRelM 与 TAH-X v0 的实现差异 · 按 §9 分量消融步骤分组,右侧为该组差异的实测贡献",
        fontsize=15, color=INK, weight="bold", va="top")
hy = H - TOP_H + 2.0
for x, s in [(X_G, "消融步(§9)"), (X_D, "维度"), (X_R, "TAHRelM(train/tah.py · 4,740 参数)"),
             (X_X, "TAH-X v0(abl_allactor/tahx.py · 564,610 参数)")]:
    ax.text(x, hy, s, fontsize=11.5, color=INK, weight="bold", va="bottom")
ax.text(F0, hy + 7.6, "实测贡献:自更新配对差(95% CI)", fontsize=11.5, color=INK, weight="bold", va="bottom")
for i, (k, c, lab) in enumerate(METRICS):                # legend (2 series → legend + direct value labels)
    lx, ly = F0, hy + 4.6 - i * 2.6                    # stacked legend rows
    ax.plot([lx, lx + 2.4], [ly, ly], color=c, lw=1.6, solid_capstyle="round")
    ax.plot(lx + 1.2, ly, "o", ms=6, color=c, mec=SURF, mew=1.2)
    ax.text(lx + 3.4, ly, lab, fontsize=9.6, color=INK2, va="center")
ax.plot([0.5, 179.5], [H - TOP_H, H - TOP_H], color=BASE, lw=1.2)

# ---- forest axis: gridlines, zero baseline ----
for v in (-0.2, 0.2, 0.4):
    ax.plot([fx(v), fx(v)], [Y(BODY), Y(0)], color=GRID, lw=0.8, zorder=0)
ax.plot([fx(0), fx(0)], [Y(BODY), Y(0)], color=BASE, lw=1.2, zorder=1)
for v in (-0.2, 0.0, 0.2, 0.4):
    ax.text(fx(v), Y(BODY) - 1.2, f"{v:+.1f}" if v else "0", fontsize=9.2, color=MUTED, ha="center", va="top")

for gi, ((g, rows, eff, note), (d0, hs)) in enumerate(zip(GROUPS, layout)):
    gh = sum(hs)
    if gi % 2 == 0:
        ax.add_patch(Rectangle((0.5, Y(d0 + gh)), 179, gh, facecolor=BAND, edgecolor="none", zorder=-1))
    ax.text(X_G, Y(d0) - 1.3, g[0], fontsize=11, color=INK, weight="bold", va="top")
    ax.text(X_G, Y(d0) - 4.0, g[1], fontsize=9.6, color=MUTED, va="top")
    d = d0
    for ri, ((dim, a, b), h) in enumerate(zip(rows, hs)):
        ax.text(X_D, Y(d) - 1.3, dim, fontsize=9.8, color=INK, va="top")
        ax.text(X_R, Y(d) - 1.3, a, fontsize=9.6, color=INK2, va="top", linespacing=1.55)
        ax.text(X_X, Y(d) - 1.3, b, fontsize=9.6, color=INK2, va="top", linespacing=1.55)
        if ri:
            ax.plot([X_D, F0 - 2.5], [Y(d), Y(d)], color=GRID, lw=0.8)
        d += h
    ax.plot([0.5, 179.5], [Y(d0 + gh), Y(d0 + gh)], color=BASE, lw=1.0)
    yc = Y(d0) - 3.3                                   # interval marks at the group's top, note below
    for mi, (k, c, lab) in enumerate(METRICS):
        v, lo, hi = eff[k]; yy = yc - mi * 2.6
        ax.plot([fx(lo), fx(hi)], [yy, yy], color=c, lw=1.6, solid_capstyle="round", zorder=3)
        ax.plot(fx(v), yy, "o", ms=6, color=c, mec=SURF, mew=1.2, zorder=4)
        sig = lo > 0 or hi < 0
        ax.text(X_V, yy, f"{v:+.3f}{' *' if sig else ''}", fontsize=9.4, color=INK if sig else MUTED, va="center")
    ax.text(F0 + 0.3, yc - 5.4, note, fontsize=8.8, color=INK2, va="top", zorder=5,
            bbox=dict(facecolor=BAND if gi % 2 == 0 else SURF, edgecolor="none", pad=1.5))

ax.text(1.5, Y(BODY) - 4.6,
        "消融中间模型(runs/abl_allactor/tahrelmn.py)只把每组差异最小化地加进 TAHRelM:rn 的 null 是 3→16→1 小 MLP(输入 log1p Δt、"
        "最大候选 logit、log1p n);rnw 在 rn 上改用世界系运动 + Δt,保留 msoft。", fontsize=9.6, color=INK2, va="top")
ax.text(1.5, Y(BODY) - 7.4,
        "配对差 = 后一模型 - 前一模型;64 个测试集 × 2 种子,按集 bootstrap 3000 次;* = 95% CI 不含 0(值为灰色表示不显著)。"
        "世界坐标取自仿真真值(训练加 0.5 m 噪声),部署需在估计误差下复核。", fontsize=9.6, color=INK2, va="top")

out = __file__.replace(".py", ".png")
fig.savefig(out, bbox_inches="tight", facecolor=SURF)
print("saved", out)
