"""TAHRelM structure diagram (train/tah.py::TAHRelM, 4,740 params). Palette: dataviz reference categorical slots 1-3
(validated light, all-pairs: CVD ΔE 9.2, normal 24.0; aqua < 3:1 contrast → every box carries a visible text label).
  python3 acot_note/figs/fig_tahrelm_structure.py  → acot_note/figs/fig_tahrelm_structure.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

font_manager.fontManager.addfont("/home/hque/.local/share/fonts/SimHei.ttf")
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # frozen input · learned · memory state
TINT = {BLUE: "#e9f1fb", ORANGE: "#fdeee7", AQUA: "#e6f6f0"}

fig, ax = plt.subplots(figsize=(17, 9.2), dpi=150)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.set_xlim(0, 170); ax.set_ylim(0, 92); ax.axis("off")


def box(x, y, w, h, color, title, lines=(), title_size=12.5, body_size=10.2, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", lw=2,
                                edgecolor=color, facecolor=TINT[color], linestyle="--" if dashed else "-"))
    ax.text(x + 1.4, y + h - 2.2, title, fontsize=title_size, color=INK, weight="bold", va="top")
    for i, s in enumerate(lines):
        ax.text(x + 1.4, y + h - 6.2 - i * 3.35, s, fontsize=body_size, color=INK2, va="top")


def arrow(x0, y0, x1, y1, text=None, color=INK2, rad=0.0, ls="-", tx=None, ty=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15, lw=1.6,
                                 color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text(tx if tx is not None else (x0 + x1) / 2, ty if ty is not None else (y0 + y1) / 2 + 1.2, text,
                fontsize=9.5, color=MUTED, ha="center")


ax.text(2, 89.5, "TAHRelM(train/tah.py)· 每个决策 tick 对 n 个候选打分 · 共 4,740 个参数(MLP 4,737 + 3 个可学标量)",
        fontsize=15, color=INK, weight="bold", va="top")

# ---- inputs (frozen) ----
box(2, 58, 33, 24, BLUE, "① 候选输入(每个候选 i)",
    ["RGB 帧 + 候选框(伪方框)", "→ 裁剪(框 ×1.3)→ 缩放 224", "→ 冻结 DINOv2 ViT-S/14", "→ feat_i ∈ R^384(L2 归一化)",
     "pos_i = [u/W, v/H, depth/100]"])
# ---- memory ----
box(2, 20, 33, 28, AQUA, "② 目标记忆 mem(非参数状态)",
    ["app ∈ R^384:外观模板(EMA)", "pos ∈ R^2:上次位置 (u, v)", "vel ∈ R^2:图像平面速度(EMA)",
     "匀速预测 pred = pos + vel", "冷启动:mem 为空 → 特征全 0", "(部署时由语言选车锁定)"])
# ---- relative features ----
box(45, 43, 55, 39, BLUE, "③ 7 个车辆无关的相对特征(每个候选)",
    ["cos    = feat_i · app            外观相似度",
     "relp_x,y = pos_i - pred           相对匀速预测的偏移",
     "depth  = pos_i[2]",
     "rank   = softmax_i(5·cos)         在候选中的外观排名",
     "dmot   = ‖relp‖                   离预测位置的距离",
     "msoft  = exp(-dmot / r)           软运动门",
     "         r = softplus(mgate) ≈ 0.10·W(≈53 px)",
     "",
     "只有相对量:不记住具体车辆 → 不过拟合 64 集",
     "外观度量 = 冻结 DINOv2 余弦,不可学"], body_size=10.2)
# ---- MLP ----
box(111, 58, 30, 24, ORANGE, "④ 打分 MLP(所有候选共享)",
    ["7 → 64 → 64 → 1(GELU)", "参数 4,737", "输出 logit_i", "选择 = argmax_i logit_i",
     "conf = σ(top1 - top2)"])
# ---- output ----
box(146, 62, 23, 20, BLUE, "⑤ 输出",
    ["选中的候选 s*", "置信度 conf", "→ PGIF 证据 / 承诺门控", "→ WHERE / HOW 控制"])
# ---- update ----
box(101, 8, 68, 30, ORANGE, "⑥ 记忆更新(3 个可学标量)",
    ["app ← normalize((1-g)·app + g·feat_s*)     g = σ(gate) ≈ 0.30",
     "vel ← (1-b)·vel + b·(pos_s* - pos)          b = σ(vbeta) ≈ 0.46",
     "pos ← pos_s*                               (运动门半径 mgate 见 ③)",
     "",
     "训练:用真值目标写入(teacher forcing);目标缺席时保持不动",
     "部署:用模型自己的选择写入(缺席时也写 → 确认偏差)",
     "数值取自 10-03 消融学到的参数"], body_size=10.2)

# ---- flows ----
arrow(36, 72, 44.5, 72, "feat, pos", tx=40.2, ty=73.6)
arrow(36, 36, 44.5, 46, "app, pred", tx=39.0, ty=40.0)
arrow(101, 72, 110.5, 72, "x ∈ R^7", ty=73.6)
arrow(142, 72, 145.5, 72)
arrow(157, 61.5, 152, 38.5, "s* 的 feat, pos", tx=162.5, ty=50)
arrow(100.5, 20, 36, 28, "写回 mem(下一 tick 使用)", rad=-0.10, ls="--", color=AQUA, tx=68, ty=16)

# ---- legend (≥2 categories → legend present; identity never colour-alone: every box is labelled) ----
lx, ly = 2, 7
for i, (c, lab) in enumerate([(BLUE, "冻结输入 / 无参计算"), (ORANGE, "可学参数"), (AQUA, "记忆状态(随时间更新)")]):
    ax.add_patch(FancyBboxPatch((lx + i * 30, ly), 3.2, 2.6, boxstyle="round,pad=0.2,rounding_size=0.6",
                                edgecolor=c, facecolor=TINT[c], lw=2))
    ax.text(lx + i * 30 + 4.6, ly + 1.3, lab, fontsize=10.5, color=INK2, va="center")
ax.text(2, 2.2, "局限:单一外观模板;图像平面运动(相机在动);没有\"目标不在\"输出;外观度量不可学;输入只有 7 维",
        fontsize=10.5, color=INK2)

out = __file__.replace(".py", ".png")
fig.savefig(out, bbox_inches="tight", facecolor=SURF)
print("saved", out)
