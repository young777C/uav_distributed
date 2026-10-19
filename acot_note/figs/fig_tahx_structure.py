"""TAH-X v0 structure diagram (runs/abl_allactor/tahx.py::TAHX, 564,610 params). Same palette / style as
fig_tahrelm_structure.py (dataviz reference categorical slots 1-3, validated light all-pairs: CVD ΔE 9.2, normal 24.0;
aqua < 3:1 contrast → every box carries a visible text label).
  python3 acot_note/figs/fig_tahx_structure.py  → acot_note/figs/fig_tahx_structure.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

font_manager.fontManager.addfont("/home/hque/.local/share/fonts/SimHei.ttf")
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # frozen input / param-free · learned · memory state
TINT = {BLUE: "#e9f1fb", ORANGE: "#fdeee7", AQUA: "#e6f6f0"}

fig, ax = plt.subplots(figsize=(18, 10.8), dpi=150)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.set_xlim(0, 180); ax.set_ylim(0, 108); ax.axis("off")


def box(x, y, w, h, color, title, lines=(), title_size=12.5, body_size=10.2, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", lw=2,
                                edgecolor=color, facecolor=TINT[color], linestyle="--" if dashed else "-"))
    ax.text(x + 1.4, y + h - 2.2, title, fontsize=title_size, color=INK, weight="bold", va="top")
    for i, s in enumerate(lines):
        ax.text(x + 1.4, y + h - 6.2 - i * 3.35, s, fontsize=body_size, color=INK2, va="top")


def arrow(x0, y0, x1, y1, text=None, color=INK2, rad=0.0, ls="-", tx=None, ty=None, cs=None, ha="center"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15, lw=1.6,
                                 color=color, linestyle=ls, connectionstyle=cs or f"arc3,rad={rad}"))
    if text:
        ax.text(tx if tx is not None else (x0 + x1) / 2, ty if ty is not None else (y0 + y1) / 2 + 1.2, text,
                fontsize=9.5, color=MUTED, ha=ha)


ax.text(2, 107, "TAH-X v0(runs/abl_allactor/tahx.py)· 每个决策 tick 对 n 个候选 + 1 个 null 打分 · 共 564,610 个参数",
        fontsize=15, color=INK, weight="bold", va="top")

# ---- legend (≥2 categories → legend; identity never colour-alone: every box is labelled) ----
for i, (c, lab) in enumerate([(BLUE, "冻结输入 / 无参计算"), (ORANGE, "可学参数"), (AQUA, "记忆状态(随时间更新)")]):
    ax.add_patch(FancyBboxPatch((2 + i * 30, 99.2), 3.2, 2.6, boxstyle="round,pad=0.2,rounding_size=0.6",
                                edgecolor=c, facecolor=TINT[c], lw=2))
    ax.text(2 + i * 30 + 4.6, 100.5, lab, fontsize=10.5, color=INK2, va="center")

# ---- ① inputs (frozen) ----
box(2, 64, 34, 32, BLUE, "① 候选输入(每个候选 i)",
    ["RGB 帧 + 框 → 裁剪 ×1.3 → 224", "→ 冻结 DINOv2 ViT-S/14", "→ feat_i ∈ R^384(L2 归一化)",
     "pos_i = [u/W, v/H, depth/100]", "wpos_i ∈ R^2:世界坐标(米)", "Δt_frame:距上一处理帧的时间",
     "(只用模型可知量,不用真值可见性)"])
# ---- ③ memory ----
box(2, 25, 34, 35, AQUA, "③ 目标记忆 mem(非参数状态)",
    ["slots:K = 8 个外观槽 s_k ∈ R^128", "  + 槽位置 sp_k(米)+ 写入时刻 t_k", "raw ∈ R^384:原始 DINO 模板(EMA)",
     "pos, vel:世界系位置 / 速度(m, m/s)", "Δt = t_now - t_last(距上次写入)", "pred = pos + vel·Δt",
     "σ = 2 + 3·Δt 米(缺席越久越宽)", "冷启动:无槽 → 相对特征置 0"])
# ---- ② projection ----
box(44, 76, 36, 20, ORANGE, "② 可学外观投影 proj",
    ["384 → 256 → 128(GELU)→ L2", "p_i ∈ R^128 · 参数 131,456", "替代 TAHRelM 的冻结余弦"])
# ---- ④ relative features ----
box(44, 30, 52, 40, BLUE, "④ 11 维相对特征 rel_i(每个候选)",
    ["smax, smean = max / mean_k (p_i · s_k)   多视角外观",
     "craw  = feat_i · raw                    原始 DINO 余弦",
     "rank  = softmax_i(5·smax)               候选间排名",
     "d = (wpos_i - pred) / σ → d_x, d_y, ‖d‖  世界系运动",
     "log1p(Δt)                               缺席时长",
     "u, v, depth                             图像位置",
     "",
     "运动在世界系:不受无人机自身运动影响",
     "全是相对量 + 显式时间:不记具体车辆"], body_size=10.0)
# ---- ⑤ tokens ----
box(104, 54, 36, 42, ORANGE, "⑤ Token 化(d = 128)",
    ["候选 token(n 个):", "  x_i = cin([p_i ; rel_i])   139→128", "null token(1 个,目标不在):",
     "  x_null = null + null_in([log1p Δt,", "                     log1p n])",
     "记忆 token(k ≤ 8 个):", "  m_k = min([s_k ; (sp_k - pred)/σ ;", "             log1p(t_now - t_k)])",
     "参数:cin 17,920 · min 16,896", "      null + null_in 512"], body_size=9.8)
# ---- ⑥ attention blocks ----
box(146, 54, 32, 42, ORANGE, "⑥ 注意力块 × 2(4 头)",
    ["LN → 交叉注意力:", "  n+1 个 token 读 k 个记忆 token", "LN → 自注意力:", "  n+1 个 token 互相比较",
     "  (相似车竞争;null 作对照)", "LN → FFN 128→256→128", "全部残差连接", "参数 397,568"], body_size=10.0)
# ---- ⑦ heads ----
box(146, 25, 32, 24, ORANGE, "⑦ 输出头",
    ["head 128→1 → logits(n+1)", "  最后一维 = null(目标不在)", "wgate 128→1(仅候选)→ 写门",
     "训练:CE(n+1;缺席 → null)", "  + 0.5·BCE(写门;标签 = 选对否)"], body_size=10.0)
# ---- ⑧ output ----
box(104, 25, 36, 24, BLUE, "⑧ 决策",
    ["s* = argmax(n+1 个 logits)", "s* = null → 判定目标不在,不写记忆", "conf_ok = s* ≠ null", "          且 σ(wgate_s*) > 0.5",
     "→ PGIF 证据 / WHERE·HOW 控制"], body_size=10.0)
# ---- ⑨ memory update (hand-coded rules, no params) ----
box(44, 5, 134, 16.5, BLUE, "⑨ 记忆更新(无参规则;仅在写入时执行)",
    ["vel ← 0.5·vel + 0.5·clip((wpos_s* - pos)/Δt, ±30) ;  pos ← wpos_s* ;  t_last ← t_now ;  "
     "raw ← norm(0.7·raw + 0.3·feat_s*)",
     "槽:距上次建槽 < 1 s → 与最后一槽 0.5/0.5 融合(保留建槽时刻);与某槽 cos > 0.98 → 替换;"
     "否则新建一槽,超过 K = 8 时先进先出",
     "何时写:训练 = 真值(以 p_ss 概率改用自身选择,p_ss 由 0 升到 0.5);部署 SU = 非 null 即写;"
     "SUG = 仅 conf_ok 时写"], body_size=10.0)

# ---- flows ----
arrow(36.5, 88, 43.5, 88, "feat_i", ty=89.4)
arrow(36.5, 67, 43.5, 67, "pos,wpos", tx=40, ty=68.2)
arrow(36.5, 45, 43.5, 45, "mem", tx=40, ty=46.2)
arrow(60, 75.5, 60, 70.5, "p_i", tx=62, ty=72.2, ha="left")
arrow(80.5, 88, 103.5, 88, "p_i", ty=89.4)
arrow(96.5, 62, 103.5, 62, "rel_i", ty=63.4)
ax.plot([36.6, 100, 100], [27, 27, 58], color=AQUA, lw=1.6, ls="--")
arrow(100, 58, 103.5, 58, color=AQUA)
ax.text(70, 24.3, "s_k, sp_k, t_k → 记忆 token", fontsize=9.5, color=MUTED, ha="center")
arrow(140.5, 75, 145.5, 75)
arrow(162, 53.5, 162, 49.5, "n+1 个输出 token", tx=163.5, ty=51, ha="left")
arrow(145.5, 37, 140.5, 37)
arrow(122, 24.5, 122, 22.0, "s*(非 null 时)", tx=124, ty=22.6, ha="left")
arrow(44, 13, 19, 24.5, "写回 mem", rad=-0.25, ls="--", color=AQUA, tx=27, ty=13.5)

ax.text(2, 1.2, "相对 TAHRelM 新增:可学外观投影 · K 槽多视角记忆 · 世界系运动 + 显式 Δt · 候选间注意力竞争 · null(目标不在)输出 · 写门 · "
        "计划采样;§9 变体 xs 另加来源 token(加到全部 n+1 个 token,默认关闭)", fontsize=10.5, color=INK2)

out = __file__.replace(".py", ".png")
fig.savefig(out, bbox_inches="tight", facecolor=SURF)
print("saved", out)
