"""ACoT-UAV-Track 最终模型完整架构(部署设定 v2 + 开局属性模块 v1),并附开局变体 v2–v4 参考面板。dataviz 方法绘制。
= 当前最终模型(2026-10-04 起)。旧图已标为 legacy:fig_architecture_legacy_pgif.*(TAHRelM + PGIF,理想 ID 旧设定)、
fig_architecture_legacy_tahrelm.*(仅 TAHRelM)。10-04 更新：如实披露两项理想化感知输入(候选集仅含任务车辆、
WHERE 用仿真真值世界坐标),注明 TAHRelM 权重 tah_relm_dag.pt(FOV 修正重训经 dev 验证未采用)。
主图 = 数据流示意(非数据图表)。配色按「作用」取色：冻结 = 中性灰;类别色只给三个非冻结角色，按调色板固定顺序取槽位:
  slot1 蓝 #2a78d6 = TAHRelM(训练过，部署冻结);slot2 橙 #eb6834 = PGIF(推理期规则,0 参数);
  slot3 aqua #1baf7a = 开局属性模块 OPENING=attr(新增，训练过，部署冻结)。
  validate_palette.py "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all → 全部 PASS;aqua 对比度 2.74:1 → WARN,
  按 relief 规则所有 aqua 元素都配可见文字标签。文字一律用正文墨色，类别由边框和填充承担。
参考面板 = 小多图:4 个指标各用一张图、各自的坐标轴，不用双轴。每张图都是水平点图加 95% CI,配置顺序固定;
v1(采用)用 aqua 高亮，其余用中性墨色。数字全部从评测文件读取，不手抄:
  开局锁对(窗口后，留出种子)  carla_uav_tracking/rollout/eval_out/open_val_<tag>_<k>.json + runs/opening_dump/val/<tag>_<k>.jsonl
  dev 下游三项指标           eval_out/paper_it_{v2h_pgif_trk, zoa_pgif_trk_*, zo3_pgif_trk_*, zo4_pgif_trk_*}*.json
数据流与代码对应:rollout/policy.py::act,开局模块在 rollout/opening_zoom.py::AttrScorer,PGIF 在 rollout/asd.py,
TAHRelM 在 train/tah.py。方案文档：acot_note/1004_opening_fix.md
  python acot_note/figs/fig_architecture_v1.py → fig_architecture_v1.{png,pdf,svg}"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ROLL = os.path.join(REPO, "carla_uav_tracking", "rollout")
sys.path.insert(0, ROLL)
import iter_gate as ig  # noqa: E402
import opening_cl as oc  # noqa: E402

_f = os.path.expanduser("~/.local/share/fonts/SimHei.ttf")
if os.path.exists(_f):
    fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 8.5})

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TAH, TAH_BG = "#2a78d6", "#dbe8f8"          # categorical slot 1
PG, PG_BG = "#eb6834", "#fde3d7"            # categorical slot 2
OP, OP_BG = "#1baf7a", "#d9f2e8"            # categorical slot 3
FROZ, FROZ_E, INP = "#e6e5e1", "#b9b8b2", "#ffffff"


# ============================================================ data (read from the evaluation outputs)
def opening_rate(tag, ks):
    """closed-loop opening lock after the window (first S2 tick with the target present at t ≥ 12) → (k, n)."""
    T = {}
    for k in ks:
        j = os.path.join(ROLL, "eval_out", f"open_val_{tag}_{k}.json")
        j = j if os.path.exists(j) else j + ".partial"
        jl = os.path.join(REPO, "runs", "opening_dump", "val", f"{tag}_{k}.jsonl")
        if os.path.exists(j) and os.path.exists(jl):
            T.update(oc.load([(jl, j)]))
    ok = []
    for R in T.values():
        P = [r for r in R if r.get("true") is not None and r["true"] >= 0]
        rl = next((r for r in P if r["t"] >= 12), None)
        if rl is not None:
            ok.append(oc.out_id(rl) == rl["aidx"][rl["true"]])
    return sum(ok), len(ok)


def arm(stems):
    E = {}
    for s in stems:
        E.update(ig.load(s) or {})
    return E


def mean_ci(x, seed=20261003):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (20000, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


CFG = [  # (row label, opening tag + jobs, seeds, downstream arm stems)
    ("H-A 原开局\n(grounding 头)", ("head", (5, 6, 7)), "93100–239", ["v2h_pgif_trk"]),
    ("v1 attr(采用)", ("attr", (1, 2, 3, 4, 9)), "93100–239", ["zoa_pgif_trk_a", "zoa_pgif_trk_b", "zoa_pgif_trk_c"]),
    ("v2", ("fix", tuple(range(20, 26))), "93240–379", []),
    ("v3", ("v3", tuple(range(40, 48))), "93380–519", ["zo3_pgif_trk_a", "zo3_pgif_trk_b"]),
    ("v4", ("v4", tuple(range(60, 65))), "93520–659", [f"zo4_pgif_trk_{x}" for x in "abcde"]),
]
ROWS = []
for lab, (tag, ks), seeds, stems in CFG:
    k, n = opening_rate(tag, ks); lo, hi = oc.wilson(k, n)
    r = {"lab": lab, "open": (k / n, lo, hi, k, n), "seeds": seeds}
    E = arm(stems) if stems else {}
    if E:
        eps = list(E.values())
        r["n_ds"] = len(eps)
        r["correct"] = mean_ci([ig.METRICS["correct"](e) for e in eps])
        r["longC"] = mean_ci([ig.METRICS["longC"](e) for e in eps])
        r["longW"] = mean_ci([ig.METRICS["longW"](e) for e in eps])
    ROWS.append(r)

# ============================================================ canvas
fig = plt.figure(figsize=(20, 16.5), facecolor=SURF)
ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, title, sub="", fc=FROZ, ec=FROZ_E, lw=1.0, fs=9.2, bold=False, ls="-", top=False, sfs=7.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=lw,
                                linestyle=ls, zorder=2))
    if top:
        ax.text(x + w / 2, y + h - 0.8, title, ha="center", va="top", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", zorder=3)
        return
    if sub:
        ax.text(x + w / 2, y + h - 0.9, title, ha="center", va="top", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", zorder=3)
        ax.text(x + w / 2, y + h - 1.15 - fs * 0.2, sub, ha="center", va="top", fontsize=sfs, color=INK2,
                zorder=3, linespacing=1.4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", zorder=3)


def arrow(x1, y1, x2, y2, col=INK2, lw=1.2, ls="-", rad=0.0, label=None, lx=None, ly=None, fs=7.8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=col, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        ax.text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2, label, fontsize=fs,
                color=INK2, ha="center", va="center", zorder=4, bbox=dict(fc=SURF, ec="none", pad=0.4))


# ---------------- title + legend
v1 = ROWS[1]; ha = ROWS[0]
ax.text(0.8, 99.4, "ACoT-UAV-Track 最终模型完整架构:属性开局(v1)+ TAHRelM + PGIF(部署设定 v2:身份决策不使用真值)",
        fontsize=15, fontweight="bold", color=INK, va="top")
ax.text(0.8, 97.3, "训练过的部件:开局属性分类器(DINOv2 线性头 + 微调头，仿真属性标签，独立种子 96000–96279)与 TAHRelM"
                   "(4,740 参数，权重 tah_relm_dag.pt);两者部署时都冻结。PGIF 为推理期规则(0 参数);"
                   "Qwen3-VL / IAR / crop-DINOv2 / DiT 全部冻结。", fontsize=9, color=INK2, va="top")
ax.text(0.8, 95.75, "理想化感知输入(所有对比臂相同，如实披露):候选框为仿真检测替身(真值投影与真值深度，按射线遮挡漏检),"
                    "且只含目标与场景指定的干扰车(背景车流不作为候选);WHERE 使用匹配框的仿真真值世界坐标(理想定位)。",
        fontsize=9, color=INK, va="top")
ax.legend(handles=[Patch(fc=OP_BG, ec=OP, lw=2, label="开局属性模块(新增，训练过，部署冻结)"),
                   Patch(fc=TAH_BG, ec=TAH, lw=2, label="TAHRelM(训练过，部署冻结)"),
                   Patch(fc=PG_BG, ec=PG, lw=2, label="PGIF(推理期规则,0 参数)"),
                   Patch(fc=FROZ, ec=FROZ_E, label="冻结模块"), Patch(fc=INP, ec=FROZ_E, label="输入 / 环境")],
          loc="upper left", bbox_to_anchor=(0.006, 0.942), ncol=5, fontsize=8.8, frameon=False, handlelength=1.6)

# ============================================================ main architecture (y 41–93), four columns
C1, C2, C3, C4 = (0.8, 12.0), (15.5, 23.0), (41.5, 23.0), (67.5, 31.7)
# ---- column 1: inputs (order chosen so that every input feeds the module beside it without crossings)
box(C1[0], 84.5, C1[1], 7.4, "语言指令", "去位置化，只保留身份\n“Follow the gold\nMini Cooper.”", fc=INP)
box(C1[0], 75.5, C1[1], 7.5, "高清开局帧", "同位姿 3× 相机(1536×1536)\n仅 tick 0 / 6 / 12,随后销毁", fc=INP)
box(C1[0], 63.5, C1[1], 10.5, "候选集 + 跟踪器 ID", "仿真框(检测器替身，仅任务车辆;\n背景车流不作为候选)\n射线遮挡过滤：只留可见车\nGT-free 跟踪器(匈牙利 + 外观)", fc=INP)
box(C1[0], 53.0, C1[1], 8.0, "无人机 RGB", "512×512 机载相机\n每帧", fc=INP)
box(C1[0], 44.5, C1[1], 6.5, "Proprio 5D", "cam_z, pitch, uav_vel×3", fc=INP)

# ---- column 2: opening module (NEW) + frozen perception
x, w = C2
ax.add_patch(FancyBboxPatch((x, 70), w, 22, boxstyle="round,pad=0.25,rounding_size=1.0", fc=OP_BG, ec=OP, lw=2.2, zorder=1.5))
ax.text(x + w / 2, 91.7, "开局身份模块 OPENING=attr(v1,新增)", fontsize=10, fontweight="bold", color=INK, ha="center", va="top")
ax.text(x + w / 2, 89.9, "只在开局窗口 tick 0 / 6 / 12 运行", fontsize=8, color=INK2, ha="center", va="top")
box(x + 1, 84.0, w - 2, 4.6, "① 语言解析 describe()", "去方位 / 距离 → 颜色词 c*、车型名词 m*", fc="#ffffff", ec=OP, lw=1.2, sfs=7.6)
box(x + 1, 76.3, w - 2, 6.9, "② 属性分类器(DINOv2 ViT-S/14)",
    "高清帧真实像素裁剪(1.5× / 2.0× 框 + 翻转)\n线性头(冻结特征 ⊕ RGB 直方图)⊕ 微调头(后 4 block)\n按 ft = 0.5 等权混合(17 车型 · 18 颜色)\n分数 = log p(m* | 裁剪) + log p(c* | 裁剪)",
    fc="#ffffff", ec=OP, lw=1.2, sfs=7.4)
box(x + 1, 70.8, w - 2, 4.7, "③ 开局窗口聚合(v1)", "每 tick 分数 - max,按跟踪器 ID 累加 → argmax\n= 开局身份(替代语言 grounding 选车)",
    fc="#ffffff", ec=OP, lw=1.2, sfs=7.4)
box(x, 61.0, w, 7.0, "crop-DINOv2 ViT-S/14(冻结,S2)", "每个候选框从 RGB 裁剪 → 384 维外观\n+ 位置(u/W, v/H, depth)", bold=True)
box(x, 51.5, w, 7.5, "Qwen3-VL-4B(冻结,44.4 亿,S2)", "指令文本在前、RGB 在后，一次前向\n第 24 层隐藏态 → vlm_ctx(256×2560)\n其语言 grounding 头不再用于开局", bold=True)
box(x, 44.5, w, 5.5, "IAR(冻结,97 万,S2)", "8 个可学 query 对第 24 层特征\ncross-attention → z_im(8×256)", bold=True)

# ---- column 3: TAHRelM + identity memory + DiT
x, w = C3
ax.add_patch(FancyBboxPatch((x, 70), w, 22, boxstyle="round,pad=0.25,rounding_size=1.0", fc=TAH_BG, ec=TAH, lw=2.2, zorder=1.5))
ax.text(x + w / 2, 91.7, "TAHRelM(WHICH,4,740 参数,S2)", fontsize=10, fontweight="bold", color=INK, ha="center", va="top")
box(x + 1, 83.6, w - 2, 6.0, "7 个车辆不变的相对特征", "cos(外观, 记忆)· 相对位移 x / y · 深度\n排名 · 距离 · 软运动门 exp(-d / gate)", fc="#ffffff", ec=TAH, lw=1.2, sfs=7.6)
box(x + 1, 79.0, w - 2, 3.6, "MLP 7·64·64·1 → 每个候选的分数", fc="#ffffff", ec=TAH, lw=1.2, fs=8.6)
box(x + 1, 70.8, w - 2, 7.2, "原始选择 $s^M$ = argmax,置信度 c", "开局窗口 tick 0 / 6:记忆重置，记忆为空(冷启动)\n→ $s^M$ = 开局身份，并写入记忆\ntick 12 起：由记忆打分", fc="#ffffff", ec=TAH, lw=1.2, sfs=7.6)
box(x, 60.5, w, 7.5, "身份记忆 {app, pos, vel}", "外观：可学门 EMA;速度：可学衰减 EMA\n开局由开局身份播种，之后由 PGIF 门控写入", fc=TAH_BG, ec=TAH, lw=1.6, bold=True)
box(x, 44.5, w, 13.5, "HOW:DiT(冻结,32 层,1.37 亿)", "S1 快流，每帧运行\nflow-matching 生成动作块(H = 16)\n条件:z_ex · z_im · vlm_ctx · proprio\n→ (dx, dy, dz, dyaw) + search_mode", bold=True)

# ---- column 4: PGIF + committed identity + WHERE + CARLA
x, w = C4
ax.add_patch(FancyBboxPatch((x, 70), w, 22, boxstyle="round,pad=0.25,rounding_size=1.0", fc=PG_BG, ec=PG, lw=2.2, zorder=1.5))
ax.text(x + w / 2, 91.7, "PGIF(推理期规则,0 可训练参数,S2)", fontsize=10, fontweight="bold", color=INK, ha="center", va="top")
box(x + 1, 84.3, w - 2, 5.3, "① 置信度秩 → 投票权重", "ρ = c 在最近 W = 30 步中的分位数;w = 0.5 + ρ", fc="#ffffff", ec=PG, lw=1.2)
box(x + 1, 78.0, w - 2, 5.3, "② 回火 tracklet 证据(按跟踪器 ID)", "$L_k \\leftarrow 0.98\\,L_k$;$L_{s^M}$ += w(τ ≈ 50 步 ≈ 30 s)", fc="#ffffff", ec=PG, lw=1.2)
box(x + 1, 70.8, w - 2, 6.2, "③ 决策 + 不可见假设(absent-hold)",
    "MAP tracklet 本 tick 不可见 → 输出「缺席」:保持承诺、撤销本步记忆写入(≤ 8 tick)\n否则 $\\hat{s}$ = argmax 可见 $L_k$;若 $\\hat{s} \\neq s^M$ → 改写承诺，并用 $\\hat{s}$ 门控记忆更新",
    fc="#ffffff", ec=PG, lw=1.2, sfs=7.4)
box(x, 60.5, w, 7.5, "承诺身份 $\\hat{s}$(跟踪器 ID)", "K = 5 帧滞回;PGIF 改写时直接承诺;缺席时保持上一承诺", fc=INP, ec=PG, lw=1.6, bold=True)
box(x, 51.5, w, 7.0, "WHERE(冻结规则)", "承诺 ID 本 tick 匹配到的框 → 仿真真值世界坐标(理想定位)→ 匀速外推 → 投影到相机 → z_ex\n承诺 ID 不可见时按匀速估计滑行", bold=True)
box(x, 44.5, w, 5.0, "CARLA 闭环", "无人机执行动作 → 下一帧(选哪辆车决定下一帧看到什么)", fc=INP, bold=True)

# ---- arrows: inputs → column 2
arrow(12.8, 86.3, 16.5, 86.3, col=OP, lw=1.6)                               # language → ①
arrow(12.8, 79.2, 16.5, 79.2, col=OP, lw=1.6)                               # hi-res → ②
arrow(12.8, 72.0, 16.5, 73.0, col=OP, lw=1.6)                               # candidates (boxes + ids) → ③
arrow(12.8, 65.5, 15.5, 65.0)                                               # candidates → crop-DINO
arrow(12.8, 59.5, 15.5, 62.5)                                               # RGB → crop-DINO
arrow(12.8, 55.0, 15.5, 55.0)                                               # RGB → Qwen
ax.plot([12.8, 14.15, 14.15], [85.0, 85.0, 57.6], color=MUTED, lw=1.0, zorder=1)  # language → Qwen (channel)
arrow(14.15, 57.6, 15.5, 57.6, col=MUTED, lw=1.0)
ax.text(14.15, 60.0, "指令", fontsize=7.4, color=INK2, ha="center", va="center", rotation=90,
        bbox=dict(fc=SURF, ec="none", pad=0.3), zorder=4)
arrow(27.0, 84.0, 27.0, 83.2, col=OP); arrow(27.0, 76.3, 27.0, 75.5, col=OP)  # ① → ② → ③
arrow(27.0, 51.5, 27.0, 50.0)                                               # Qwen → IAR
# ---- column 2 → column 3
arrow(37.5, 73.2, 42.5, 74.4, col=OP, lw=1.8, label="开局身份", lx=40.0, ly=75.6)       # opening → TAHRelM cold start
arrow(38.5, 66.0, 42.0, 71.0, label="外观 + 位置", lx=40.0, ly=68.4)                     # crop-DINO → TAHRelM
arrow(38.5, 55.0, 41.5, 55.0, label="vlm_ctx", lx=40.0, ly=56.4)                        # Qwen → DiT
arrow(38.5, 47.0, 41.5, 47.0, label="z_im", lx=40.0, ly=48.4)                           # IAR → DiT
# ---- column 3 ↔ column 4
arrow(53.0, 68.0, 53.0, 70.0, col=TAH, lw=1.4)                                          # memory → TAHRelM
ax.text(54.0, 69.0, "记忆参与打分", fontsize=7.4, color=INK2, ha="left", va="center")
arrow(63.5, 74.4, 68.5, 74.4, col=TAH, lw=1.6, label="$s^M$, c", lx=66.0, ly=75.7)     # TAHRelM → PGIF
arrow(68.8, 70.0, 64.5, 66.0, col=PG, lw=1.6, label="门控写入", lx=66.6, ly=69.4)       # PGIF → memory
arrow(83.35, 70.0, 83.35, 68.0, col=PG, lw=1.6)                                         # PGIF → committed
arrow(83.35, 60.5, 83.35, 58.5)                                                         # committed → WHERE
arrow(67.5, 53.5, 64.5, 53.5, label="z_ex", lx=66.0, ly=54.8)                           # WHERE → DiT
arrow(64.5, 47.0, 67.5, 47.0, label="动作", lx=66.0, ly=48.4)                           # DiT → CARLA
# ---- proprio → DiT and the closed loop (routed below the boxes)
ax.plot([6.8, 6.8, 50.0], [44.5, 42.9, 42.9], color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=1)
arrow(50.0, 42.9, 50.0, 44.3, col=MUTED, lw=1.0, ls=(0, (3, 2)))
ax.text(28.0, 42.9, "proprio", fontsize=7.4, color=INK2, ha="center", va="center", bbox=dict(fc=SURF, ec="none", pad=0.3))
ax.plot([83.35, 83.35, 0.35, 0.35], [44.5, 41.7, 41.7, 57.0], color=INK2, lw=1.1, zorder=1)
arrow(0.35, 57.0, 0.8, 57.0, col=INK2, lw=1.1)
ax.text(42.0, 41.7, "下一帧(闭环)", fontsize=7.4, color=INK2, ha="center", va="center", bbox=dict(fc=SURF, ec="none", pad=0.3))

# ============================================================ reference panel (y 0–40)
ax.plot([0.8, 99.2], [40.2, 40.2], color=AXIS, lw=1.0)
ax.text(0.8, 39.4, "参考：开局模块的窗口逻辑变体 v2 / v3 / v4(均为 OPENING_CFG 开关，默认关闭)与 v1 的对比",
        fontsize=12, fontweight="bold", color=INK, va="top")
ax.text(0.8, 37.3, "四个变体共用同一个属性分类器(② 相同)，只差在开局窗口怎样把逐 tick 的分数变成承诺，以及怎样初始化 TAH 记忆和 PGIF 证据。"
                   "开局锁对 = 窗口结束后(首个 t ≥ 12 且目标在候选中的 S2 tick)承诺身份等于目标。",
        fontsize=8.6, color=INK2, va="top")

# ---- design table
cols = [(0.8, "版本"), (8.5, "每个开局 tick 的决定"), (30.0, "最后开局 tick(t = 12)的 TAH 记忆"),
        (53.0, "PGIF 证据"), (67.0, "结论")]
ty = 34.4
for x, t in cols:
    ax.text(x, ty, t, fontsize=8.8, color=INK2, fontweight="bold", va="center")
ax.plot([0.8, 99.2], [ty - 1.0, ty - 1.0], color=GRID, lw=0.8)
TAB = [
    ("H-A 原开局", "语言 grounding 头 argmax(512×512,中心先验)", "—", "正常累积", "开局锁对约 35%:瓶颈在细粒度属性识别(车型、颜色)"),
    ("v1(采用)", "分数 - max,按跟踪器 ID 累加 → argmax", "不重置：由记忆打分并写入(t = 6、12 两个样本)", "正常累积",
     "开局 ≥ 90%,下游三项指标最好；剩余开局失败约一半来自窗口内 ID 互换"),
    ("v2", "按跟踪器 ID 取原始分均值", "重置 → 写入开局选择(1 个样本)", "每个开局 tick 清空", "ID 互换后旧 ID 的历史仍跟着干扰车；开局没有改善"),
    ("v3", "只用本 tick 分数(不受 ID 互换影响)", "重置 → 写入开局选择(1 个样本)", "每个开局 tick 清空",
     "开局略高，但记忆与 PGIF 惯性变弱 → 开局后更早锁错"),
    ("v4", "只用本 tick 分数", "不重置：强制写入开局选择(2 个样本)", "重新播种 {开局 ID: 3.0}", "开局与 v1 持平，下游仍不如 v1"),
]
for i, row in enumerate(TAB):
    y = ty - 2.6 - i * 2.2
    if i == 1:
        ax.add_patch(FancyBboxPatch((0.6, y - 0.95), 98.8, 1.9, boxstyle="round,pad=0.05,rounding_size=0.4", fc=OP_BG,
                                    ec="none", zorder=0.5))
        ax.add_patch(FancyBboxPatch((0.6, y - 0.95), 0.45, 1.9, boxstyle="square,pad=0", fc=OP, ec="none", zorder=0.6))
    for (x, _), t in zip(cols, row):
        ax.text(x + (0.7 if x < 1 else 0), y, t, fontsize=8.4, color=INK, va="center",
                fontweight="bold" if (i == 1 and x < 1) else "normal")

# ---- small multiples: one metric per chart, own axis, fixed row order (no dual axis)
CH = [("开局锁对率(%,留出种子)", "open", 100.0, (0, 100)),
      ("dev 承诺正确锁定比例 ↑", "correct", 1.0, (0, 1)),
      ("dev 最长连续正确(s)↑", "longC", 1.0, (0, 30)),
      ("dev 最长锁错(s)↓", "longW", 1.0, (0, 30))]
labels = ["H-A 原开局", "v1(采用)", "v2", "v3", "v4"]
x0s = [0.06, 0.30, 0.54, 0.78]
for (title, key, scale, xlim), x0 in zip(CH, x0s):
    a = fig.add_axes((x0, 0.045, 0.2, 0.14), facecolor=SURF)
    a.set_title(title, fontsize=9.4, color=INK, loc="left", pad=8)
    for spine in ("top", "right", "left"):
        a.spines[spine].set_visible(False)
    a.spines["bottom"].set_color(AXIS)
    a.tick_params(axis="x", colors=MUTED, labelsize=8, length=0, pad=4)
    a.tick_params(axis="y", length=0, labelsize=8.6, colors=INK)
    a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True)
    a.set_xlim(*xlim); a.set_ylim(len(ROWS) - 0.4, -0.75)
    a.set_yticks(range(len(ROWS)))
    a.set_yticklabels(labels if x0 == x0s[0] else [""] * len(ROWS))
    span = xlim[1] - xlim[0]
    if key == "open":
        a.axvline(90, color=INK2, lw=1.0, ls=(0, (3, 2)), zorder=1)
        a.text(90, -0.72, "目标 90%", fontsize=7.8, color=INK2, ha="center", va="bottom")
    for i, r in enumerate(ROWS):
        v = r.get(key)
        if v is None:
            a.text(xlim[0] + 0.02 * span, i, "未测(未跑下游)", fontsize=8, color=MUTED, va="center")
            continue
        m, lo, hi = v[0] * scale, v[1] * scale, v[2] * scale
        col = OP if i == 1 else INK2
        a.plot([lo, hi], [i, i], color=col, lw=2.0, solid_capstyle="round", zorder=2)
        a.scatter([m], [i], s=64, color=col, edgecolor=SURF, linewidth=2.0, zorder=3)
        if key == "open":
            txt = f"{m:.1f}%({v[3]}/{v[4]})"
        elif key == "correct":
            txt = f"{m:.2f}" + (f"(n={r['n_ds']})" if r["n_ds"] != 19 else "")
        else:
            txt = f"{m:.1f} s" + (f"(n={r['n_ds']})" if r["n_ds"] != 19 else "")
        right = hi + 0.02 * span < xlim[1] - 0.3 * span           # room on the right of the CI → label there
        a.text(hi + 0.02 * span if right else lo - 0.02 * span, i, txt, fontsize=7.9, color=INK, va="center",
               ha="left" if right else "right", fontweight="bold" if i == 1 else "normal")
ax.text(0.8, 0.4, "点 = 均值(开局为比例);横线 = 95% CI(开局用 Wilson,下游用种子 bootstrap)。v1 以 aqua 高亮，其余为中性墨色。"
                  "开局种子:H-A 与 v1 为 93100–239(同种子配对,McNemar p = 1e-18),v2 为 93240–379,v3 为 93380–519(全新，无偏),v4 为 93520–659。"
                  "下游为 dev 91000–19 全长闭环(v3 提前停止,n = 13)。",
        fontsize=7.8, color=INK2, va="bottom")

for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(HERE, f"fig_architecture_v1.{ext}"), dpi=150 if ext == "png" else None, facecolor=SURF)
print("saved fig_architecture_v1.{png,pdf,svg}")
for r in ROWS:
    print(r["lab"].replace("\n", " "), "open %.1f%% [%.1f, %.1f] %d/%d" % (r["open"][0] * 100, r["open"][1] * 100, r["open"][2] * 100, r["open"][3], r["open"][4]),
          {k: tuple(round(x, 3) for x in r[k]) for k in ("correct", "longC", "longW") if k in r}, r.get("n_ds"))
