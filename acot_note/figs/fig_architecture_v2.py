"""ACoT-UAV-Track 当前模型架构 v2(2026-10-06):属性开局 v1 + TAHRelM + PGIF(tempered + absent-hold),
部署设定 v2,并带两处改动——跟踪器匹配 贪心 → 匈牙利(perception.tracker.assign = hungarian)、承诺目标本 tick
无匹配框时 WHERE 从最后一次确信位置匀速外推(control.absent_where = cv)。对应实验臂 ab_where_assign/cv_hung。
按代码绘制(acot/runtime/{perception,frontend,heads,pgif,policy}.py,见 1001_innovation-design.md §3.2),
取代 fig_architecture_v1 中三处与代码不符的画法：跟踪器标为匈牙利(v1 实际为贪心)、「不可见时匀速滑行」(v1 实际沿用
EAR 航点)、缺少 EAR 模块。
配色沿用 v1 图(validate_palette 已过):slot1 蓝 = TAHRelM,slot2 橙 = PGIF,slot3 aqua = 开局属性模块;冻结 = 中性灰。
本版改动的两个模块不新增类别色，用墨色虚线边框 + 「v2 改动」标签标出(文字始终可读)。
底部面板：改动 ① 的离线回放数字由 acot.analysis.trk_assign 现算;闭环结果读 eval_out/exp/ab_where_assign,
未完成时显示进度。
变体(2026-10-06 §5.5c 之后):
  python acot_note/figs/fig_architecture_v2.py          → fig_architecture_v2.*  cv_hung(两处改动都打开)
  python acot_note/figs/fig_architecture_v2.py hung     → fig_architecture_v3.*  hung = 当前模型(匈牙利 + 原 WHERE)"""
import glob
import json
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
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "carla_uav_tracking"))

_f = os.path.expanduser("~/.local/share/fonts/SimHei.ttf")
if os.path.exists(_f):
    fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 8.5})

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TAH, TAH_BG = "#2a78d6", "#dbe8f8"
PG, PG_BG = "#eb6834", "#fde3d7"
OP, OP_BG = "#1baf7a", "#d9f2e8"
FROZ, FROZ_E, INP = "#e6e5e1", "#b9b8b2", "#ffffff"
CHG = (0, (4, 2))                                   # line style of the two v2 changes


# ============================================================ data
def tracker_offline():
    from scipy.optimize import linear_sum_assignment
    from acot.analysis import trk_assign as TA
    seqs = TA.episodes(["zo", "aa_legacy", "v2h"])
    out = {}
    for name, lsa in (("greedy", None), ("hungarian", linear_sum_assignment)):
        ids = [TA.replay(s, lsa) for s in seqs]
        ev = {}
        for s, i in zip(seqs, ids):
            for k, v in TA.events(s, i).items():
                ev[k] = ev.get(k, 0) + v
        out[name] = dict(TA.idf1(seqs, ids), **ev)
    return len(seqs), out


def closed_loop():
    """(progress, per-arm means on the seeds all four arms completed | None while the run is incomplete)."""
    base = os.path.join(REPO, "carla_uav_tracking", "rollout", "eval_out", "exp", "ab_where_assign")
    st = {}
    for a in ("ours", "cv", "hung", "cv_hung"):
        S = [json.load(open(f)).get("status") for f in glob.glob(os.path.join(base, a, "seeds", "*", "status.json"))]
        st[a] = (sum(s in ("complete", "skipped", "failed") for s in S), 60)
    if any(k < n for k, n in st.values()):
        return st, None
    from acot.analysis import metrics as M
    from acot.analysis.results import load
    from acot.config.resolve import resolve_experiment
    D = load(resolve_experiment(os.path.join(REPO, "acot", "configs", "experiments", "ab_where_assign.yaml")))
    ok = {a: {s: r for s, r in rows.items() if r["status"] == "complete" and r["episode"]} for a, rows in D.items()}
    seeds = sorted(set.intersection(*[set(v) for v in ok.values()]))
    res = {}
    for a, rows in ok.items():
        E = [rows[s]["episode"] for s in seeds]
        op = [M.opening_lock_ticks(rows[s]["ticks"]) for s in seeds]
        res[a] = {"n": len(seeds), "open": (sum(bool(x) for x in op), sum(x is not None for x in op)),
                  "correct": float(np.mean([M.value("correct", e) for e in E])),
                  "longC": float(np.mean([M.value("longC", e) for e in E])),
                  "longW": float(np.mean([M.value("longW", e) for e in E])),
                  "ctime": float(np.mean([(1 - e["mis_follow_sustained"]) * e["track_seconds"] for e in E]))}
    return st, res


N_EP, TRK = tracker_offline()
CL, RES = closed_loop()
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "cv_hung"
assert VARIANT in ("cv_hung", "hung"), VARIANT
CV = VARIANT == "cv_hung"
OUT = "fig_architecture_v2" if CV else "fig_architecture_v3"

# ============================================================ canvas
fig = plt.figure(figsize=(20, 14.5), facecolor=SURF)
ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, title, sub="", fc=FROZ, ec=FROZ_E, lw=1.0, fs=9.2, bold=False, ls="-", sfs=7.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=lw,
                                linestyle=ls, zorder=2))
    if sub:
        ax.text(x + w / 2, y + h - 0.9, title, ha="center", va="top", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", zorder=3)
        ax.text(x + w / 2, y + h - 1.25 - fs * 0.2, sub, ha="center", va="top", fontsize=sfs, color=INK2,
                zorder=3, linespacing=1.4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", zorder=3)


def badge(x, y, text="改动"):
    ax.text(x, y, text, fontsize=7.8, color=SURF, fontweight="bold", ha="right", va="center", zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc=INK, ec="none"))


def arrow(x1, y1, x2, y2, col=INK2, lw=1.2, ls="-", rad=0.0, label=None, lx=None, ly=None, fs=7.8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=col, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        ax.text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2, label, fontsize=fs,
                color=INK2, ha="center", va="center", zorder=4, bbox=dict(fc=SURF, ec="none", pad=0.4))


def line(xs, ys, col=INK2, lw=1.2, ls="-"):
    ax.plot(xs, ys, color=col, lw=lw, ls=ls, zorder=1)


# ---------------- title + legend
ax.text(0.8, 99.4, ("ACoT-UAV-Track 模型架构(cv_hung 组合):属性开局 + TAHRelM + PGIF + 匈牙利跟踪器 + 不可见时匀速外推"
                    if CV else "ACoT-UAV-Track 当前模型架构 v3:属性开局 + TAHRelM + PGIF + 匈牙利跟踪器")
        + "(部署设定 v2,身份决策不使用真值)", fontsize=15, fontweight="bold", color=INK, va="top")
ax.text(0.8, 97.2, ("按代码绘制(acot/runtime/,方法 ours_absent_cv + 感知消融 tracker.assign = hungarian = 实验臂 "
                    "ab_where_assign/cv_hung)。" if CV else
                    "按代码绘制(acot/runtime/,方法 ours_final + 感知消融 tracker.assign = hungarian = 实验臂 "
                    "ab_where_assign/hung,2026-10-06 起的当前模型)。") + ("S2 决策 tick 每 0.6 s 一次;DiT 每帧(0.1 s)出动作。训练过的部件只有开局属性"
                   "分类器与 TAHRelM(4,740 参数),部署时都冻结;PGIF、跟踪器、WHERE 是推理期规则。"), fontsize=9, color=INK2,
        va="top")
ax.text(0.8, 95.6, "理想化感知输入(所有对比臂相同):候选框由真值投影得到(只因射线遮挡漏检),只含目标与场景指定的干扰车;"
                   "WHERE 使用匹配框所对应车辆的仿真真值世界坐标。", fontsize=9, color=INK, va="top")
ax.legend(handles=[Patch(fc=OP_BG, ec=OP, lw=2, label="开局属性模块(训练过，冻结)"),
                   Patch(fc=TAH_BG, ec=TAH, lw=2, label="TAHRelM(训练过，冻结)"),
                   Patch(fc=PG_BG, ec=PG, lw=2, label="PGIF(推理期规则,0 参数)"),
                   Patch(fc=FROZ, ec=FROZ_E, label="冻结模块 / 规则"), Patch(fc=INP, ec=FROZ_E, label="输入 / 环境"),
                   Patch(fc=FROZ, ec=INK, lw=1.6, ls="--", label="相对 10-03 模型的改动(fig_architecture_v1)")],
          loc="upper left", bbox_to_anchor=(0.006, 0.945), ncol=6, fontsize=8.8, frameon=False, handlelength=1.6)

# ============================================================ columns
C1, C2, C3, C4 = (0.8, 12.0), (15.5, 23.0), (41.5, 23.0), (67.5, 31.7)
# ---- column 1: inputs
box(C1[0], 84.0, C1[1], 7.6, "语言指令", "去位置化，只保留身份\n“Follow the gold\nMini Cooper.”", fc=INP)
box(C1[0], 74.5, C1[1], 7.5, "高清开局帧", "同位姿、同 FOV 的 3× 相机\n仅开局 tick 0 / 6 / 12", fc=INP)
box(C1[0], 57.0, C1[1], 15.5, "候选框(仿真替身)",
    "任务车辆中在画面内的\n按真值投影成伪框\n(车长 4 m,框边 10–140 px)\n\n射线遮挡检测：被遮挡的\n车整框丢掉(目标与干扰车\n同一规则)\n\n每集固定种子打乱顺序", fc=INP)
box(C1[0], 44.0, C1[1], 11.0, "无人机 RGB", "512×512 机载相机\n每帧", fc=INP)
box(C1[0], 27.5, C1[1], 6.0, "Proprio 5D", "cam_z, pitch, uav_vel×3", fc=INP)

# ---- column 2: opening module + crop-DINOv2 + tracker + Qwen + IAR / EAR
x, w = C2
ax.add_patch(FancyBboxPatch((x, 69.5), w, 22.3, boxstyle="round,pad=0.25,rounding_size=1.0", fc=OP_BG, ec=OP, lw=2.2,
                            zorder=1.5))
ax.text(x + w / 2, 91.5, "开局身份模块 OPENING=attr(v1)", fontsize=10, fontweight="bold", color=INK, ha="center", va="top")
ax.text(x + w / 2, 89.7, "只在开局 tick 0 / 6 / 12 运行", fontsize=8, color=INK2, ha="center", va="top")
box(x + 1, 83.6, w - 2, 4.8, "① 语言解析 describe()", "→ 颜色词 c*、车型词 m*", fc="#ffffff", ec=OP, lw=1.2)
box(x + 1, 76.0, w - 2, 6.8, "② 属性分类器(DINOv2 ViT-S/14)",
    "高清帧裁剪，多尺度 + 翻转\n线性头 ⊕ 微调头，各占 0.5(17 车型 · 18 颜色)\n分数 = log p(m* | 裁剪) + log p(c* | 裁剪)",
    fc="#ffffff", ec=OP, lw=1.2, sfs=7.4)
box(x + 1, 70.3, w - 2, 4.9, "③ 窗口聚合", "每 tick 分数 - max,按跟踪器 ID 累加 → argmax\n= 开局选择", fc="#ffffff",
    ec=OP, lw=1.2, sfs=7.4)
box(x, 57.0, 10.8, 10.5, "crop-DINOv2", "ViT-S/14(冻结)\n每个候选框裁剪\n224 px,1.3× 框\n→ 384 维外观\n(L2 归一化)",
    bold=True, sfs=7.4)
box(x + 12.2, 57.0, w - 12.2, 10.5, "GT-free 跟踪器", "匈牙利匹配(numpy)\n代价 0.15·d/门限\n   + 0.85·(1 - cos)\n"
    "门限 max(150 px, 2×框)\ncos < 0.2 禁止;8 tick\n未匹配 → 删除,新 ID",
    ec=INK, lw=1.6, ls=CHG, bold=True, sfs=7.2)
badge(x + w - 0.2, 67.6)
box(x, 44.0, w, 11.0, "Qwen3-VL-4B(冻结,44.4 亿,S2)",
    "指令文本 + RGB 一次前向(language_mode = neutral)\n第 24 层隐藏态 → vlm_ctx\nStage-2 grounding 头的语言 logits\n只记日志，不参与 TAHRelM 选车",
    bold=True, sfs=7.4)
box(x, 33.5, 11.0, 8.5, "IAR(冻结,97 万)", "8 个 query\ncross-attention\n→ z_im", bold=True, sfs=7.4)
box(x + 12.0, 33.5, w - 12.0, 8.5, "EAR(冻结)", "flow-matching\n采样航点先验\n→ z_ex(默认值)", bold=True, sfs=7.4)

# ---- column 3: TAHRelM + memory + confidence + DiT
x, w = C3
ax.add_patch(FancyBboxPatch((x, 69.5), w, 22.3, boxstyle="round,pad=0.25,rounding_size=1.0", fc=TAH_BG, ec=TAH, lw=2.2,
                            zorder=1.5))
ax.text(x + w / 2, 91.5, "TAHRelM(WHICH,4,740 参数,S2)", fontsize=10, fontweight="bold", color=INK, ha="center", va="top")
box(x + 1, 82.6, w - 2, 7.0, "输入与 7 个相对特征", "每个候选：外观 + (u/W, v/H, 深度/100)\ncos(外观, 记忆)· 相对位移 · 深度 · 排名\n距离 · 软运动门",
    fc="#ffffff", ec=TAH, lw=1.2, sfs=7.4)
box(x + 1, 78.0, w - 2, 3.6, "MLP 7·64·64·1 → 每个候选的 logit", fc="#ffffff", ec=TAH, lw=1.2, fs=8.6)
box(x + 1, 70.3, w - 2, 6.8, "原始选择 $s^M$ = argmax",
    "开局 tick 0 / 6:先清空记忆 → $s^M$ = 当前累计开局选择\ntick 12 起由记忆打分(累计开局分只记日志)\n写入记忆前保存快照(供 PGIF 改写 / 撤销)",
    fc="#ffffff", ec=TAH, lw=1.2, sfs=7.3)
box(x, 59.5, w, 8.0, "身份记忆 {app, pos, vel}", "外观：可学门 EMA;速度：可学衰减 EMA\n按 $s^M$ 写入;PGIF 可改写或撤销本次写入",
    fc=TAH_BG, ec=TAH, lw=1.6, bold=True, sfs=7.6)
box(x, 52.5, w, 5.0, "置信度 c = top1 - top2 logit 差", "(TAHRelM 自带的学习置信度被覆盖，未使用)", sfs=7.4)
box(x, 27.5, w, 22.5, "HOW:DiT(冻结,32 层,1.37 亿)",
    "S1 快流，每帧(0.1 s)运行\nflow-matching 生成动作块(H = 16)\n\n条件：\nz_ex(航点先验,来自 WHERE)\nz_im · vlm_ctx · proprio\n\n→ (dx, dy, dz, dyaw) + 搜索标志\nframe_gain = 0",
    bold=True, sfs=7.8)

# ---- column 4: PGIF + commitment + WHERE + CARLA
x, w = C4
ax.add_patch(FancyBboxPatch((x, 69.5), w, 22.3, boxstyle="round,pad=0.25,rounding_size=1.0", fc=PG_BG, ec=PG, lw=2.2,
                            zorder=1.5))
ax.text(x + w / 2, 91.5, "PGIF(tempered + absent-hold,0 可训练参数,S2)", fontsize=10, fontweight="bold", color=INK,
        ha="center", va="top")
box(x + 1, 84.3, w - 2, 5.1, "① 置信度秩 → 投票权重(c 来自左侧置信度模块)", "ρ = c 在最近 30 个置信度中的分位(不足 5 个取 0.5);w = 0.5 + ρ",
    fc="#ffffff", ec=PG, lw=1.2)
box(x + 1, 78.2, w - 2, 5.1, "② 回火证据(按跟踪器 ID 记账)", "所有 ID:$L_k \\leftarrow 0.98\\,L_k$;再 $L_{s^M}$ += w",
    fc="#ffffff", ec=PG, lw=1.2)
box(x + 1, 70.3, w - 2, 6.9, "③ 决策",
    "absent-hold:最近 8 tick 内出现过的 ID 中，证据最高者本 tick 不可见且高于所有可见 ID\n"
    "→ 输出「缺席」:不选车、c = 0、撤销记忆写入、承诺不变\n"
    "否则 $\\hat{s}$ = argmax 可见 $L_k$;若 $\\hat{s} \\neq s^M$ → 从快照用 $\\hat{s}$ 重写记忆，直接承诺,c = 1",
    fc="#ffffff", ec=PG, lw=1.2, sfs=7.3)
box(x, 59.5, w, 8.0, "承诺身份(跟踪器 ID)", "只有 c ≥ 0.5 时才更新承诺;挑战者需连续 K = 5 tick 才能取代\nPGIF 改写时跳过滞回直接承诺;「缺席」时保持上一承诺",
    fc=INP, ec=PG, lw=1.6, bold=True, sfs=7.6)
box(x, 37.0, w, 20.5, "", "", ec=INK if CV else FROZ_E, lw=1.6 if CV else 1.0, ls=CHG if CV else "-")
ax.text(x + w / 2, 56.6, "WHERE(冻结规则)", fontsize=9.2, color=INK, fontweight="bold", ha="center", va="top", zorder=3)
if CV:
    badge(x + w - 0.3, 56.4)
ax.text(x + 1.2, 53.9, "承诺 ID 本 tick 有匹配框：", fontsize=8.2, color=INK, fontweight="bold", va="top")
ax.text(x + 1.2, 51.9, "取该车仿真真值世界坐标;若 c ≥ 0.5 且在画面内 → 更新匀速目标估计器\n"
                       "估计器外推出的航点投影到相机系 → 替换 z_ex", fontsize=7.6, color=INK2, va="top", linespacing=1.45)
ax.text(x + 1.2, 47.4, "承诺 ID 本 tick 没有匹配框(遮挡 / 出画 / 缺席):", fontsize=8.2, color=INK, fontweight="bold",
        va="top")
ax.text(x + 1.2, 45.4, "从最后一次确信的位置和速度匀速外推 → 替换 z_ex(control.absent_where = cv)\n"
                       "10-03 模型为：不覆盖，沿用 EAR 自己采样的航点" if CV else
                       "不覆盖 z_ex,沿用 EAR 自己采样的航点(与 10-03 模型相同)\n"
                       "匀速外推已测(cv 臂):跟随更久，但身份不变、最长锁错有变长趋势 → 未采用",
        fontsize=7.6, color=INK2, va="top", linespacing=1.45)
ax.text(x + 1.2, 40.6, "尚未有承诺 / 估计器未初始化时：沿用 EAR 航点", fontsize=7.6, color=MUTED, va="top")
box(x, 27.5, w, 7.0, "CARLA 闭环", "无人机执行动作 → 下一帧(选哪辆车决定下一帧看到什么)", fc=INP, bold=True)

# ============================================================ arrows
arrow(12.8, 86.0, 16.5, 86.0, col=OP, lw=1.6)                                    # language → ①
arrow(12.8, 78.6, 16.5, 78.6, col=OP, lw=1.6)                                    # hi-res → ②
arrow(12.8, 64.0, 15.5, 62.5)                                                    # boxes → crop-DINO
line([12.8, 14.2, 14.2], [70.0, 70.0, 73.0], col=OP, lw=1.6)                     # boxes → ② (hi-res crops)
arrow(14.2, 73.0, 16.5, 73.0, col=OP, lw=1.6)
arrow(12.8, 50.0, 15.5, 50.0)                                                    # RGB → Qwen
line([12.8, 14.6, 14.6], [53.0, 53.0, 58.5], col=INK2)                           # RGB → crop-DINO
arrow(14.6, 58.5, 15.5, 58.5)
line([12.8, 13.6, 13.6], [85.0, 85.0, 52.5], col=MUTED, lw=1.0)                 # language → Qwen
arrow(13.6, 52.5, 15.5, 52.5, col=MUTED, lw=1.0)
arrow(26.3, 62.0, 27.7, 62.0, label=None)                                        # crop-DINO → tracker
arrow(33.2, 67.5, 33.2, 70.1, col=INK, lw=1.4)                                   # tracker → ③ (keys)
ax.text(33.6, 68.8, "ID", fontsize=7.4, color=INK2, ha="left", va="center")
arrow(27.0, 83.6, 27.0, 82.8, col=OP); arrow(27.0, 76.0, 27.0, 75.2, col=OP)     # ① → ② → ③
arrow(38.5, 72.8, 42.5, 73.6, col=OP, lw=1.8, label="开局选择", lx=40.0, ly=75.0)  # opening → TAHRelM
line([26.0, 26.0], [57.0, 56.2], col=INK2)                                       # crop-DINO → TAHRelM (外观)
line([21.0, 21.0, 39.6, 39.6], [57.0, 56.2, 56.2, 84.5], col=INK2)
arrow(39.6, 84.5, 42.5, 84.5, label="外观 + 位置", lx=40.0, ly=86.6)
arrow(27.0, 44.0, 27.0, 42.2)                                                    # Qwen → IAR/EAR
arrow(38.5, 50.0, 41.5, 45.0, label="vlm_ctx", lx=40.0, ly=48.6)                 # Qwen → DiT
arrow(26.5, 36.0, 41.5, 36.0, label="z_im", lx=40.0, ly=37.2)                    # IAR → DiT (passes under EAR? routed)
# EAR → WHERE (default z_ex), routed below DiT
line([33.0, 33.0, 66.2, 66.2], [33.5, 25.6, 25.6, 38.5], col=INK2, ls=(0, (3, 2)))
arrow(66.2, 38.5, 67.5, 38.5, col=INK2, ls=(0, (3, 2)))
ax.text(50.0, 25.6, "EAR 航点(WHERE 不覆盖时的默认 z_ex)", fontsize=7.4, color=INK2, ha="center", va="center",
        bbox=dict(fc=SURF, ec="none", pad=0.3))
# tracker IDs → PGIF evidence + commitment (routed in the gap y = 68.6 above the memory row)
line([38.5, 40.7, 40.7, 66.0, 66.0], [64.0, 64.0, 68.6, 68.6, 70.3], col=INK, lw=1.2)
arrow(66.0, 70.3, 68.5, 80.5, col=INK, lw=1.2)
ax.text(52.0, 68.6, "跟踪器 ID:开局累加、PGIF 证据、承诺都按它记账", fontsize=7.4, color=INK, ha="center", va="center",
        bbox=dict(fc=SURF, ec="none", pad=0.3), zorder=4)
# column 3 ↔ 4
arrow(53.0, 67.5, 53.0, 69.3, col=TAH, lw=1.4)                                    # memory → TAHRelM
arrow(63.5, 73.5, 68.5, 73.5, col=TAH, lw=1.6, label="$s^M$", lx=66.0, ly=74.7)   # TAHRelM → PGIF
arrow(64.5, 55.0, 67.5, 61.0, col=INK2, lw=1.0, label="c", lx=66.4, ly=57.4)      # c → commitment gate
line([64.5, 65.3, 65.3], [56.5, 56.5, 52.4], col=INK2, lw=0.0)
arrow(68.5, 70.3, 64.5, 64.5, col=PG, lw=1.6, label="改写 / 撤销写入", lx=66.4, ly=66.3)  # PGIF → memory
arrow(83.35, 69.5, 83.35, 67.6, col=PG, lw=1.6)                                   # PGIF → commitment
arrow(83.35, 59.5, 83.35, 57.6)                                                   # commitment → WHERE
arrow(67.5, 42.0, 64.5, 42.0, label="z_ex", lx=66.0, ly=43.3)                     # WHERE → DiT
arrow(64.5, 31.0, 67.5, 31.0, label="动作", lx=66.0, ly=32.2)                     # DiT → CARLA
line([6.8, 6.8, 40.6, 40.6], [27.5, 26.6, 26.6, 29.0], col=MUTED, lw=1.0, ls=(0, (3, 2)))   # proprio → DiT
arrow(40.6, 29.0, 41.5, 29.0, col=MUTED, lw=1.0, ls=(0, (3, 2)))
line([83.35, 83.35, 0.35, 0.35], [27.5, 24.2, 24.2, 49.5], col=INK2, lw=1.1)       # closed loop
arrow(0.35, 49.5, 0.8, 49.5, col=INK2, lw=1.1)
ax.text(22.0, 24.2, "下一帧(闭环)", fontsize=7.4, color=INK2, ha="center", va="center",
        bbox=dict(fc=SURF, ec="none", pad=0.3))

# ============================================================ bottom panel: the two changes
ax.plot([0.8, 99.2], [22.4, 22.4], color=AXIS, lw=1.0)
ax.text(0.8, 21.6, "相对 fig_architecture_v1(10-03 最终模型)的两处改动" if CV else
        "相对 fig_architecture_v1(10-03 最终模型):采用 ①,② 已测未采用", fontsize=12, fontweight="bold", color=INK,
        va="top")

# change 1: tracker
ax.text(0.8, 18.6, "① 跟踪器匹配：贪心 → 匈牙利", fontsize=10, fontweight="bold", color=INK, va="top")
ax.text(0.8, 16.7, "旧策略镜像没有 scipy,TrackletAssigner 一直退回贪心;而跟踪器参数是在装有 scipy 的主机上(匈牙利)调出来的。\n"
                   "PGIF 证据与承诺按跟踪器 ID 记账：目标换 ID → 证据清零;ID 被干扰车接走 → 证据转给干扰车。\n"
                   f"离线回放({N_EP} 集最终模型记录;跟踪器只依赖候选框与外观，回放对固定输入是精确的):",
        fontsize=8.2, color=INK2, va="top", linespacing=1.5)
rows = [("目标 IDF1 ↑", "tgt_IDF1", "{:.3f}"), ("每集目标 ID 数 ↓", "tgt_ids", "{:.2f}"),
        ("目标 ID 变化次数 ↓", "sw_tgt", "{:d}"), ("目标与他车 ID 互换 ↓", "steal_tgt", "{:d}")]
ty = 10.4
for x, t in ((1.4, "指标"), (13.0, "贪心(v1)"), (21.0, "匈牙利(v2)")):
    ax.text(x, ty, t, fontsize=8.4, color=INK2, fontweight="bold", va="center")
ax.plot([1.2, 28.5], [ty - 0.9, ty - 0.9], color=GRID, lw=0.8)
for i, (lab, k, f) in enumerate(rows):
    y = ty - 2.2 - i * 1.9
    g, h = TRK["greedy"][k], TRK["hungarian"][k]
    ax.text(1.4, y, lab, fontsize=8.4, color=INK, va="center")
    ax.text(13.0, y, f.format(int(g)) if f == "{:d}" else f.format(g), fontsize=8.4, color=INK, va="center")
    ax.text(21.0, y, f.format(int(h)) if f == "{:d}" else f.format(h), fontsize=8.4, color=INK, va="center",
            fontweight="bold")

# change 2: WHERE
ax.text(34.0, 18.6, "② 承诺目标本 tick 无匹配框时的 WHERE:EAR 航点 → 匀速外推" + ("" if CV else "(未采用)"), fontsize=10, fontweight="bold", color=INK,
        va="top")
ax.text(34.0, 16.7, "v1 文档写的是「不可见时按匀速估计滑行」,但代码在承诺 ID 没有匹配框时不覆盖 z_ex,无人机按 EAR 自己\n"
                    "采样的航点飞(旧代码与 acot 运行时逐位一致)。v1 最终模型的身份错误 tick 中 54% 发生在目标不在候选中时，\n"
                    "遮挡期间找不回目标是最长连续正确偏低的主因。\n\n"
                    "改动:承诺 ID 无框时，用估计器从最后一次确信位置按匀速外推出航点，投影到相机系后替换 z_ex。\n"
                    "外推一直持续到承诺 ID 重新出现匹配框，或 episode 因丢失超过 18 s 结束;估计器本身没有改动。",
        fontsize=8.2, color=INK2, va="top", linespacing=1.5)

# closed-loop status
ax.text(70.0, 18.6, "闭环验证:ab_where_assign(2×2,dev2 93000–93059,同种子配对)", fontsize=10, fontweight="bold",
        color=INK, va="top")
ax.text(70.0, 16.7, "四个臂：ours(v1)· cv(仅 ②)· hung(仅 ①)· cv_hung(①+②)\n"
                    "每个主效应用 120 对估计;承诺正确的最小可检出效应约 0.07–0.10。", fontsize=8.2, color=INK2, va="top",
        linespacing=1.5)
if RES is None:
    for i, a in enumerate(("ours", "cv", "hung", "cv_hung")):
        k, n = CL[a]
        y = 11.6 - i * 1.9
        ax.text(70.6, y, a, fontsize=8.4, color=INK, va="center", fontweight="bold" if a == "cv_hung" else "normal")
        ax.add_patch(FancyBboxPatch((77.0, y - 0.45), 16.0, 0.9, boxstyle="square,pad=0", fc=GRID, ec="none", zorder=1))
        ax.add_patch(FancyBboxPatch((77.0, y - 0.45), 16.0 * k / n, 0.9, boxstyle="square,pad=0",
                                    fc=INK if a == "cv_hung" else INK2, ec="none", zorder=2))
        ax.text(93.6, y, f"{k}/{n}", fontsize=8.2, color=INK, va="center")
    ax.text(70.0, 3.2, "运行中:结论待全部 240 集完成后写入 1001_innovation-design.md;本图不报中间数字。",
            fontsize=8.2, color=INK2, va="top")
else:
    cols = [(70.4, "臂"), (78.4, "开局"), (83.0, "承诺正确↑"), (88.0, "最长正确↑"), (93.0, "最长锁错↓"), (97.6, "正确时长↑")]
    ty = 12.6
    for x, t in cols:
        ax.text(x, ty, t, fontsize=7.9, color=INK2, fontweight="bold", va="center", ha="left" if x < 72 else "center")
    ax.plot([70.2, 99.4], [ty - 0.9, ty - 0.9], color=GRID, lw=0.8)
    lab = {"ours": "ours(v1)", "cv": "cv(仅 ②)", "hung": "hung(仅 ①)" + ("" if CV else " 本图"),
           "cv_hung": "cv_hung" + (" 本图" if CV else "")}
    for i, a in enumerate(("ours", "cv", "hung", "cv_hung")):
        r, y = RES[a], ty - 2.0 - i * 1.7
        vals = [lab[a], f"{r['open'][0]}/{r['open'][1]}", f"{r['correct']:.3f}", f"{r['longC']:.1f} s",
                f"{r['longW']:.1f} s", f"{r['ctime']:.1f} s"]
        for (x, _), t in zip(cols, vals):
            ax.text(x, y, t, fontsize=7.9, color=INK, va="center", ha="left" if x < 72 else "center",
                    fontweight="bold" if a == "hung" else "normal")
    ax.text(70.0, 4.6, f"n = {RES['ours']['n']}:四个臂都完成的种子(cv_hung 有 4 个种子因 CARLA 服务器故障失败),各臂单次运行。"
                       "\n匈牙利：承诺正确主效应 +0.085(p = 0.02);外推：最长正确 +3.3 s(p = 0.002),"
                       "\n承诺正确不变、最长锁错 +3.4 s(p = 0.07)。当前模型 = hung(fig_architecture_v3),外推需改进后再用;详见 §5.5c。",
            fontsize=7.7, color=INK2, va="top", linespacing=1.45)
ax.text(0.8, 0.6, "源 acot_note/figs/fig_architecture_v2.py" + ("" if CV else " hung") + ";改动 ① 的离线数字由 acot/analysis/trk_assign.py 现算,"
                  "最终模型记录 = runs/asd/iter/{zo*, aa_legacy*, v2h*}.trk.jsonl。",
        fontsize=7.6, color=MUTED, va="bottom")

for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(HERE, f"{OUT}.{ext}"), dpi=150 if ext == "png" else None, facecolor=SURF)
print(f"saved {OUT}.{{png,pdf,svg}}")
