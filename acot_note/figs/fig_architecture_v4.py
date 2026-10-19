"""ACoT-UAV-Track 当前模型架构 v4(2026-10-12):属性开局 v1 + TAHRelM + PGIF(tempered + absent-hold)+ 匈牙利跟踪器,
conf_tau 0.05(= 当前 Ours:opt1_screen_devB3/tau005 + opt2a_screen_devB3/tau005_r1,协议 sim_proxy_v2@3)。
布局沿用 fig_architecture_v2.py hung(= v3 图);按 10-12 代码核实(Claude + codex 只读审计 runs/arch_check/codex_arch_r1.md)
更正 v3 图的出入,更正处用墨色虚线边框 + 「v4 更正」标签:
  ① 承诺门限 0.5 → 0.05;② Stage-2 模块(EAR / IAR / DiT / grounding 头)是离线训练过的,不是只有开局与 TAHRelM;
  ③ EAR 无 Stage-1 预热(train/train_v5_e1.sh),每个 S2 tick 采样,无 WHERE 覆盖时驱动 DiT;
  ④ DiT 训练时 z_ex = 真值航点(train/stage2.py:183),从未见过 EAR 输出;⑤ grounding 头在 TAH 记忆为空时给初选。
底部 EAR / WHERE 占比由闭环 ticks 日志现算(承诺 ID 不在本 tick 的跟踪器 ID 中 = WHERE 不覆盖)。
  python acot_note/figs/fig_architecture_v4.py   → fig_architecture_v4.{png,pdf,svg}"""
import glob
import json
import os
import sys

from matplotlib.patches import FancyBboxPatch, Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _archlib import (CHG, FROZ, FROZ_E, GRID, INK, INK2, INP, MUTED, OP, OP_BG, PG, PG_BG, REPO, SURF, TAH,  # noqa: E402
                      TAH_BG, Canvas)

RUNS = (("opt1_screen_devB3", "tau005"), ("opt2a_screen_devB3", "tau005_r1"))


# ============================================================ data
def ear_share():
    """S2 ticks of the current Ours runs: where WHERE leaves the EAR sample in z_ex (no commitment yet, or the committed
    tracker ID has no box this tick) — a lower bound (the estimator may also be uninitialised)."""
    n = none_c = miss = eps = 0
    for exp, arm in RUNS:
        for f in glob.glob(os.path.join(REPO, "carla_uav_tracking", "rollout", "eval_out", "exp", exp, arm, "seeds", "*",
                                        "ticks.jsonl")):
            eps += 1
            for line_ in open(f):
                r = json.loads(line_)
                n += 1
                c = r.get("committed")
                if c is None:
                    none_c += 1
                elif c not in r.get("keys", []):
                    miss += 1
    return {"eps": eps, "ticks": n, "none": none_c / n, "miss": miss / n, "ear": (none_c + miss) / n}


S = ear_share()
EAR_P, WHERE_P = round(100 * S["ear"]), round(100 * (1 - S["ear"]))

# ============================================================ canvas
C = Canvas(20, 14.5)
ax = C.ax


def box(*a, **k):
    C.box(*a, **k)


def fix(x, y, w, h):
    """dashed ink frame + badge = corrected relative to the v3 figure"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc="none", ec=INK, lw=1.6,
                                linestyle=CHG, zorder=2.5))
    C.badge(x + w - 0.2, y + h + 0.1, "v4 更正")


arrow, line = C.arrow, C.line

# ---------------- title + legend
C.text(0.8, 99.4, "ACoT-UAV-Track 当前模型架构 v4:属性开局 + TAHRelM + PGIF + 匈牙利跟踪器(部署设定 v2,身份决策不使用真值)",
       fs=15, col=INK, bold=True)
C.text(0.8, 97.2, "按代码绘制并经 2026-10-12 代码核实(acot/runtime/;方法 ours_final + identity.conf_tau = 0.05,协议 sim_proxy_v2@3 "
                  "= 当前 Ours,即 opt1_screen_devB3/tau005)。S2 决策 tick 每 0.6 s 一次;DiT 每帧(0.1 s)出动作。\n"
                  "离线训练过的部件:Stage-2 模块(EAR / IAR / DiT / grounding 头,runs/stage2_v5,08-17)、开局属性分类器、TAHRelM;"
                  "部署时全部冻结。PGIF、跟踪器、WHERE 是推理期规则。", fs=9, col=INK2, ls=1.5)
C.text(0.8, 94.4, "理想化感知输入(所有对比臂相同):候选框由真值投影得到(只因射线遮挡漏检),只含目标与场景指定的干扰车;"
                  "WHERE 使用匹配框所对应车辆的仿真真值世界坐标。", fs=9, col=INK)
ax.legend(handles=[Patch(fc=OP_BG, ec=OP, lw=2, label="开局属性模块(训练过,冻结)"),
                   Patch(fc=TAH_BG, ec=TAH, lw=2, label="TAHRelM(训练过,冻结)"),
                   Patch(fc=PG_BG, ec=PG, lw=2, label="PGIF(推理期规则,0 参数)"),
                   Patch(fc=FROZ, ec=FROZ_E, label="Stage-2 模块(训练过,冻结)/ 冻结骨干 / 规则"),
                   Patch(fc=INP, ec=FROZ_E, label="输入 / 环境"),
                   Patch(fc="none", ec=INK, lw=1.6, ls="--", label="v4 更正(相对 fig_architecture_v3)")],
          loc="upper left", bbox_to_anchor=(0.006, 0.935), ncol=6, fontsize=8.8, frameon=False, handlelength=1.6)

# ============================================================ columns
C1, C2, C3, C4 = (0.8, 12.0), (15.5, 23.0), (41.5, 23.0), (67.5, 31.7)
TOP = 90.0                       # top of the three group panels (v3: 91.8; title block is one line taller)
# ---- column 1: inputs
box(C1[0], 82.6, C1[1], 7.4, "语言指令", "去位置化,只保留身份\n“Follow the gold\nMini Cooper.”", fc=INP)
box(C1[0], 73.5, C1[1], 7.3, "高清开局帧", "同位姿、同 FOV 的 3× 相机\n仅开局 tick 0 / 6 / 12", fc=INP)
box(C1[0], 56.5, C1[1], 15.2, "候选框(仿真替身)",
    "任务车辆中在画面内的\n按真值投影成伪框\n(车长 4 m,框边 10–140 px)\n\n射线遮挡检测:被遮挡的\n车整框丢掉(目标与干扰车\n同一规则)\n\n每集固定种子打乱顺序", fc=INP)
box(C1[0], 44.0, C1[1], 10.8, "无人机 RGB", "512×512 机载相机\n每帧", fc=INP)
box(C1[0], 27.5, C1[1], 6.0, "Proprio 5D", "cam_z, pitch, uav_vel×3", fc=INP)

# ---- column 2: opening module + crop-DINOv2 + tracker + Qwen + IAR / EAR
x, w = C2
C.group(x, 69.0, w, TOP - 69.0, "开局身份模块 OPENING=attr(v1)", OP_BG, OP, sub="只在开局 tick 0 / 6 / 12 运行")
box(x + 1, 81.6, w - 2, 4.2, "① 语言解析 describe()", "→ 颜色词 c*、车型词 m*", fc="#ffffff", ec=OP, lw=1.2)
box(x + 1, 75.0, w - 2, 5.9, "② 属性分类器(DINOv2 ViT-S/14)",
    "高清帧裁剪,多尺度 + 翻转\n线性头 ⊕ 微调头,各占 0.5(17 车型 · 18 颜色)\n分数 = log p(m* | 裁剪) + log p(c* | 裁剪)",
    fc="#ffffff", ec=OP, lw=1.2, sfs=7.4)
box(x + 1, 69.7, w - 2, 4.8, "③ 窗口聚合", "每 tick 分数 - max,按跟踪器 ID 累加 → argmax\n= 开局选择", fc="#ffffff",
    ec=OP, lw=1.2, sfs=7.4)
box(x, 56.5, 10.8, 10.5, "crop-DINOv2", "ViT-S/14(冻结)\n每个候选框裁剪\n224 px,1.3× 框\n→ 384 维外观\n(L2 归一化)",
    bold=True, sfs=7.4)
box(x + 12.2, 56.5, w - 12.2, 10.5, "GT-free 跟踪器", "匈牙利匹配(numpy)\n代价 0.15·d/门限\n   + 0.85·(1 - cos)\n"
    "门限 max(150 px, 2×框)\ncos < 0.2 禁止;8 tick\n未匹配 → 删除,新 ID", bold=True, sfs=7.2)
box(x, 43.5, w, 11.3, "Qwen3-VL-4B(冻结,44.4 亿,S2)",
    "指令 + RGB 一次前向(language_mode = neutral)\n只取第 24 层隐藏态(L = 1)→ vlm_ctx\n"
    "Stage-2 grounding 头:TAH 记忆为空时给初选\n(开局 tick 被属性开局覆盖,自然开局下基本不触发)", bold=True, sfs=7.2)
fix(x, 43.5, w, 11.3)
box(x, 29.0, 11.0, 12.6, "IAR(97 万)", "Stage 2 训练,冻结\n8 个 query\ncross-attention\n(1 层 vlm_ctx)\n→ z_im",
    bold=True, sfs=7.3)
box(x + 12.0, 29.0, w - 12.0, 12.6, "EAR", "Stage 2 从头训练,冻结\n(无 Stage-1 预热)\n每个 S2 tick\nflow-matching 采样\n→ z_ex(默认值)",
    bold=True, sfs=7.3)
fix(x + 12.0, 29.0, w - 12.0, 12.6)

# ---- column 3: TAHRelM + memory + confidence + DiT
x, w = C3
C.group(x, 69.0, w, TOP - 69.0, "TAHRelM(WHICH,4,740 参数,S2)", TAH_BG, TAH)
box(x + 1, 81.4, w - 2, 6.8, "输入与 7 个相对特征", "每个候选:外观 + (u/W, v/H, 深度/100)\ncos(外观, 记忆)· 相对位移 · 深度 · 排名\n距离 · 软运动门",
    fc="#ffffff", ec=TAH, lw=1.2, sfs=7.4)
box(x + 1, 77.0, w - 2, 3.5, "MLP 7·64·64·1 → 每个候选的 logit", fc="#ffffff", ec=TAH, lw=1.2, fs=8.6)
box(x + 1, 69.7, w - 2, 6.6, "原始选择 $s^M$ = argmax",
    "开局 tick 0 / 6:先清空记忆 → $s^M$ = 当前累计开局选择\ntick 12 起由记忆打分(累计开局分只记日志)\n写入记忆前保存快照(供 PGIF 改写 / 撤销)",
    fc="#ffffff", ec=TAH, lw=1.2, sfs=7.3)
box(x, 59.0, w, 8.0, "身份记忆 {app, pos, vel}", "外观:可学门 EMA;速度:可学衰减 EMA\n按 $s^M$ 写入;PGIF 可改写或撤销本次写入",
    fc=TAH_BG, ec=TAH, lw=1.6, bold=True, sfs=7.6)
box(x, 52.0, w, 5.0, "置信度 c = top1 - top2 logit 差", "(TAHRelM 自带的学习置信度被覆盖,未使用)", sfs=7.4)
box(x, 21.5, w, 28.5, "HOW:DiT(32 层,1.37 亿)",
    "Stage 2 训练,冻结;S1 快流,每帧(0.1 s)运行\nflow-matching 生成动作块(H = 16)\n\n"
    "条件:z_ex · z_im · vlm_ctx · proprio\n\n"
    f"部署时的 z_ex(当前 Ours,S2 tick 占比):\n· WHERE 匀速航点 ≤ {WHERE_P}%(承诺 ID 本 tick 有框)\n"
    f"· EAR 采样航点 ≥ {EAR_P}%(无承诺 / 承诺 ID 无框)\n\n"
    "训练时的 z_ex:真值航点(30% 概率加噪),\n从未见过 EAR 或 WHERE 的输出\n(train/stage2.py:183;离线验证同样用真值)\n\n"
    "→ (dx, dy, dz, dyaw) + 搜索标志;frame_gain = 0", bold=True, sfs=7.4)
fix(x, 21.5, w, 28.5)

# ---- column 4: PGIF + commitment + WHERE + CARLA
x, w = C4
C.group(x, 69.0, w, TOP - 69.0, "PGIF(tempered + absent-hold,0 可训练参数,S2)", PG_BG, PG)
box(x + 1, 82.9, w - 2, 5.0, "① 置信度秩 → 投票权重(c 来自左侧置信度模块)", "ρ = c 在最近 30 个置信度中的分位(不足 5 个取 0.5);w = 0.5 + ρ",
    fc="#ffffff", ec=PG, lw=1.2)
box(x + 1, 77.1, w - 2, 5.0, "② 回火证据(按跟踪器 ID 记账)", "所有 ID:$L_k \\leftarrow 0.98\\,L_k$;再 $L_{s^M}$ += w",
    fc="#ffffff", ec=PG, lw=1.2)
box(x + 1, 69.7, w - 2, 6.7, "③ 决策",
    "absent-hold:最近 8 tick 内出现过的 ID 中,证据最高者本 tick 不可见且高于所有可见 ID\n"
    "→ 输出「缺席」:不选车、c = 0、撤销记忆写入、承诺不变\n"
    "否则 $\\hat{s}$ = argmax 可见 $L_k$;若 $\\hat{s} \\neq s^M$ → 从快照用 $\\hat{s}$ 重写记忆,直接承诺,c = 1",
    fc="#ffffff", ec=PG, lw=1.2, sfs=7.3)
box(x, 59.0, w, 8.0, "承诺身份(跟踪器 ID)", "只有 c ≥ conf_tau = 0.05 时才更新承诺;挑战者需连续 K = 5 tick 才能取代\n"
    "PGIF 改写时跳过滞回直接承诺;「缺席」时保持上一承诺", fc=INP, ec=PG, lw=1.6, bold=True, sfs=7.6)
fix(x, 59.0, w, 8.0)
box(x, 33.5, w, 23.5, "", "")
C.text(x + w / 2, 56.1, "WHERE(冻结规则)", fs=9.2, col=INK, bold=True, ha="center")
C.text(x + 1.2, 53.6, f"承诺 ID 本 tick 有匹配框(≤ {WHERE_P}% 的 S2 tick):", fs=8.2, col=INK, bold=True)
C.text(x + 1.2, 51.6, "取该车仿真真值世界坐标;若 c ≥ 0.05 且在画面内 → 更新匀速目标估计器\n"
                      "估计器外推出的航点投影到相机系 → 替换 z_ex", fs=7.6)
C.text(x + 1.2, 47.0, f"承诺 ID 本 tick 没有匹配框(遮挡 / 出画 / 换 ID,≈ {round(100 * S['miss'])}%):", fs=8.2, col=INK,
       bold=True)
C.text(x + 1.2, 45.0, "不覆盖 z_ex,沿用 EAR 本 tick 采样的航点(control.absent_where = ear,缺省)\n"
                      "匀速外推已测(ab_where_assign cv 臂):跟随更久,但身份不变、最长锁错有变长趋势 → 未采用", fs=7.6)
C.text(x + 1.2, 40.2, f"尚未有承诺 / 估计器未初始化(≈ {round(100 * S['none'])}%):沿用 EAR 航点", fs=7.6, col=MUTED)
C.text(x + 1.2, 37.6, "注:因此 EAR 不是摆设——它是 DiT 在多数 S2 tick 的航点来源,\n而 DiT 训练时只见过真值航点(训练-部署错位)",
       fs=7.6, col=INK)
fix(x, 33.5, w, 23.5)
box(x, 23.5, w, 7.0, "CARLA 闭环", "无人机执行动作 → 下一帧(选哪辆车决定下一帧看到什么)", fc=INP, bold=True)

# ============================================================ arrows (v3 routing, shifted to the new rows)
arrow(12.8, 84.6, 16.5, 84.6, col=OP, lw=1.6)                                    # language → ①
arrow(12.8, 77.6, 16.5, 77.6, col=OP, lw=1.6)                                    # hi-res → ②
arrow(12.8, 63.5, 15.5, 62.0)                                                    # boxes → crop-DINO
line([12.8, 14.2, 14.2], [69.5, 69.5, 72.2], col=OP, lw=1.6)                     # boxes → ② (hi-res crops)
arrow(14.2, 72.2, 16.5, 72.2, col=OP, lw=1.6)
arrow(12.8, 49.5, 15.5, 49.5)                                                    # RGB → Qwen
line([12.8, 14.6, 14.6], [52.5, 52.5, 58.0], col=INK2)                           # RGB → crop-DINO
arrow(14.6, 58.0, 15.5, 58.0)
line([12.8, 13.6, 13.6], [83.6, 83.6, 52.0], col=MUTED, lw=1.0)                 # language → Qwen
arrow(13.6, 52.0, 15.5, 52.0, col=MUTED, lw=1.0)
arrow(26.3, 61.5, 27.7, 61.5)                                                    # crop-DINO → tracker
arrow(33.2, 67.0, 33.2, 69.5, col=INK, lw=1.4)                                   # tracker → ③ (keys)
C.text(33.6, 68.6, "ID", fs=7.4, va="center")
arrow(27.0, 81.6, 27.0, 81.15, col=OP); arrow(27.0, 75.0, 27.0, 74.75, col=OP)     # ① → ② → ③
arrow(38.5, 72.0, 42.5, 72.8, col=OP, lw=1.8, label="开局选择", lx=40.0, ly=74.2)  # opening → TAHRelM
line([26.0, 26.0], [56.5, 55.9], col=INK2)                                       # crop-DINO → TAHRelM (外观)
line([21.0, 21.0, 39.6, 39.6], [56.5, 55.9, 55.9, 83.4], col=INK2)
arrow(39.6, 83.4, 42.5, 83.4, label="外观 + 位置", lx=40.0, ly=85.4)
arrow(27.0, 43.5, 27.0, 41.8)                                                    # Qwen → IAR/EAR
arrow(38.5, 49.0, 41.5, 45.0, label="vlm_ctx", lx=40.0, ly=48.0)                 # Qwen → DiT
arrow(26.5, 33.0, 41.5, 33.0, label="z_im", lx=40.0, ly=34.2)                    # IAR → DiT
line([33.0, 33.0, 66.2, 66.2], [29.0, 19.6, 19.6, 35.0], col=INK2, ls=(0, (3, 2)))  # EAR → WHERE (default z_ex)
arrow(66.2, 35.0, 67.5, 35.0, col=INK2, ls=(0, (3, 2)))
C.text(50.0, 19.6, f"EAR 航点(WHERE 不覆盖时的 z_ex,≥ {EAR_P}% 的 S2 tick)", fs=7.4, ha="center", va="center",
       bbox=dict(fc=SURF, ec="none", pad=0.3))
line([38.5, 40.7, 40.7, 66.0, 66.0], [63.5, 63.5, 68.1, 68.1, 69.7], col=INK, lw=1.2)   # tracker IDs → PGIF / commit
arrow(66.0, 69.7, 68.5, 79.5, col=INK, lw=1.2)
C.text(52.0, 68.1, "跟踪器 ID:开局累加、PGIF 证据、承诺都按它记账", fs=7.4, col=INK, ha="center", va="center",
       bbox=dict(fc=SURF, ec="none", pad=0.3), zorder=4)
arrow(53.0, 67.0, 53.0, 68.8, col=TAH, lw=1.4)                                    # memory → TAHRelM
arrow(63.5, 72.8, 68.5, 72.8, col=TAH, lw=1.6, label="$s^M$", lx=66.0, ly=74.0)   # TAHRelM → PGIF
arrow(64.5, 54.5, 67.5, 60.5, lw=1.0, label="c", lx=66.4, ly=56.9)                # c → commitment gate
arrow(68.5, 69.7, 64.5, 64.0, col=PG, lw=1.6, label="改写 / 撤销写入", lx=66.4, ly=65.8)  # PGIF → memory
arrow(83.35, 69.0, 83.35, 67.1, col=PG, lw=1.6)                                   # PGIF → commitment
arrow(83.35, 59.0, 83.35, 57.1)                                                   # commitment → WHERE
arrow(67.5, 42.0, 64.5, 42.0, label="z_ex", lx=66.0, ly=43.3)                     # WHERE → DiT
arrow(64.5, 27.0, 67.5, 27.0, label="动作", lx=66.0, ly=28.2)                     # DiT → CARLA
line([6.8, 6.8, 40.6, 40.6], [27.5, 26.6, 26.6, 25.0], col=MUTED, lw=1.0, ls=(0, (3, 2)))   # proprio → DiT
arrow(40.6, 25.0, 41.5, 25.0, col=MUTED, lw=1.0, ls=(0, (3, 2)))
line([83.35, 83.35, 0.35, 0.35], [23.5, 18.2, 18.2, 49.0], lw=1.1)               # closed loop
arrow(0.35, 49.0, 0.8, 49.0, lw=1.1)
C.text(22.0, 18.2, "下一帧(闭环)", fs=7.4, ha="center", va="center", bbox=dict(fc=SURF, ec="none", pad=0.3))

# ============================================================ bottom panel
ax.plot([0.8, 99.2], [16.6, 16.6], color="#c3c2b7", lw=1.0)
C.text(0.8, 15.9, "相对 fig_architecture_v3 的更正(2026-10-12 代码核实:Claude + codex 只读审计,结论「图与代码不符」)",
       fs=11.5, col=INK, bold=True)
C.text(0.8, 13.6,
       "① 承诺门限:v3 写 c ≥ 0.5;当前 Ours 为 conf_tau = 0.05(10-08 调参后,resolved.json)。\n"
       "② 训练过的部件:v3 写「只有开局属性分类器与 TAHRelM」;EAR / IAR / DiT / grounding 头都在 Stage 2 离线训练过\n"
       "    (runs/stage2_v5/stage2_best.pt,08-17,第 20 轮,按 grounding 头 mis_follow 选),部署时冻结。\n"
       "③ EAR:无 Stage-1 预热(train/train_v5_e1.sh:5;runs/train_v5.log 无 warm-start,EAR 损失从 25 起从头训),\n"
       f"    但每个 S2 tick 都采样,且在 ≥ {EAR_P}% 的 S2 tick 里就是 DiT 的 z_ex(acot/runtime/policy.py:150-155, 386-405)。\n"
       "④ DiT 训练时 z_ex = 真值航点(30% 加噪,train/stage2.py:183),离线验证 act_mse 也用真值(:229)→ EAR → DiT 的耦合\n"
       "    只在闭环部署时存在,离线测试测不到。\n"
       "⑤ grounding 头不只记日志:TAH 记忆为空时其 argmax 作初选(policy.py:248);自然开局下被属性开局覆盖,基本不触发。",
       fs=7.9, ls=1.45)

C.text(57.0, 15.9, "z_ex 的来源:训练 vs 部署", fs=10, col=INK, bold=True)
C.table(57.2, 13.4, [(57.4, "环节"), (68.0, "EAR"), (80.5, "DiT 的条件 z_ex")],
        [["Stage-1 预热", "跳过(stage1_ear_v5 是 08-18 为别的模型生成)", "—"],
         ["Stage-2 训练", "从头训练,L_ear 权重 0.3", "真值航点,30% 概率加噪"],
         ["Stage-2 离线验证", "—", "真值航点"],
         ["闭环部署(当前 Ours)", "每 S2 tick 采样", f"WHERE ≤ {WHERE_P}% / EAR ≥ {EAR_P}%"]],
        fs=7.8, dy=1.85, hl={3}, hl_fc="#f1f0ec", x1=99.2)
C.text(57.0, 3.6, f"占比来源:当前 Ours 两次运行(opt1_screen_devB3/tau005 + opt2a_screen_devB3/tau005_r1,dev-B,{S['eps']} 集,"
                  f"{S['ticks']} 个 S2 tick);\n「EAR」= 无承诺({100 * S['none']:.1f}%)+ 承诺 ID 本 tick 无框({100 * S['miss']:.1f}%),"
                  "为下界(估计器未初始化时也沿用 EAR)。", fs=7.4, col=MUTED)
C.text(0.8, 0.6, "源 acot_note/figs/fig_architecture_v4.py(布局沿用 fig_architecture_v2.py hung);核实记录 runs/arch_check/codex_arch_r1.md、"
                 "memory acot-uav-arch-facts。", fs=7.4, col=MUTED, va="bottom")

C.save("fig_architecture_v4")
