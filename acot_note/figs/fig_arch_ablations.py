"""消融方法的实现架构(2026-10-07 按协议 sim_proxy_v2@3 与 1006 设计 v4 更新)。共享部分与两种开局设定的定义见
fig_arch_overview;本图只画 WHICH 槽位内的身份链(开局 → TAHRelM → PGIF → 承诺),顶部为我们的方法作参照，下面每个消融一格，
与参照不同的模块用墨色虚线框标出;每格标出所在表与开局设定:
  自然开局(绿)        部署设定 —— 每个臂自己跑属性开局 + 统一交接(表 1 族 A、表 1 灰色诊断行)
  强制共享开局(紫)    机制设定 —— 开局窗口内身份 / 承诺 / WHERE 强制为参考 actor(表 2 H-P1–P5 与探索臂)
按代码绘制:acot/runtime/policy.py(_identity / _opening / _filter)、acot/runtime/pgif.py;配置 acot/configs/methods/*.yaml。
证据：T0 pilot(协议 @2,dev-B,r = 2;1007_closed_loop_summary.md);其余为旧协议证据，只说明方向。
  python acot_note/figs/fig_arch_ablations.py → fig_arch_ablations.{png,pdf,svg}"""
from _archlib import (CHG, INK, MUTED, OP, OP_BG, PG, PG_BG, SH, SH_BG, TAH, TAH_BG, Canvas)
from matplotlib.patches import FancyBboxPatch

C = Canvas(20, 15)
ax = C.ax
C.text(0.8, 99.3, "消融方法的实现架构:与我们的方法只差身份链中的一个模块;每个消融标出所在表与开局设定", fs=15, col=INK,
       bold=True)
C.text(0.8, 97.2, "共享部分与两种开局设定的定义见 fig_arch_overview。墨色虚线框 = 该消融改动的模块。"
                  "绿色标签 = 自然开局(部署设定，表 1);紫色标签 = 强制共享开局(机制设定，表 2)。协议 sim_proxy_v2@3。", fs=9)

KIND = {"op": (OP_BG, OP), "sh": (SH_BG, SH), "both": ("#ffffff", OP), "tah": (TAH_BG, TAH), "pg": (PG_BG, PG),
        "cm": ("#ffffff", "#b9b8b2")}
NAT, SHR = OP, SH


def chain(x0, y0, w, h, blocks, gap=1.6, arrows=True):
    """blocks = [(title, sub, kind, changed)] laid out left → right with arrows."""
    n = len(blocks)
    bw = (w - gap * (n - 1)) / n
    for i, (t, s, k, chg) in enumerate(blocks):
        x = x0 + i * (bw + gap)
        fc, ec = KIND[k]
        if chg:
            C.box(x, y0, bw, h, t, s, fc="#ffffff", ec=INK, lw=1.8, ls=CHG, bold=True, sfs=7.1, fs=8.5)
            C.badge(x + bw - 0.2, y0 + h + 0.15, "改动")
        else:
            C.box(x, y0, bw, h, t, s, fc=fc, ec=ec, lw=1.4 if k == "both" else 1.2, bold=True, sfs=7.1, fs=8.5)
        if arrows and i < n - 1:
            C.arrow(x + bw, y0 + h / 2, x + bw + gap, y0 + h / 2)


def tags(x, y, items):
    """items = [(text, colour)] left → right (width estimated from the character count)."""
    for t, col in items:
        C.tag(x, y, t, fc=col, fs=7.8)
        x += sum(1.05 if ord(ch) > 0x2E80 else 0.55 for ch in t) * 0.62 + 1.6


# ---------------------------------------------------------------- reference: ours
ax.add_patch(FancyBboxPatch((0.6, 79.4), 98.8, 15.6, boxstyle="round,pad=0.2,rounding_size=0.8", fc="#f3f2ee",
                            ec="none", zorder=0.5))
C.text(1.2, 94.4, "参照：我们的方法 ours = ours_final(TAHRelM + PGIF + 属性开局，现有权重 runs/tah_relm_dag.pt)", fs=10.5,
       col=INK, bold=True)
chain(1.2, 80.2, 97.6, 12.0, [
    ("开局(两种设定)", "自然：属性开局 3 tick 累加,\n第 0、6 步统一交接\n共享：参考 actor 强制\n(第 0–17 步,c = 1)", "both", False),
    ("TAHRelM 记忆", "7 个相对特征 → MLP\n开局 tick 清空记忆、写入开局选择;\n之后自己打分\n按自身选择写入(留快照)", "tah", False),
    ("PGIF ①②", "秩权重 w = 0.5 + ρ\n回火证据 L ← 0.98 L,\nL[选择] += w\n(按跟踪器 ID 记账)", "pg", False),
    ("PGIF ③", "缺席保持：证据最高的 ID 近 8 tick\n出现过但本 tick 不可见 → 「缺席」\n(不选车、撤销写入)\n否则 argmax 可见证据;不同则门控", "pg", False),
    ("承诺", "c = top1 - top2 差\nc ≥ 0.5 才更新;K = 5\nPGIF 改写直接承诺(c = 1)\n缺席保持", "cm", False),
])

# ---------------------------------------------------------------- ablation panels
SAME = "同参照"
OPN = ("属性开局", "自然开局\n(同参照)", "op", False)
OPS = ("共享开局", "参考 actor 强制\n(第 0–17 步)", "sh", False)
P = [
    ("w/o PGIF(filter.mode = off)", [("自然 · 表 1 族 A(H-A1)", NAT), ("共享 · 表 2 H-P1", SHR)],
     "PGIF 整体的贡献(命题 1:确认偏差)",
     [("开局(两种)", "自然 / 共享\n(同参照)", "both", False), ("TAHRelM 记忆", "写入不再被\n门控 / 撤销", "tah", False),
      ("PGIF 移除", "TAHRelM 选择直通\n无证据、无缺席、\n无门控写入", "pg", True), ("承诺", "c = TAHRelM\nlogit 差", "cm", False)],
     "T0 pilot(@2,dev-B,r = 2;w/o PGIF − Ours):自然开局 −0.132 [−0.206, −0.058];\n"
     "共享开局 +0.028 [−0.044, +0.102] —— 增益集中在开局交接(承诺更早、无承诺帧少 8 个百分点)。"),
    ("w/o 缺席保持(absent_hold = false)", [("共享 · 表 2 H-P2", SHR)], "目标不可见时不选车的作用",
     [OPS, ("TAHRelM 记忆", SAME, "tah", False), ("PGIF ①②", SAME, "pg", False),
      ("PGIF ③ 无缺席", "只在可见 ID 中\n取 argmax;目标被\n遮挡时改选可见车", "pg", True), ("承诺", SAME, "cm", False)],
     "旧协议证据(仅方向):S3 → S4(v2,贪心，语言开局)+0.060 [−0.064, +0.201]。"),
    ("w/o 回火(resid_decay = 1.0)", [("共享 · 表 2 H-P3", SHR)], "回火的必要性(命题 2a,单变量)",
     [OPS, ("TAHRelM 记忆", SAME, "tah", False),
      ("PGIF ② 不衰减", "λ = 1:L[选择] += w\n证据只累加、不遗忘\n(权重、缺席、门控\n全部不变)", "pg", True),
      ("PGIF ③", SAME, "pg", False), ("承诺", SAME, "cm", False)],
     "新的单变量臂;旧的贝叶斯变体改变了整个证据形式，改为探索对照。"),
    ("w/o 置信加权(filter_kind = resid)", [("共享 · 表 2 H-P4", SHR)], "置信加权的必要性(命题 2b)",
     [OPS, ("TAHRelM 记忆", SAME, "tah", False),
      ("PGIF ① 等权", "每票权重 w = 1\n(不用置信度秩;\n衰减、缺席、门控\n不变)", "pg", True),
      ("PGIF ③", SAME, "pg", False), ("承诺", SAME, "cm", False)],
     "旧协议证据(仅方向):v1 口径等权驻留 0.376 vs 0.613(1001 §5.2)。"),
    ("w/o 记忆门控(gate_memory = false)", [("共享 · 表 2 H-P5", SHR)], "门控必须作用于记忆",
     [OPS, ("TAHRelM 记忆", "保留自身选择的写入\nPGIF 不改写 / 撤销", "tah", True),
      ("PGIF ①②", SAME, "pg", False), ("PGIF ③", "仍改写选择与承诺\n(只是不回写记忆)", "pg", False),
      ("承诺", SAME, "cm", False)],
     "旧协议证据(仅方向):v1 口径去门控 0.396 vs 0.613(1001 §5.2)。"),
    ("w/o 属性开局(开局 = grounding 头,window = 1)", [("自然 · 表 1 族 A(H-A2)", NAT)], "属性开局的贡献(开局是第一瓶颈)",
     [("grounding 头开局", "第 0 步:Stage-2\n语言 grounding 头\nargmax(512 px)", "op", True),
      ("TAHRelM 记忆", "冷启动采用该选择", "tah", False), ("PGIF ①②", SAME, "pg", False),
      ("PGIF ③", SAME, "pg", False), ("承诺", SAME, "cm", False)],
     "T0 pilot(@2,dev-B,r = 2):承诺正确 −0.247 [−0.362, −0.129];开局锁对 0.94 → 0.33。\n"
     "只在自然开局下比较(强制共享开局会覆盖开局本身)。"),
    ("探索臂(不进 Holm 族)", [("共享 · 表 2 探索", SHR)], "贝叶斯变体 / w/o 承诺门控 / w/o 强制置信",
     [("贝叶斯变体", "filter_kind = bayes\n粘滞 HMM 后验\n(该分支没有缺席输出,\nabsent_hold 不生效)", "pg", True),
      ("w/o 承诺门控", "gate_commit = false\n记忆按 PGIF 更新,\n承诺跟随记忆原选\n(开关已实现)", "pg", True),
      ("w/o 强制置信", "force_conf = false\n改写时 c = PGIF\n证据 top1 - top2,\n走正常承诺规则(D5)", "pg", True)],
     "只报效应量与 CI;三者开局均为强制共享开局，其余同参照。贝叶斯变体与 w/o 缺席保持对照(都没有缺席输出)。"),
    ("U3 真值「目标不在」(诊断上界)", [("自然 · 表 1 灰色行", NAT)], "完美的「目标不在候选中」检测能带来多少",
     [OPN, ("TAHRelM 记忆", "真值判目标不在时\n撤销本 tick 写入", "tah", True),
      ("PGIF ①②", "真值判目标不在时\n回滚本 tick 证据", "pg", True),
      ("真值缺席", "目标不在候选中\n→ 不选车、c = 0、\n承诺保持", "pg", True), ("承诺", SAME, "cm", False)],
     "只作诊断(使用真值),不参与比较。旧协议证据(v2,10-03):+0.115 [−0.010, +0.248]。"),
]
H = 24.0
cells = [(0.8, 54.0), (34.0, 54.0), (67.2, 54.0), (0.8, 28.8), (34.0, 28.8), (67.2, 28.8), (0.8, 3.6), (34.0, 3.6)]
for (title, tg, q, blocks, ev), (x, y) in zip(P, cells):
    ax.add_patch(FancyBboxPatch((x, y), 32.0, H, boxstyle="round,pad=0.2,rounding_size=0.8", fc="#fcfcfb", ec=MUTED,
                                lw=0.8, zorder=0.4))
    C.text(x + 0.7, y + H - 0.4, title, fs=9.2, col=INK, bold=True)
    tags(x + 0.7, y + H - 2.9, tg)
    C.text(x + 0.7, y + H - 4.1, "回答的问题:" + q, fs=7.8)
    chain(x + 0.7, y + 5.4, 30.6, 12.4, blocks, gap=0.9, arrows=not title.startswith("探索臂"))
    C.text(x + 0.7, y + 4.4, ev, fs=7.2, col="#3d3c39", ls=1.45)

# notes cell
x, y = 67.2, 3.6
ax.add_patch(FancyBboxPatch((x, y), 32.0, H, boxstyle="round,pad=0.2,rounding_size=0.8", fc="#f6f5f1", ec="none",
                            zorder=0.4))
C.text(x + 0.7, y + H - 0.4, "说明(对应 1006 设计 v4 §5.1–5.2)", fs=9.2, col=INK, bold=True)
C.text(x + 0.7, y + H - 2.6,
       "· 表 1 族 A(Holm,2 个;自然开局):\n  H-A1 Ours > w/o PGIF;H-A2 Ours > w/o 属性开局。\n"
       "· 表 2 预注册族(Holm,5 个;强制共享开局):\n  H-P1–P5,均为「Full − 消融」;另需 TR90\n  非劣(95% CI 下界 > −0.02)。\n"
       "· w/o 属性开局只在自然开局下有意义;\n  H-P2–P5 只在共享开局下比较，排除开局差异。\n"
       "· T0 pilot 提示：共享开局下 Full 与 w/o PGIF\n  无可见差别(0.581 vs 0.609),PGIF 在开局\n  之后的作用待正式表 2(@3,test)确认。\n"
       "· 「旧协议证据」来自不同协议 / 跟踪器 /\n  开局 / 指标口径，只说明方向。\n"
       "· 配置:acot/configs/methods/ours_no_pgif /\n  ours_no_absent / ours_no_temper / ours_pgif_resid /\n"
       "  ours_no_gate_memory / ours_pgif_bayes /\n  ours_no_gate_commit / ours_no_force_conf /\n  diag_oracle_absent。",
       fs=7.5, ls=1.45)
C.save("fig_arch_ablations")
