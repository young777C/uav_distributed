"""所有方法(我们的方法 / 消融 / 外部基线)的共享架构总览(2026-10-07 按协议 sim_proxy_v2@3 与 1006 设计 v4 更新)。按 acot
运行时代码绘制:acot/runtime/{perception,frontend,policy}.py 中对所有臂相同的部分画成实框;各臂唯一不同的「WHICH 身份模块」
画成虚线槽位。新增「两种开局设定」对照带(policy.py:_opening 的统一交接 F-c 与 shared_open_slot 的强制共享开局，设计 §3.4):
  自然开局(部署设定,表 1 / 3(a))    每个臂自己跑共享属性开局;第 0、6 步各身份模块按累计开局选择重置，第 12 步起自己打分
  强制共享开局(机制设定,表 2–5)     开局窗口内身份输出、承诺、WHERE 强制为参考运行的 actor;第 18 步释放
下方表格逐臂列出槽位内容与所属设定(配置来自 acot/configs/methods/*.yaml,用 acot.config.resolve 解析，不手抄)。
  python acot_note/figs/fig_arch_overview.py → fig_arch_overview.{png,pdf,svg}"""
import sys

from _archlib import (CHG, FROZ, FROZ_E, INK, INK2, INP, MUTED, OP, OP_BG, PG, PG_BG, REPO, SH, SH_BG, TAH, TAH_BG,
                      Canvas)
from matplotlib.patches import FancyBboxPatch, Patch

sys.path.insert(0, REPO)
from acot.config.resolve import load_method  # noqa: E402

NAMES = ("ours_final", "ours_no_pgif", "ours_no_absent", "ours_no_temper", "ours_pgif_resid", "ours_no_gate_memory",
         "ours_pgif_bayes", "ours_no_gate_commit", "ours_no_force_conf", "diag_oracle_head", "diag_gt_memory",
         "diag_oracle_absent", "lang_xattn", "ext_assoc_deepsort", "ext_assoc_motion", "ext_reid", "ext_dam4sam",
         "ext_oavat", "ext_reid_pgif", "ext_dam4sam_pgif")
M = {n: load_method(n)[0] for n in NAMES}


def tau(n):
    return f"{float(M[n]['identity']['conf_tau']):g}"


def pg(n):
    f = M[n]["filter"]
    if f["mode"] != "filter":
        return "无"
    k = {"tempered": "回火", "bayes": "贝叶斯 HMM", "resid": "等权"}[f["filter_kind"]]
    if float(f["resid_decay"]) == 1.0 and f["filter_kind"] == "tempered":
        k = "不回火(λ = 1)"
    out = [k]
    if f.get("absent_hold") and f["filter_kind"] != "bayes":     # pgif.py: the bayes branch has no absence output
        out.append("缺席保持")
    elif f["filter_kind"] == "bayes":
        out.append("无缺席输出")
    if f.get("gate_memory") is False:
        out.append("不回写记忆")
    if f.get("gate_commit") is False:
        out.append("不门控承诺")
    if f.get("force_conf") is False:
        out.append("不强制置信")
    return " + ".join(out)


C = Canvas(20, 20)
ax = C.ax
C.text(0.8, 99.4, "所有方法的共享架构总览:只有「WHICH 身份模块」不同;开局有两种设定(协议 sim_proxy_v2@3)", fs=15, col=INK,
       bold=True)
C.text(0.8, 97.9, "实框 = 所有臂完全相同的部分(同一协议、同一前端、同一承诺与 WHERE 规则);虚线槽位 = 各臂不同的部分。"
                  "S2 决策 tick 每 0.6 s 一次(开局窗口 = 第 0 / 6 / 12 步),DiT 每帧(0.1 s)出动作。"
                  "理想化感知(候选框真值投影、WHERE 真值坐标)对所有臂相同。", fs=9)
ax.legend(handles=[Patch(fc=OP_BG, ec=OP, lw=2, label="共享属性开局模块 / 自然开局(部署设定)"),
                   Patch(fc=SH_BG, ec=SH, lw=2, label="强制共享开局(机制设定)"),
                   Patch(fc=FROZ, ec=FROZ_E, label="共享冻结模块 / 规则"), Patch(fc=INP, ec=FROZ_E, label="输入 / 环境"),
                   Patch(fc="#ffffff", ec=INK, lw=1.6, ls="--", label="WHICH 槽位(各臂不同)")],
          loc="upper left", bbox_to_anchor=(0.006, 0.968), ncol=5, fontsize=8.8, frameon=False, handlelength=1.6)

# ---------------------------------------------------------------- row 1: inputs + perception + opening + frontend
C.box(0.8, 84.2, 10.5, 7.4, "输入", "语言指令\n高清开局帧(3×)\n无人机 RGB\nProprio", fc=INP, bold=True)
C.group(13.5, 77.5, 30.5, 15.4, "共享感知(sim_proxy_v2@3:匈牙利跟踪器 + 环境确定性)", "#f3f2ee", FROZ_E, lw=1.2)
C.box(14.5, 83.3, 9.0, 7.0, "候选框", "任务车辆真值投影\n射线遮挡过滤\n每集固定顺序打乱", sfs=7.4)
C.box(24.5, 83.3, 8.5, 7.0, "crop-DINOv2", "ViT-S/14 冻结\n384 维外观", sfs=7.4)
C.box(34.0, 83.3, 9.0, 7.0, "跟踪器", "匈牙利匹配\n运动 + 外观代价\n→ 每个候选的 ID", sfs=7.4)
C.text(28.7, 81.7, "输出:每个候选 = (ID, 框, 深度, 外观, 真值世界坐标*)", fs=7.6, ha="center")
C.text(28.7, 80.0, "*只给 WHERE 用;身份链路不读标签", fs=7.2, col=MUTED, ha="center")
C.arrow(23.5, 86.8, 24.5, 86.8); C.arrow(33.0, 86.8, 34.0, 86.8)
C.group(46.0, 77.5, 22.0, 15.4, "共享属性开局模块(开局 tick 第 0 / 6 / 12 步)", OP_BG, OP)
C.box(47.0, 79.6, 20.0, 10.6, "", "解析指令颜色词 + 车型词\n高清帧裁剪 → DINOv2 线性头 / 微调头\nlog p(车型) + log p(颜色)\n"
                              "按跟踪器 ID 跨 tick 累加 → argmax\n= 累计开局选择(如何交给槽位见下方两种设定)",
      fc="#ffffff", ec=OP, lw=1.2, sfs=7.6)
C.group(70.0, 77.5, 29.2, 15.4, "共享冻结前端(Qwen3-VL-4B + Stage-2)", "#f3f2ee", FROZ_E, lw=1.2)
C.box(71.0, 85.1, 27.2, 5.1, "Qwen3-VL-4B", "指令 + RGB → 第 24 层 vlm_ctx", sfs=7.6)
C.box(71.0, 78.6, 8.5, 5.4, "EAR", "航点先验\nz_ex(默认)", sfs=7.4)
C.box(80.4, 78.6, 8.5, 5.4, "IAR", "隐式先验\nz_im", sfs=7.4)
C.box(89.7, 78.6, 8.5, 5.4, "grounding 头", "每个候选的\n语言 logits", sfs=7.4)
C.arrow(11.3, 88.0, 14.5, 88.0)
C.arrow(43.0, 86.8, 47.0, 86.8, label="ID", lx=45.0, ly=87.7)

# ---------------------------------------------------------------- row 2: WHICH slot + confidence + commit + WHERE
ax.add_patch(FancyBboxPatch((13.5, 58.0), 54.5, 16.3, boxstyle="round,pad=0.25,rounding_size=1.0", fc="#ffffff", ec=INK,
                            lw=1.8, linestyle=CHG, zorder=1.5))
C.text(40.75, 74.0, "WHICH 槽位(各臂不同):每个 S2 tick 输入候选 + 开局选择 → 输出每个候选的分数(或弃权)", fs=10, col=INK,
       bold=True, ha="center")
C.box(14.5, 59.4, 15.0, 12.0, "我们的方法", "TAHRelM 关联记忆\n+ PGIF(回火 + 缺席保持)\n门控记忆写入\n\n详见 fig_architecture_v3",
      fc=TAH_BG, ec=TAH, lw=1.6, bold=True, sfs=7.6)
C.box(30.5, 59.4, 17.5, 12.0, "消融",
      "表 1 族 A:w/o PGIF · w/o 属性开局\n表 2(H-P1–P5):w/o PGIF · w/o 缺席保持\n  w/o 回火 · w/o 置信加权 · w/o 记忆门控\n"
      "探索：贝叶斯 · w/o 承诺门控 · w/o 强制置信\n诊断上界 U1–U3(用真值)\n详见 fig_arch_ablations",
      fc=PG_BG, ec=PG, lw=1.6, bold=True, sfs=7.3)
C.box(49.0, 59.4, 18.0, 12.0, "外部基线(6 个)",
      "单帧语言 grounding · DINOv2 re-ID\nDeepSORT 式 · OC-SORT 式\nDAM4SAM · OA-VAT identity adaptation\n"
      "均无 PGIF;跨记忆：re-ID / DAM4SAM\n+ PGIF(表 3(b))\n详见 fig_arch_baselines",
      fc=FROZ, ec=FROZ_E, lw=1.2, bold=True, sfs=7.3)
C.box(70.0, 65.9, 13.5, 8.3, "置信度 c", "槽位分数的 top1 − top2 差\n(PGIF 改写 → c = 1;\n弃权 → c = 0)", sfs=7.4)
C.box(85.0, 65.9, 14.2, 8.3, "承诺(K = 5 滞回)", "c ≥ conf_tau 才更新\n挑战者需连续 5 tick\nconf_tau 按臂设定(见下表)", sfs=7.4)
C.box(70.0, 58.0, 29.2, 6.5, "WHERE(共享规则)", "承诺 ID 有匹配框 → 真值世界坐标 → 匀速估计器 → z_ex\n承诺 ID 无匹配框 → 沿用 EAR 航点", sfs=7.4)
C.box(14.5, 52.9, 53.5, 3.6, "DiT(冻结,32 层):z_ex · z_im · vlm_ctx · proprio → (dx, dy, dz, dyaw) + 搜索标志 → CARLA 闭环",
      fs=8.6)
C.arrow(40.0, 77.5, 40.0, 74.6, label="候选 + ID", lx=36.5, ly=76.0)
C.arrow(57.0, 77.5, 57.0, 74.6, col=OP, lw=1.6, label="开局选择", lx=60.6, ly=76.0)
C.arrow(93.9, 78.6, 66.0, 71.6, col=MUTED, lw=1.0, rad=-0.15, label="语言 logits(xattn 与 w/o 属性开局使用)", lx=82.0,
        ly=75.7)
C.arrow(68.0, 68.5, 70.0, 69.6, label="分数", lx=68.9, ly=70.8)
C.arrow(83.5, 70.1, 85.0, 70.1)
C.arrow(92.1, 65.9, 92.1, 64.5)
C.arrow(70.0, 60.2, 68.0, 55.5, label="z_ex", lx=70.6, ly=57.4)

# ---------------------------------------------------------------- row 3: the two opening settings
C.text(0.8, 50.9, "两种开局设定:开局窗口(第 0–17 步)内由谁决定身份", fs=11, col=INK, bold=True)


def setting(x, y, w, h, tag, tag_fc, title, blocks, note, bg, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=bg, ec=ec, lw=1.6,
                                zorder=0.4))
    C.tag(x + 0.8, y + h - 1.0, tag, fc=tag_fc, fs=8.4)
    C.text(x + 0.8 + len(tag) * 0.62 + 2.2, y + h - 0.35, title, fs=9.6, col=INK, bold=True)
    n, gap = len(blocks), 1.0
    bw = (w - 1.6 - gap * (n - 1)) / n
    for i, (t, s, fc, bec, ls) in enumerate(blocks):
        bx = x + 0.8 + i * (bw + gap)
        C.box(bx, y + 5.6, bw, h - 8.6, t, s, fc=fc, ec=bec, lw=1.3, ls=ls, bold=True, sfs=7.25, fs=8.6)
        if i < n - 1:
            C.arrow(bx + bw, y + 5.6 + (h - 8.6) / 2, bx + bw + gap, y + 5.6 + (h - 8.6) / 2)
    C.text(x + 0.8, y + 4.8, note, fs=7.5, col=INK2, ls=1.5)


setting(0.8, 30.6, 48.6, 19.0, "自然开局", OP, "部署设定 · 表 1(主对比)、表 3(a)(跨地图)", [
    ("第 0 / 6 / 12 步", "每个臂自己运行\n共享属性开局模块\n(画面来自本臂的\n闭环轨迹)", OP_BG, OP, "-"),
    ("统一交接(F-c)", "第 0、6 步：身份模块按\n累计开局选择重置\nTAHRelM 清空记忆后写入;\n关联 / DAM4SAM / re-ID /\nOA-VAT 重新初始化", OP_BG, OP, "-"),
    ("第 12 步(窗口末)", "不再重置、不强制：\n身份模块自己打分\n(xattn 仍用开局选择);\n承诺按 c ≥ conf_tau\n且 K = 5 滞回", "#ffffff", FROZ_E, "-"),
    ("第 18 步起", "各臂完全按\n自身机制运行", "#ffffff", FROZ_E, "-")],
    "代表独立部署表现：开局结果可因各臂轨迹、承诺建立方式与渲染噪声而不同。\n"
    "T0 pilot(@2,dev-B):开局锁对 Ours 0.94、w/o PGIF 0.75、w/o 属性开局 0.33;承诺正确 0.479 / 0.347 / 0.226。",
    "#f4fbf8", OP)
setting(50.6, 30.6, 48.6, 19.0, "强制共享开局", SH, "机制设定 · 表 2(PGIF 组件)、3(b)、4、5", [
    ("参考开局(干预)", "Ours 跑 19 步 × 2 遍\n记录第 0 / 6 / 12 步的\n参考 actor 编号\n(dev-B:devB3.json;\ntest 在正式批次前生成)", SH_BG, SH, "-"),
    ("强制窗口(0–17 步)", "身份输出 = 承诺 =\n参考 actor(c = 1,\nWHERE 跟随它);\n所有身份模块按它重置;\nPGIF 只累加同一票", SH_BG, SH, "-"),
    ("释放(第 18 步起)", "各臂按自身机制运行\n→ 只比较开局之后的\n身份维持", "#ffffff", FROZ_E, "-"),
    ("fallback", "参考 actor 不在\n候选中 → 该种子\n从所有机制表剔除", "#ffffff", INK, CHG)],
    "参考 actor 经 actor 编号映射到当前候选 = 对所有机制臂相同的特权输入 shared_opening_actor;不代表部署表现。\n"
    "T0 pilot(@2,dev-B):Full 0.581、w/o PGIF 0.609(自然开局下 0.508 / 0.357)——统一开局后 PGIF 的优势不再可见(pilot,待表 2 确认)。",
    "#f8f6fd", SH)

# ---------------------------------------------------------------- table: what goes into the slot, per arm
NAT, SHR, BOTH = "自然", "共享", "两者"
spec = [
    ("ours", "我们的方法", "ours_final", "TAHRelM 关联记忆", "自身选择,经 PGIF 门控", "自然(表 1、3(a))+ 共享(表 2–5)", "否",
     "pilot @2 完成;@3 进行中"),
    ("no_pgif", "消融", "ours_no_pgif", "TAHRelM 关联记忆", "自身选择(无门控)", "自然(表 1 族 A)+ 共享(表 2 H-P1)", "否",
     "pilot @2 完成(两种设定);@3 进行中"),
    ("open_head", "消融", "ours_final", "TAHRelM 关联记忆(开局 = grounding 头)", "自身选择,经 PGIF 门控", "自然(表 1 族 A)", "否",
     "pilot @2 完成;@3 进行中"),
    ("no_absent_hold", "消融", "ours_no_absent", "TAHRelM 关联记忆", "自身选择,经 PGIF 门控", "共享(表 2 H-P2)", "否", "test 阶段运行"),
    ("no_temper", "消融", "ours_no_temper", "TAHRelM 关联记忆", "自身选择,经 PGIF 门控", "共享(表 2 H-P3)", "否", "test 阶段运行"),
    ("pgif_resid", "消融", "ours_pgif_resid", "TAHRelM 关联记忆", "自身选择,经 PGIF 门控", "共享(表 2 H-P4)", "否", "test 阶段运行"),
    ("no_gate_memory", "消融", "ours_no_gate_memory", "TAHRelM 关联记忆", "自身选择(PGIF 不回写)", "共享(表 2 H-P5)", "否",
     "test 阶段运行"),
    ("pgif_bayes", "探索", "ours_pgif_bayes", "TAHRelM 关联记忆", "自身选择,经 PGIF 门控", "共享(表 2 探索)", "否", "test 阶段运行"),
    ("no_gate_commit", "探索", "ours_no_gate_commit", "TAHRelM 关联记忆", "记忆按 PGIF;承诺跟记忆原选", "共享(表 2 探索)", "否",
     "开关已实现;test 阶段运行"),
    ("no_force_conf", "探索", "ours_no_force_conf", "TAHRelM 关联记忆", "自身选择,经 PGIF 门控", "共享(表 2 探索)", "否",
     "开关已实现;test 阶段运行"),
    ("lang_xattn", "外部基线", "lang_xattn", "Stage-2 grounding 头(逐 tick,无记忆)", "—", "自然(表 1 族 B)", "否", "调参网格 @3 排队"),
    ("ext_deepsort", "外部基线", "ext_assoc_deepsort", "运动门 + 外观 EMA 单目标关联", "自身关联结果", "自然(表 1 族 B)", "否",
     "调参网格 @3 排队"),
    ("ext_ocsort", "外部基线", "ext_assoc_motion", "运动门单目标关联(无外观)", "自身关联结果", "自然(表 1 族 B)", "否", "调参网格 @3 排队"),
    ("ext_reid", "外部基线", "ext_reid", "DINOv2 外观 gallery re-ID", "存承诺身份的视图", "自然(表 1 族 B)", "否", "调参网格 @3 排队"),
    ("ext_dam4sam", "外部基线", "ext_dam4sam", "DAM4SAM(SAM2.1++-L 掩码记忆)", "SAM2 自身记忆", "自然(表 1 族 B、3(a))", "否",
     "调参网格 @3 排队"),
    ("ext_oavat", "外部基线", "ext_oavat", "OA-VAT:DINOv3 原型 + 置信 KF", "锁定框在线增强原型", "自然(表 1 族 B、3(a))", "否",
     "网格 @3 排队;须过 G9(b′)"),
    ("reid + PGIF", "跨记忆", "ext_reid_pgif", "DINOv2 外观 gallery re-ID", "存承诺身份的视图", "共享(表 3(b))", "否", "test 阶段运行"),
    ("dam4sam + PGIF", "跨记忆", "ext_dam4sam_pgif", "DAM4SAM(SAM2.1++-L 掩码记忆)", "SAM2 记忆;PGIF 改写时重初始化", "共享(表 3(b))",
     "否", "test 阶段运行"),
    ("U1 oracle", "诊断上界", "diag_oracle_head", "真值身份(直接输出真目标)", "—", "自然(表 1 灰色行)", "是(诊断)", "冒烟通过;test 阶段运行"),
    ("U2 gt_memory", "诊断上界", "diag_gt_memory", "TAHRelM 关联记忆", "记忆每步用真值目标刷新", "自然(表 1 灰色行)", "是(诊断)",
     "冒烟通过;test 阶段运行"),
    ("U3 oracle_absent", "诊断上界", "diag_oracle_absent", "TAHRelM 关联记忆", "真值判「目标不在」时撤销", "自然(表 1 灰色行)", "是(诊断)",
     "test 阶段运行"),
]
rows = [[arm, cat, which, upd, pg(mn), tau(mn), gt, op, st] for arm, cat, mn, which, upd, op, gt, st in spec]
cols = [(1.0, "臂"), (9.2, "类别"), (14.8, "WHICH 身份模块"), (32.5, "记忆 / 状态更新来源"), (47.5, "PGIF"),
        (60.0, "conf_tau"), (64.6, "用真值"), (69.6, "开局设定(所在表)"), (85.2, "T0 状态(dev-B)")]
C.text(0.8, 28.8, "槽位逐臂对照(配置由 acot/configs/methods/*.yaml 解析;conf_tau 为默认值，外部基线在 T0 网格中调参)",
       fs=10.5, col=INK, bold=True)
C.table(0.8, 27.0, cols, rows, fs=7.4, dy=1.13, hl={0}, hl_fc=TAH_BG)
for i, (_, _, _, _, _, op, _, _) in enumerate(spec):           # setting colour bar next to the setting column
    y = 27.0 - 1.13 * (i + 1)
    has_n, has_s = op.startswith("自然") or "+ 自然" in op, "共享" in op
    if has_n:
        ax.add_patch(FancyBboxPatch((68.9, y - 0.42), 0.32, 0.84, boxstyle="square,pad=0", fc=OP, ec="none", zorder=2))
    if has_s:
        ax.add_patch(FancyBboxPatch((69.25, y - 0.42), 0.32, 0.84, boxstyle="square,pad=0", fc=SH, ec="none", zorder=2))
C.text(0.8, 2.3, "注:① 外部基线是单目标简化或移植实现(DeepSORT / OC-SORT 只取关联思想;re-ID 为 1004 移植，与旧入口逐位一致;"
                 "DAM4SAM 经服务调用原实现;OA-VAT 复用原仓库身份代码),各自在 dev-B 上按运行前冻结的规则调参(G12)。"
                 "② UAST 因目标分支权重不可得不纳入(1006_t0_log D2)。\n"
                 "③ 所有臂共享协议 sim_proxy_v2@3 = 匈牙利跟踪器 + 统一开局交接 + CARLA 环境确定性(@2 + 同步生成场景、每种子重载世界)。"
                 "表中色条:绿 = 自然开局，紫 = 强制共享开局。",
       fs=7.6, col=INK2)
C.save("fig_arch_overview")
