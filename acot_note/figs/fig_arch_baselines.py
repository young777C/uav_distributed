"""外部基线的实现架构(2026-10-07 按协议 sim_proxy_v2@3 与 1006 设计 v4 更新)。共享部分与两种开局设定的定义见
fig_arch_overview;本图逐个画外部基线在 WHICH 槽位里的内部流程，并标出它们在两种开局设定下的用法:
  自然开局(绿)        表 1 族 B、表 3(a):每个基线跑共享属性开局;第 0、6 步按累计开局选择重新初始化(统一交接 F-c,
                      policy.py:_opening → head.reset() → 下一次 score 用 init_slot),第 12 步起按自身机制运行
  强制共享开局(紫)    表 3(b) 跨记忆:re-ID ± PGIF、DAM4SAM ± PGIF;开局窗口内身份 / 承诺 / WHERE = 参考 actor
按代码绘制:acot/runtime/heads.py(AssocHead / ReidHead / Dam4samHead)、acot/runtime/head_oavat.py(OavatHead)、
acot/runtime/policy.py(_identity / _opening);配置 acot/configs/methods/*.yaml;调参网格 t0_grid_devB.yaml。
  python acot_note/figs/fig_arch_baselines.py → fig_arch_baselines.{png,pdf,svg}"""
import sys

from _archlib import (CHG, FROZ, FROZ_E, INK, INK2, MUTED, OP, OP_BG, REPO, SH, Canvas)
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, REPO)
from acot.config.resolve import load_method  # noqa: E402

M = {n: load_method(n)[0] for n in ("lang_xattn", "ext_assoc_deepsort", "ext_assoc_motion", "ext_reid", "ext_dam4sam",
                                    "ext_oavat")}
I = {n: m["identity"] for n, m in M.items()}
gate = I["ext_assoc_deepsort"]["assoc"]["gate_px"]
lam = I["ext_assoc_deepsort"]["assoc"]["lambda"]
dg = I["ext_dam4sam"]["dam4sam"]
rg = I["ext_reid"]["reid"]
og = I["ext_oavat"]["oavat"]


def tau(n):
    return f"{float(I[n]['conf_tau']):g}"


C = Canvas(20, 17)
ax = C.ax
C.text(0.8, 99.3, "外部基线的实现架构:各自替换 WHICH 槽位，其余与我们的方法共享;标出两种开局设定下的用法", fs=15, col=INK,
       bold=True)
C.text(0.8, 97.5, "共享部分(感知 + 匈牙利跟踪器、属性开局模块、冻结前端、置信度 c = top1 - top2、承诺 K = 5、WHERE、DiT)见 "
                  "fig_arch_overview。外部基线本身都没有 PGIF。墨色虚线框 = 该基线自己的身份机制;绿色 = 自然开局下的开局交接;"
                  "白框 = 共享的承诺。协议 sim_proxy_v2@3。", fs=9)

# ---------------------------------------------------------------- the two opening settings for the baselines
for x, tg, col, bg, title, body in (
        (0.8, "自然开局", OP, "#f4fbf8", "表 1 族 B(主对比)、表 3(a)(跨地图,DAM4SAM / OA-VAT)",
         "每个基线自己运行共享属性开局模块。统一交接(F-c,协议 @2 起):第 0、6 步把身份模块重置，再用当时的累计开局选择\n"
         "重新初始化(轨迹 / gallery / SAM2 / OA-VAT 参考与锁定);第 12 步起各自按自身机制运行(xattn 在窗口内一直用开局选择)。\n"
         "旧版本只在第 0 步读一次开局选择，已改。"),
        (50.6, "强制共享开局", SH, "#f8f6fd", "表 3(b)(跨记忆):re-ID ± PGIF、DAM4SAM ± PGIF",
         "开局窗口(第 0–17 步)内身份输出、承诺与 WHERE 强制为参考 actor,所有身份模块按它重置;第 18 步释放。\n"
         "只比较「同一记忆加不加 PGIF」,不与表 1 混用。预测:re-ID + PGIF 更高(自置信 AUROC 0.82);\n"
         "DAM4SAM + PGIF 无显著增益(SAM2 IoU 对正确性的 AUROC 只有 0.53)。")):
    ax.add_patch(FancyBboxPatch((x, 87.0), 48.6, 8.2, boxstyle="round,pad=0.2,rounding_size=0.8", fc=bg, ec=col, lw=1.6,
                                zorder=0.4))
    C.tag(x + 0.8, 94.2, tg, fc=col, fs=8.4)
    C.text(x + 0.8 + len(tg) * 0.62 + 2.2, 94.85, title, fs=9.4, col=INK, bold=True)
    C.text(x + 0.8, 92.6, body, fs=7.7, col=INK2, ls=1.5)


def chain(x0, y0, w, h, blocks, gap=1.2):
    n = len(blocks)
    bw = (w - gap * (n - 1)) / n
    for i, (t, s, k) in enumerate(blocks):
        x = x0 + i * (bw + gap)
        if k == "op":
            C.box(x, y0, bw, h, t, s, fc=OP_BG, ec=OP, lw=1.2, bold=True, sfs=7.1, fs=8.5)
        elif k == "own":
            C.box(x, y0, bw, h, t, s, fc="#ffffff", ec=INK, lw=1.8, ls=CHG, bold=True, sfs=7.1, fs=8.5)
        elif k == "svc":
            C.box(x, y0, bw, h, t, s, fc=FROZ, ec=INK, lw=1.8, ls=CHG, bold=True, sfs=7.1, fs=8.5)
        else:
            C.box(x, y0, bw, h, t, s, fc="#ffffff", ec=FROZ_E, lw=1.2, bold=True, sfs=7.1, fs=8.5)
        if i < n - 1:
            C.arrow(x + bw, y0 + h / 2, x + bw + gap, y0 + h / 2)


HAND = "第 0、6 步：重置后用\n累计开局选择{}\n(统一交接 F-c);\n第 12 步起自己运行"
P = [
    ("lang_xattn:单帧语言 grounding(Stage-2 TargetIDHead,无记忆)", [("自然 · 表 1 族 B", OP)],
     [("开局窗口", "第 0 / 6 / 12 步:\n采用属性开局的\n累计选择", "op"),
      ("TargetIDHead", "每个候选框内的 VLM\n网格池化特征 ×\nvlm_ctx 交叉注意力\n→ 每个候选的 logit", "own"),
      ("逐 tick argmax", "每个 tick 独立选车\n没有记忆、没有\n跨 tick 的身份状态", "own"),
      ("承诺", f"c = logit 差\nc ≥ {tau('lang_xattn')}(默认)\nK = 5 滞回", "cm")],
     "身份完全依赖单帧语言;look-alike 同框时只能靠外观细节。冻结，与 w/o 属性开局用的是同一个头。调参:conf_tau ∈ {0, 0.2, 0.5}。"),
    ("ext_deepsort:DeepSORT 式单目标关联(运动门 + 外观 EMA)", [("自然 · 表 1 族 B", OP)],
     [("开局交接", HAND.format("建立轨迹\n(位置、外观)"), "op"),
      ("运动预测 + 门", f"预测 = 上次位置 + 速度\nd = 候选中心到预测距离\nd ≥ {gate:g} px 的候选排除", "own"),
      ("打分", f"分数 = −d / gate\n + λ × cos(外观, 轨迹外观)\n(λ = {lam:g})门内 argmax", "own"),
      ("自我更新", "用自己选中的框更新\n位置、速度;外观\nEMA 0.9 / 0.1", "own"),
      ("承诺", f"c = 分数差\nc ≥ {tau('ext_assoc_deepsort')}(默认)\nK = 5 滞回", "cm")],
     "单目标简化实现，只取 DeepSORT 的关联思想;轨迹由自己的选择更新(确认偏差风险同 w/o PGIF)。"
     "调参:gate ∈ {120, 160, 240} × λ ∈ {0.5, 1, 2} × conf_tau。"),
    ("ext_ocsort:OC-SORT 式纯运动关联(无外观)", [("自然 · 表 1 族 B", OP)],
     [("开局交接", HAND.format("建立轨迹\n(位置)"), "op"),
      ("运动预测 + 门", f"预测 = 上次位置 + 速度\nd ≥ {gate:g} px 的候选排除", "own"),
      ("打分", "分数 = −d / gate\n(不用外观)\n门内 argmax", "own"),
      ("自我更新", "用自己选中的框更新\n位置、速度", "own"),
      ("承诺", f"c = 分数差\nc ≥ {tau('ext_assoc_motion')}(默认)\nK = 5 滞回", "cm")],
     "与 ext_deepsort 只差外观项;遮挡后目标位置跳变或 look-alike 交会时最容易换车。调参:gate × conf_tau。"),
    ("ext_reid:DINOv2 外观 gallery re-ID(1004 移植)", [("自然 · 表 1 族 B", OP), ("共享 · 表 3(b)(± PGIF)", SH)],
     [("开局交接", HAND.format("清空 gallery,\n存它的视图"), "op"),
      ("gallery 打分", f"crop-DINOv2(384 维)\nK = {int(rg['bank'])} 个视图的 gallery\n候选对 gallery 的\ntop-{int(rg['topm'])} 余弦", "own"),
      ("时间平均", f"可选 EMA(tavg)\n平滑每个 ID 的分数\n(默认 {float(rg['tavg']):g})", "own"),
      ("存视图", "先打分、后存:\n存承诺身份的框\n(未承诺或刚重置时\n存开局选择),EMA 0.9", "own"),
      ("承诺", f"c = 分数差\nc ≥ {tau('ext_reid')}(默认)\nK = 5 滞回", "cm")],
     "接 PGIF 时(表 3(b))门控为空操作:gallery 存的是承诺身份,PGIF 改写承诺后下一 tick 才进入 gallery(与旧入口逐位一致)。"
     "调参:tavg ∈ {0, 0.4} × conf_tau。"),
    (f"ext_dam4sam:DAM4SAM(CVPR'25,{dg['model']} 掩码记忆，独立服务)", [("自然 · 表 1 族 B、3(a)", OP),
                                                                     ("共享 · 表 3(b)(± PGIF)", SH)],
     [("开局交接", HAND.format("的框\n初始化 SAM2"), "op"),
      ("DAM4SAM 跟踪", "每个 S2 tick(0.6 s)\n送一帧 → 掩码框 + IoU\n(干扰物感知记忆\n在服务内部更新)", "svc"),
      ("掩码框 → 候选", f"候选中心与掩码框中心\n距离 < gate({dg['gate_px']:g} px):\n分数 = −d / gate;\n无掩码 → 全部弃权", "own"),
      ("承诺", f"c = 分数差\nc ≥ {tau('ext_dam4sam')}(默认)\nK = 5 滞回;\n弃权时保持", "cm")],
     f"经 dam4sam-svc 容器调用原实现;every_frame = {str(dg['every_frame']).lower()}(只在 S2 tick 更新)。接 PGIF 时 PGIF 改写 → "
     "SAM2 用 MAP 框重新初始化。调参:gate ∈ {160, 240, 320} × conf_tau。"),
    ("ext_oavat:OA-VAT identity adaptation(CVPR'26;复用原仓库身份代码)", [("自然 · 表 1 族 B、3(a)", OP)],
     [("开局交接", "第 0、6 步：重置后用\n开局选择在 3× 开局帧\n上的视图(框外置零)\n+ 水平翻转 = 参考;锁定", "op"),
      ("锁定跟踪", f"锁定 ID 可见 → 保持\n(跟踪中不核验身份);\nKF 更新;每 {int(og['enhance_every'])} tick\nEMA 增强原型(α = {og['enhance_alpha']:g})", "own"),
      ("丢失 → 重捕获", f"KF 预测框;IoU ≥ {og['iou_pred']:g}\n的候选({int(og['fullframe_after'])} 次失败后全图)\n"
                     f"余弦 ≥ {og['reacq_thresh']:g} 则重锁,\n否则弃权", "own"),
      ("重检测", f"预测超过 {int(og['max_predict_ticks'])} tick →\n原型重置为参考;\n全部候选按\n{og['match_thresh']:g} 阈值匹配", "own"),
      ("承诺", f"c = top1 − top2\n(选中者 +1)\nc ≥ {tau('ext_oavat')}(默认);\nK = 5", "cm")],
     "DINOv3 ViT-B/16(原仓库权重)ROI 池化 patch token;帧单位参数已换算为 tick(D6)。调参:reacq_thresh ∈ {0.6, 0.7, 0.8} × conf_tau。\n"
     "须通过 G9(b′)(原型匹配 top-1 显著高于随机),否则删除该基线。"),
]
cells = [(0.8, 63.2), (50.6, 63.2), (0.8, 40.4), (50.6, 40.4), (0.8, 17.6), (50.6, 17.6)]
PH = 21.6
for (title, tg, blocks, note), (x, y) in zip(P, cells):
    ax.add_patch(FancyBboxPatch((x, y), 48.6, PH, boxstyle="round,pad=0.2,rounding_size=0.8", fc="#fcfcfb", ec=MUTED,
                                lw=0.8, zorder=0.4))
    C.text(x + 0.8, y + PH - 0.4, title, fs=9.8, col=INK, bold=True)
    tx = x + 0.8
    for t, col in tg:
        C.tag(tx, y + PH - 2.7, t, fc=col, fs=7.8)
        tx += sum(1.05 if ord(ch) > 0x2E80 else 0.55 for ch in t) * 0.62 + 1.8
    chain(x + 0.8, y + 5.0, 47.0, 12.4, blocks)
    C.text(x + 0.8, y + 4.0, note, fs=7.3, col="#3d3c39", ls=1.45, wrap=False)

# ---------------------------------------------------------------- fairness notes
C.text(0.8, 15.6, "公平性与可比性(1004_s55_fairness_audit、1006 设计 v4 §6 与本次代码核对)", fs=10.5, col=INK, bold=True)
C.text(0.8, 13.6,
       "1. 开局：所有臂共享同一属性开局模块与统一交接规则(F-c,已实现);机制比较(表 3(b))用强制共享开局，开局窗口内完全相同。\n"
       "2. 调参预算：每个外部基线在 dev-B(40 个种子)上按网格调参(t0_grid_devB,63 组),选择规则在运行前冻结:承诺正确最高，差 < 0.03 时取 TR90 高者(G12);"
       "我们的方法不再调参。\n"
       "3. 实现:DeepSORT / OC-SORT 式为单目标简化的自实现;re-ID 为 1004 移植(新旧入口决策逐位一致，G8);DAM4SAM 经服务调用原实现;"
       "OA-VAT 复用原仓库身份代码，检测器 / 规划器换成共享模块。\n"
       "4. 协议：所有臂共享 sim_proxy_v2@3(匈牙利跟踪器 + 统一交接 + CARLA 环境确定性)与同一跟踪器 ID 记账;同种子、同重复编号的策略随机数相同。\n"
       "5. UAST 不纳入(目标分支权重不可得，1006_t0_log D2)。旧协议(v1:理想 ID、不滤遮挡、语言开局、未调参)下的数字(xattn 0.27、OC-SORT 0.23、"
       "DeepSORT 0.29、DAM4SAM 0.37)不可比。\n"
       "6. 状态：调参网格 @3 排在 @3 pilot 之后(约 2026-10-07 04:00 启动);表 1 / 3 在 test 种子上、按冻结的重复次数运行。",
       fs=8.0, ls=1.6)
C.save("fig_arch_baselines")
