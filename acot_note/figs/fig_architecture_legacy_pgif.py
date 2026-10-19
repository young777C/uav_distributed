"""【LEGACY · 2026-10-04 标记为过时】原 fig_architecture.py。旧评测设定(仿真理想 tracklet ID、不过滤遮挡、语言
grounding 头开局、PGIF 无 absent-hold、WHERE 始终用仿真世界坐标);结果表为 10-01 PoC 批次。当前最终模型见
fig_architecture_v1.py / fig_architecture_v1.png。

原说明:ACoT-UAV-Track 最终模型结构:TAHRelM + PGIF(deployable,决策时不用任何 GT)。dataviz 方法绘制:
形式 = 数据流示意(非数据图表)。配色按"作用":冻结 = 中性灰;类别色只给两个"非冻结"角色,按调色板固定顺序取槽位:
slot1 蓝 #2a78d6 = TAHRelM(唯一训练过的部件,4,740 参数,离线 + DAgger 训练、部署时冻结);
slot2 橙 #eb6834 = PGIF(推理期规则,0 可训练参数)。validate_palette "#2a78d6,#eb6834" --mode light → 全部 PASS。
文字一律用正文墨色(不用类别色),类别由边框/填充承担。旧版(仅 TAHRelM)现为 fig_architecture_legacy_tahrelm.*。
数据流与代码对应:carla_uav_tracking/rollout/policy.py::act(S2 慢流每 6 帧:VLM/IAR/身份;S1 快流每帧:DiT),
TAHRelM = train/tah.py::TAHRelM,PGIF = rollout/asd.py::_filter_step(tempered)+ policy.py::_asd_gate_memory。
参数量:Qwen3-VL-4B safetensors 头统计;IAR/DiT/TargetIDHead 来自 runs/stage2_v5/stage2_best.pt;TAHRelM 来自 runs/tah_relm_dag.pt。
  python acot_note/figs/fig_architecture_legacy_pgif.py → fig_architecture_legacy_pgif.{png,pdf,svg}"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_f = os.path.expanduser("~/.local/share/fonts/SimHei.ttf")
if os.path.exists(_f):
    fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 8})

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
TAH, TAH_BG = "#2a78d6", "#dbe8f8"          # categorical slot 1
PG, PG_BG = "#eb6834", "#fde3d7"            # categorical slot 2
FROZ, FROZ_E, INP = "#e6e5e1", "#b9b8b2", "#ffffff"

fig = plt.figure(figsize=(15.5, 9.2), facecolor=SURF)
ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, title, sub="", fc=FROZ, ec=FROZ_E, lw=1.0, fs=8.8, bold=False, ls="-", top=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=ec, lw=lw,
                                linestyle=ls, zorder=2))
    if top:                                   # container: title pinned to the top edge
        ax.text(x + w / 2, y + h - 1.2, title, ha="center", va="top", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", zorder=3)
        return
    if sub:
        ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", zorder=3)
        ax.text(x + w / 2, y + h - 1.6 - fs * 0.42, sub, ha="center", va="top", fontsize=7.2, color=INK2,
                zorder=3, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", zorder=3)


def arrow(x1, y1, x2, y2, col=INK2, lw=1.2, ls="-", rad=0.0, label=None, lx=None, ly=None, fs=7.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=col, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        ax.text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2, label, fontsize=fs,
                color=INK2, ha="center", va="center", zorder=4, bbox=dict(fc=SURF, ec="none", pad=0.4))


# ---------------- title + legend
ax.text(1, 98.6, "ACoT-UAV-Track 最终模型结构:TAHRelM + PGIF(可部署:身份决策不使用真值;候选框与车辆定位为仿真理想输入)", fontsize=13.5,
        fontweight="bold", color=INK, va="top")
ax.text(1, 94.6, "唯一训练过的部件是 TAHRelM(4,740 参数,离线 + DAgger 训练,部署时冻结);PGIF 是推理期规则,0 个可训练参数;"
                 "其余 Qwen3-VL / IAR / crop-DINOv2 / DiT 全部冻结。", fontsize=8.6, color=INK2, va="top")
ax.legend(handles=[Patch(fc=FROZ, ec=FROZ_E, label="冻结模块"), Patch(fc=TAH_BG, ec=TAH, lw=2, label="TAHRelM(训练过,部署冻结)"),
                   Patch(fc=PG_BG, ec=PG, lw=2, label="PGIF(推理期规则,0 参数)"), Patch(fc=INP, ec=FROZ_E, label="输入 / 环境")],
          loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=4, fontsize=8, frameon=False, handlelength=1.6)

# ---------------- inputs
ax.text(8, 90.2, "每帧输入", fontsize=8, color=MUTED, ha="center")
box(1, 79, 14, 10, "无人机 RGB", "512×512 机载相机", fc=INP)
box(1, 66, 14, 11, "语言指令", "去位置化,只保留身份\n“Follow the gold\nMini Cooper.”", fc=INP)
box(1, 52, 14, 12, "候选集", "目标 + 干扰车的框\n持久 tracklet id\n(仿真理想关联)", fc=INP)
box(1, 41, 14, 9, "Proprio 5D", "cam_z, pitch, uav_vel×3", fc=INP)

# ---------------- frozen perception (S2 slow stream)
ax.add_patch(FancyBboxPatch((17.5, 39.5), 25, 51, boxstyle="round,pad=0.3,rounding_size=1.2", fc="none", ec=FROZ_E,
                            lw=1.0, ls=(0, (4, 3)), zorder=1))
ax.text(30, 90.0, "S2 慢流:每 6 帧运行一次", fontsize=7.6, color=MUTED, ha="center", va="top")
box(19, 74, 22, 14, "Qwen3-VL-4B(冻结,44.4 亿)", "文本在前、图像在后,一次前向\n取第 24 层隐藏态 → vlm_ctx(256×2560)\n语言 grounding 头(230 万,冻结)\n→ 冷启动时用语言选车",
    fs=8.4, bold=True)
box(19, 59, 22, 12, "IAR(冻结,97 万)", "8 个可学 query 对第 24 层特征\n做 cross-attention → z_im(8×256)", fs=8.4)
box(19, 42, 22, 14, "crop-DINOv2 ViT-S/14(冻结,约 2,200 万)", "每个候选框裁剪 → 384 维外观\n+ 位置 (u/W, v/H, depth)", fs=8.2)
arrow(15, 84, 19, 81); arrow(15, 71.5, 19, 79, rad=-0.15); arrow(15.6, 80, 18.8, 47)
arrow(15, 58, 19, 48)
arrow(30, 74, 30, 71.2)

# ---------------- TAHRelM
box(46, 56, 23, 32, "TAHRelM(WHICH,4,740 参数)", "", fc=TAH_BG, ec=TAH, lw=2.2, fs=9.6, bold=True, top=True)
box(47.5, 73, 20, 10, "7 个车辆不变的相对特征", "cos(外观, 记忆) · 相对位移 x/y · 深度\n排名 · 距离 · 软运动门 exp(-d/gate)", fc="#ffffff", ec=TAH, fs=8)
box(47.5, 65.5, 20, 6, "MLP 7·64·64·1 → 每个候选的分数", "", fc="#ffffff", ec=TAH, fs=7.8)
box(47.5, 57.5, 20, 7, "原始选择 s$^M$ = argmax,置信度 c", "", fc="#ffffff", ec=TAH, fs=7.8)
box(46, 42, 23, 11, "身份记忆 {app, pos, vel}", "外观:可学门 EMA;速度:可学衰减 EMA\n更新源由 PGIF 门控(见红色回路)", fc=TAH_BG, ec=TAH, lw=1.4, fs=8.2)
arrow(57.5, 73, 57.5, 71.8); arrow(57.5, 65.5, 57.5, 64.6)
arrow(41, 49, 46, 72, rad=-0.15, label="候选外观 + 位置", lx=43.4, ly=63)
arrow(57.5, 53, 57.5, 56, col=TAH, label="记忆参与打分", lx=63.5, ly=54.6)
arrow(41, 82, 46, 64, rad=0.1, label="冷启动语言选车", lx=43.6, ly=88.5)

# ---------------- PGIF
box(72, 56, 26, 32, "PGIF(推理期规则,0 可训练参数)", "", fc=PG_BG, ec=PG, lw=2.2, fs=9.6, bold=True, top=True)
box(73.5, 73.5, 23, 9, "① 置信度秩 → 投票权重", "ρ = c 在最近 W=30 步中的分位数;w = 0.5 + ρ", fc="#ffffff", ec=PG, fs=8)
box(73.5, 65, 23, 8, "② 回火的 tracklet 证据", "L$_k$ ← 0.98·L$_k$;L$_{s^M}$ += w(τ≈50 步≈30s)", fc="#ffffff", ec=PG, fs=8)
box(73.5, 57.5, 23, 6, "③ $\hat{s}$ = argmax 可见 L$_k$;若 $\hat{s}$ ≠ s$^M$ → 改写", "", fc="#ffffff", ec=PG, fs=7.8)
arrow(85, 73.5, 85, 73.2, col=PG); arrow(85, 65, 85, 63.7, col=PG)
arrow(67.5, 61, 72, 70, col=TAH, label="s$^M$, c", lx=69.6, ly=67.6)
arrow(73.5, 58.5, 69, 47.5, col=PG, lw=2.0, rad=-0.25, label="门控记忆更新:\n用 $\hat{s}$ 的截图重做本步更新", lx=79.5, ly=49.0)
ax.text(57.5, 38.6, "× 原始自更新(用 s$^M$ 更新记忆)= 闭环确认偏差回路,已被门控替代", fontsize=7.2, color=MUTED, ha="center")

# ---------------- commit → WHERE → HOW → env
box(73, 30, 25, 10, "承诺身份 $\hat{s}$", "K=5 帧滞回;PGIF 改写时直接承诺", fc="#ffffff", ec=PG, lw=1.4, fs=8.4)
arrow(85, 56, 85.5, 40.2, col=PG)
box(46, 22, 23, 12, "WHERE(冻结规则)", "被承诺车辆的世界坐标(仿真提供)\n→ 匀速外推 → 投影到相机 → z_ex", fs=8.4)
arrow(73, 35, 69, 29, label="承诺身份", lx=71.4, ly=34.0)
box(17, 18, 26, 16, "HOW:DiT(冻结,32 层,1.37 亿)", "S1 快流,每帧运行\nflow-matching 生成动作块(H=16)\n条件:z_ex · z_im · vlm_ctx · proprio\n→ (dx, dy, dz, dyaw) + search_mode", fs=8.4, bold=True)
arrow(46, 28, 43, 27, label="z_ex", lx=44.5, ly=29.6)
arrow(41.2, 63, 38, 34.2, col=MUTED, ls=(0, (3, 2)), rad=-0.45, label="z_im, vlm_ctx", lx=44.6, ly=40.0)
arrow(8, 41, 17, 30, col=MUTED, ls=(0, (3, 2)), rad=0.2, label="proprio", lx=10.4, ly=34.5)
box(1, 4, 42, 8, "CARLA 闭环:无人机执行动作 → 下一帧(选哪辆车决定下一帧看到什么)", "", fc=INP, fs=8.4)
arrow(30, 18, 22, 12.4)
arrow(1.5, 8, 1.5, 41, col=MUTED, rad=0.0)

# ---------------- key numbers
box(46, 3, 52, 16, "闭环结果(可部署,TAHRelM,Town05,19 个同种子 episode,三项客观指标)", "", fc="#ffffff", ec=FROZ_E, fs=8.6, bold=True, top=True)
rows = [("指标", "无纠偏", "+ PGIF", "配对 Δ [95% CI]"),
        ("正确锁定比例 ↑", "0.272", "0.613", "+0.341 [+0.18, +0.51]"),
        ("最长连续正确锁定 ↑", "5.9 s", "30.3 s", "+24.4 s [+12.0, +37.7]"),
        ("最长锁错 ↓", "15.7 s", "10.8 s", "-5.0 s [-14.0, +5.1](不显著)")]
xs = (47.5, 61.5, 70.5, 79.5)
for r, row in enumerate(rows):
    for c, cell in enumerate(row):
        ax.text(xs[c], 14.2 - r * 3.0, cell, fontsize=7.8, color=INK if r == 0 or c == 2 else INK2,
                fontweight="bold" if r == 0 else "normal", va="center")

# ---------------- LEGACY banner (2026-10-04)
ax.text(50, 50, "LEGACY", fontsize=110, color="#d03b3b", alpha=0.10, ha="center", va="center", rotation=20,
        fontweight="bold", zorder=10)
ax.text(99, 91.3, "【LEGACY · 已过时】旧评测设定(仿真理想 ID、不过滤遮挡、语言头开局);当前最终模型见 fig_architecture_v1.png",
        fontsize=9, color=INK, ha="right", va="top", fontweight="bold", zorder=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fde3d7", ec="#d03b3b", lw=1.2))
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(HERE, f"fig_architecture_legacy_pgif.{ext}"), dpi=200 if ext == "png" else None, facecolor=SURF)
print("wrote fig_architecture_legacy_pgif.{png,pdf,svg}")
