"""阶段 0 · Qwen3-VL LoRA grounding 模型细节图(dataviz 方法:先定形式,再按"作用"配色,配色经 validate_palette 校验)。
(a) 数据流示意(非图表):输入 → Qwen3-VL(截断到前 24 层,LoRA 在 16–23 层)→ 候选框池化 → grounding head → CE(GT 身份)
(b) 参数量(数据图):各部件实测参数量,单一对数轴、水平条形、直接标注数值
(c) r16 与 r0 的唯一差别(对照设计)
配色:冻结 = 中性灰(不是类别色);可训练/监督按类别槽位固定顺序:slot1 蓝 = grounding head,slot2 橙 = LoRA,slot3 青 = GT 监督。
校验:validate_palette "#2a78d6,#eb6834,#1baf7a" --mode light → 全部 PASS;青色对比度 2.74 < 3:1 → 所有元素带可见文字标注(relief rule)。
参数量来自 Qwen/Qwen3-VL-4B-Instruct safetensors 头信息;LoRA/head 参数量来自训练日志(trainable params)。
  python acot_note/figs/fig_stage0_lora.py → fig_stage0_lora.{png,pdf}"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_f = os.path.expanduser("~/.local/share/fonts/SimHei.ttf")
if os.path.exists(_f):
    fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "pdf.fonttype": 42, "font.size": 8})

# palette (reference instance, light): roles
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
HEAD, LORA, SUP = "#2a78d6", "#eb6834", "#1baf7a"           # categorical slots 1,2,3 (validated)
FROZ, FROZ_E, UNUSED = "#e6e5e1", "#b9b8b2", "#f3f2ef"      # neutral grays (not series colors)

fig = plt.figure(figsize=(14.5, 9.0), facecolor=SURF)
ax = fig.add_axes((0.0, 0.40, 1.0, 0.60)); ax.set_xlim(0, 100); ax.set_ylim(16, 100); ax.axis("off")


def box(x, y, w, h, title, sub="", fc=FROZ, ec=FROZ_E, lw=1.0, fs=8.6, bold=False, tc=INK, hatch=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=lw,
                                hatch=hatch, zorder=2))
    if sub:
        ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal", zorder=3)
        ax.text(x + w / 2, y + h - 2.0 - fs * 0.75, sub, ha="center", va="top", fontsize=7, color=INK2,
                zorder=3, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal", zorder=3)


def arrow(x1, y1, x2, y2, col=INK2, lw=1.2, ls="-", rad=0.0, label=None, lx=None, ly=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=col, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        ax.text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2, label,
                fontsize=7, color=INK2, ha="center", va="center", zorder=4,
                bbox=dict(fc=SURF, ec="none", pad=0.5))


ax.text(1, 99, "阶段 0 · 用 GT 身份监督 LoRA,让 Qwen3-VL 第 24 层表征具备 WHICH(在 look-alike 中认出指令所指的车)",
        fontsize=12.5, fontweight="bold", color=INK, va="top")
ax.text(1, 94.6, "(a) 数据流与可训练部分。灰色 = 冻结;蓝 = grounding head(r16、r0 都训练);橙 = LoRA(仅 r16);青 = 训练时的 GT 监督(只用于离线训练,不进入部署决策)",
        fontsize=8.2, color=INK2, va="top")

# inputs
box(1, 70, 14, 15, "输入:一帧 RGB", "512×512,来自已采集\nmvp_full_v5 episode\n(train 42 / val 7 / test 10)", fc="#ffffff")
box(1, 50, 14, 15, "输入:语言指令", "去位置化后只保留身份\n“Follow the gold\nMini Cooper.”", fc="#ffffff")
box(1, 30, 14, 15, "候选框", "目标 + 干扰车的投影框\n(与闭环 harness 同一套)\n每帧打乱顺序", fc="#ffffff")

# Qwen stack
ax.add_patch(FancyBboxPatch((19, 26), 41, 62, boxstyle="round,pad=0.4,rounding_size=1.6", fc="none", ec=FROZ_E, lw=1.2,
                            ls="--", zorder=1))
ax.text(39.5, 86.8, "Qwen3-VL-4B-Instruct(bf16;prompt:文本在前、图像在后 → 图像 token 可注意到指令)",
        fontsize=7.6, color=INK2, ha="center", va="top")
box(21, 69, 16, 14, "视觉编码器", "ViT,冻结\n4.15 亿参数", fs=8.4)
box(21, 51, 16, 14, "token embedding", "冻结\n3.89 亿参数", fs=8.4)
box(40, 72, 18, 11, "解码层 0–15", "冻结 · 16.1 亿参数", fs=8.4)
box(40, 54, 18, 15.5, "解码层 16–23", "基座权重冻结 8.07 亿\n+ LoRA(q/k/v/o,rank 16)\n262 万可训练参数(仅 r16)", fc="#fde3d7",
    ec=LORA, lw=2.0, fs=8.4, bold=True)
box(40, 27, 18, 11, "解码层 24–35", "不执行(前向截断)\n12.1 亿参数", fc=UNUSED, ec=FROZ_E, fs=8.0, hatch="..", tc=MUTED)
arrow(15, 77.5, 21, 76)
arrow(15, 57.5, 21, 58)
arrow(37, 76, 40, 76); arrow(37, 58, 40, 74, rad=-0.2)
arrow(49, 72, 49, 69.4)
ax.text(49, 46.5, "读出点 = 第 23 层输出\n= hidden_states[24]\n(与部署时的特征层一致)", fontsize=7.2, color=INK2, ha="center", va="center")
arrow(49, 54, 49, 50.5)
ax.plot([49, 49], [42.8, 38.6], ls=":", color=MUTED, lw=1); ax.text(50.5, 41.2, "截断", fontsize=6.8, color=MUTED, va="center")

# pooling + head
box(63, 34, 16, 22, "候选框池化", "图像 token 网格 16×16×2560\n每个候选框内取均值\n→ 每车一个 2560 维向量\n(可微,梯度回传到 LoRA)", fc=FROZ)
box(63, 62, 16, 16, "文本向量", "非图像 token 的均值\n2560 维", fc=FROZ, fs=8.4)
arrow(58, 64, 63, 70); arrow(58, 58, 63, 47, rad=0.15)
arrow(15, 32, 71, 34, rad=0.16, col=MUTED, ls=(0, (3, 2)), label="框坐标", lx=40, ly=24.0)
box(82, 48, 17, 30, "grounding head", "LN(候选) → MLP 2560·256·1\n+ 双线性项\n<P·LN(候选), Q·LN(文本)>\n→ 每个候选一个分数\n132 万可训练参数\n(r16、r0 结构相同)",
    fc="#dbe8f8", ec=HEAD, lw=2.0, fs=8.6, bold=True)
arrow(79, 70, 82, 66); arrow(79, 45, 82, 56)
box(82, 24, 17, 19, "交叉熵损失", "对画面内全部候选做 softmax\n标签 = GT 目标所在位置\n(仅训练时使用 GT)", fc="#d6f1e6", ec=SUP, lw=2.0, fs=8.6, bold=True)
arrow(90.5, 48, 90.5, 43.4)
arrow(82, 28, 52, 53.5, col=LORA, lw=1.4, ls=(0, (4, 2)), rad=-0.25, label="梯度 → LoRA(仅 r16)", lx=68, ly=30.5)
arrow(96, 43, 96, 48, col=HEAD, lw=1.4, ls=(0, (4, 2)), label="梯度 → head", lx=96, ly=45.5)

# ---------------- (b) parameter bars
axb = fig.add_axes((0.15, 0.06, 0.43, 0.29), facecolor=SURF)
rows = [("解码层 24–35(截断,不执行)", 1_211_169_792, UNUSED, "冻结·不执行"),
        ("解码层 0–15", 1_614_893_056, FROZ, "冻结"),
        ("解码层 16–23 基座权重", 807_446_528, FROZ, "冻结"),
        ("视觉编码器", 415_347_712, FROZ, "冻结"),
        ("token embedding", 388_956_160, FROZ, "冻结"),
        ("LoRA(仅 r16)", 2_621_440, LORA, "可训练"),
        ("grounding head(r16、r0)", 1_321_473, HEAD, "可训练")]
for i, (name, n, col, tag) in enumerate(rows):
    axb.barh(i, n, color=col, edgecolor=FROZ_E if col in (FROZ, UNUSED) else col, height=0.62,
             hatch=".." if col == UNUSED else None)
    lab = f"{n / 1e8:.2f} 亿" if n >= 1e8 else f"{n / 1e4:.0f} 万"
    axb.text(n * 1.25, i, f"{lab} · {tag}", va="center", fontsize=7.6, color=INK if tag == "可训练" else INK2)
axb.set_yticks(range(len(rows))); axb.set_yticklabels([r[0] for r in rows], fontsize=8)
axb.set_xscale("log"); axb.set_xlim(5e5, 3e10); axb.invert_yaxis()
axb.set_xlabel("参数量(对数刻度)", fontsize=8, color=INK2)
axb.grid(axis="x", color="#ecebe7", lw=0.6); axb.set_axisbelow(True)
for s in ("top", "right"):
    axb.spines[s].set_visible(False)
for s in ("left", "bottom"):
    axb.spines[s].set_color(FROZ_E)
axb.tick_params(colors=INK2, labelsize=7.5)
axb.set_title("(b) 各部件参数量(实测):可训练部分只占约 0.1%(r16:394 万 / 44.4 亿)", fontsize=9, loc="left", color=INK)

# ---------------- (c) r16 vs r0
axc = fig.add_axes((0.62, 0.05, 0.36, 0.30)); axc.axis("off")
axc.set_title("(c) 两组实验:r0 是对照组,只有 r16 加 LoRA", fontsize=9, loc="left", color=INK)
cells = [["", "r16(LoRA)", "r0(对照)"],
         ["Qwen 解码层 16–23", "LoRA rank 16,可训练", "冻结"],
         ["grounding head", "可训练(132 万)", "可训练(132 万)"],
         ["可训练参数合计", "394 万", "132 万"],
         ["数据 / GT 监督 / 超参", "相同", "相同"],
         ["评测(val、test、子集、语言替换)", "相同", "相同"]]
tb = axc.table(cellText=cells, loc="upper left", cellLoc="center", colWidths=[0.42, 0.30, 0.28], bbox=(0, 0.22, 1, 0.74))
tb.auto_set_font_size(False); tb.set_fontsize(8)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#dddcd7"); cell.set_facecolor(SURF)
    if r == 0:
        cell.set_text_props(fontweight="bold", color=INK)
    if r == 1 and c == 1:
        cell.set_facecolor("#fde3d7")
    if r in (2,) and c in (1, 2):
        cell.set_facecolor("#dbe8f8")
axc.text(0, 0.12, "LoRA 本身的贡献 = r16 - r0(同一套 val/test 帧):\n若 r16 ≈ r0,说明冻结表征已含足够 WHICH 信息、改 Qwen 无益;\n若 r16 > r0 且 train/val 差距小,LoRA 值得进入阶段 1(闭环冷启动)。",
         fontsize=7.8, color=INK2, va="top", linespacing=1.45)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"fig_stage0_lora.{ext}"), dpi=180 if ext == "png" else None, facecolor=SURF)
print("wrote fig_stage0_lora.{png,pdf}")
