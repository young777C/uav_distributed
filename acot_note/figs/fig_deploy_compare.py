"""§5.5 全部模型 deployable 对比图(dataviz 方法):三个面板 = 三项客观指标(单一 y 轴/面板,不做双轴);
每臂一根条 + seed-bootstrap 95% CI;按"角色"配色(不按排名):slot2 橙 = TAHRelM+PGIF,slot1 蓝 = 我们的其他 WHICH 模块,
slot3 青 = 外部方法,中性灰 = 单帧语言基线;validate_palette "#2a78d6,#eb6834,#1baf7a" 全部 PASS(青色对比度 WARN →
每根条都有可见数值标注)。未完成的臂标"运行中"。数据:eval_out/deploy_compare.json(rollout/deploy_compare.py)。
  python acot_note/figs/fig_deploy_compare.py → fig_deploy_compare.{png,pdf}"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_f = os.path.expanduser("~/.local/share/fonts/SimHei.ttf")
if os.path.exists(_f):
    fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "pdf.fonttype": 42, "font.size": 8.5})
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#ecebe7"
COL = {"pgif": "#eb6834", "ours": "#2a78d6", "external": "#1baf7a", "language": "#b9b8b2"}
GRP = {"pgif": "TAHRelM + PGIF(ours)", "ours": "我们的其他 WHICH 模块", "external": "外部方法", "language": "单帧语言基线"}

J = json.load(open(os.path.join(HERE, "..", "..", "carla_uav_tracking", "rollout", "eval_out", "deploy_compare.json")))
arms = J["arms"]
PANELS = [("correct", "正确锁定比例(承诺口径)↑", "%.2f"), ("longC", "最长连续正确锁定(s,均值)↑", "%.1f"),
          ("longW", "最长锁错(s,均值)↓", "%.1f")]
fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), facecolor=SURF)
for ax, (k, title, fmt) in zip(axes, PANELS):
    ax.set_facecolor(SURF)
    top = max(a[k][2] for a in arms if not a.get("pending"))
    if k == "correct":   # the former lenient value (abstention counted correct) must still fit for the hollow marker
        top = max(top, max(a["correct_l"][0] for a in arms if not a.get("pending")))
    for i, a in enumerate(arms):
        if a.get("pending"):
            ax.text(i, top * 0.04, "运行中", rotation=90, ha="center", va="bottom", fontsize=7.5, color=MUTED)
            continue
        m, lo, hi = a[k]
        ax.bar(i, m, width=0.7, color=COL[a["group"]], edgecolor=SURF, linewidth=2)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=INK2, lw=1, capsize=2.5)
        ax.text(i, hi + top * 0.02, fmt % m, ha="center", va="bottom", fontsize=7.4, color=INK)
        if k == "correct" and abs(a["correct_l"][0] - m) > 0.05:   # abstaining tracker: show the old lenient value
            ml = a["correct_l"][0]
            ax.plot(i, ml, marker="D", ms=6.5, mfc=SURF, mec=INK2, mew=1.2, zorder=5)
            ax.annotate(f"原口径 {ml:.2f}\n(弃权帧算正确;\n严格口径 {a['correct_s'][0]:.2f})", (i, ml), xytext=(-10, 4),
                        textcoords="offset points", ha="right", va="bottom", fontsize=7.0, color=INK2)
    pg = next(a for a in arms if a["stem"] == "tahrelm_pgif")
    ax.axhline(pg[k][0], color=COL["pgif"], lw=1, ls=(0, (4, 3)), zorder=0)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([a["label"] for a in arms], rotation=35, ha="right", fontsize=7.6)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylim(0, top * 1.15); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=INK2, labelsize=7.6)
fig.legend(handles=[Patch(fc=COL[g], label=GRP[g]) for g in ("pgif", "ours", "external", "language")], loc="upper left",
           ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.0, 0.935))
fig.suptitle("全部 WHICH 模块在可部署设定下的对比(身份决策不用真值,候选框/定位为仿真理想输入;CARLA 在线闭环;同 19 个种子;\n误差线 = 种子 bootstrap 95% CI;"
             "橙色虚线 = PGIF 均值;正确锁定比例按承诺身份计,与另两项一致)",
             x=0.005, ha="left", fontsize=10.5, color=INK, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.915))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"fig_deploy_compare.{ext}"), dpi=180 if ext == "png" else None, facecolor=SURF)
print("wrote fig_deploy_compare; pending:", J["pending"])
