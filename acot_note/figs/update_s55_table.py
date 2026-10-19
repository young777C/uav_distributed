"""Regenerate the §5.5 table rows in acot_note/1001_innovation-design.md from eval_out/deploy_compare.json."""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "../../carla_uav_tracking/rollout/eval_out/deploy_compare.json")))
rows = []
for a in J["arms"]:
    if a.get("pending"):
        rows.append(f"| {a['label']} | — | 运行中 | | | |"); continue
    g = {"pgif": "**ours**", "ours": "我们的", "external": "外部", "language": "语言基线"}[a["group"]]
    c, lc, lw = a["correct"], a["longC"], a["longW"]
    if a["stem"] == "tahrelm_pgif":
        d = "—"
    else:
        f = lambda k: f"{a['pgif_minus_' + k][0]:+.2f} [{a['pgif_minus_' + k][1]:+.2f}, {a['pgif_minus_' + k][2]:+.2f}]"
        d = f"{f('correct')} / {f('longC')} / {f('longW')}"
    name = f"**{a['label']}**" if a["stem"] == "tahrelm_pgif" else a["label"]
    rows.append(f"| {name} | {g} | {c[0]:.3f} [{c[1]:.2f}, {c[2]:.2f}] | {lc[0]:.1f} / {a['longC_med']:.1f} | {lw[0]:.1f} / {a['longW_med']:.1f} | {d} |")
p = os.path.join(HERE, "../1001_innovation-design.md"); s = open(p).read()
hdr = "| 方法 | 类别 | 正确锁定比例 ↑ [95% CI]"
i = s.index(hdr); j = s.index("\n", s.index("|---|", i)) + 1; k = s.index("\n\n", j)
s = s[:j] + "\n".join(rows) + s[k:]
open(p, "w").write(s); print("table updated; pending:", J["pending"])
