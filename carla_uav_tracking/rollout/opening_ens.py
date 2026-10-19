"""Offline ensemble check (1004 opening fix): blend the LINEAR attribute heads (opening_cls.py → opening_cls_v2.json) and
the FINE-TUNED heads (opening_ft.py → opening_ft.json) per candidate, M = (1-a)·M_lin + a·M_ft, C likewise; report
opening accuracy (t=0 and +agg 0/6/12) on dev2 / dev for a grid of blends.   python opening_ens.py"""
import json

import numpy as np

from opening_cls import report

L = {(r["seed"]): r for r in json.load(open("eval_out/opening_cls_v2.json"))}
F = {(r["seed"]): r for r in json.load(open("eval_out/opening_ft.json"))}
for a_m, a_c in ((0, 0), (1, 1), (0.5, 0.5), (1, 0.5), (1, 0), (0.5, 0)):
    rows = []
    for s, r in L.items():
        if s not in F:
            continue
        rr = {"seed": s, "split": r["split"], "ticks": {}}
        for k, t in r["ticks"].items():
            f = F[s]["ticks"].get(k)
            if f is None:
                continue
            M = (1 - a_m) * np.array(t["M"]) + a_m * np.array(f["M"]); C = (1 - a_c) * np.array(t["C"]) + a_c * np.array(f["C"])
            rr["ticks"][int(k)] = dict(t, M=M.tolist(), C=C.tolist())
        rows.append(rr)
    print(f"\n==== blend: model a={a_m}, colour a={a_c} ====")
    report(rows)
