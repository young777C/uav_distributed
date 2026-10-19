"""1006 §3.4: build the shared-opening reference table from a reference-opening experiment (Ours, protocol
sim_proxy_v2_openval@2, collect ticks): for every seed and opening tick (steps 0 / 6 / 12) the scene actor of the opening
pick (tick record `open` slot → `gt` actor index, −1 = target). Also reports the agreement of two reference runs.
  python -m acot.analysis.shared_opening <experiment.yaml> --arm ours --out acot/configs/shared_opening/<name>.json
      [--compare <experiment2.yaml> --compare-arm ours]"""
import argparse
import json
import os

from acot import ROOT
from acot.analysis.results import load
from acot.config.resolve import resolve_experiment

OPEN_STEPS = (0, 6, 12)


def table(exp, arm):
    D = load(resolve_experiment(exp))[arm]
    out, correct = {}, {}
    for seed, r in D.items():
        if r["status"] != "complete":
            continue
        row = {}
        for t in r["ticks"]:
            if t.get("t") in OPEN_STEPS and "open" in t and 0 <= t["open"] < len(t.get("gt", [])):
                row[str(t["t"])] = int(t["gt"][t["open"]])
        if len(row) == len(OPEN_STEPS):
            out[str(seed)] = row
            correct[str(seed)] = row["12"] == -1
    return out, correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--arm", default="ours")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare")
    ap.add_argument("--compare-arm", default="ours")
    a = ap.parse_args()
    tab, ok = table(a.experiment, a.arm)
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    json.dump(tab, open(os.path.join(ROOT, a.out), "w"), indent=1, sort_keys=True)
    rep = {"n": len(tab), "reference_correct_at_12": sum(ok.values())}
    if a.compare:
        tab2, _ = table(a.compare, a.compare_arm)
        common = sorted(set(tab) & set(tab2))
        rep["agreement_at_12"] = {"n": len(common), "same": sum(tab[s]["12"] == tab2[s]["12"] for s in common)}
    json.dump(rep, open(os.path.join(ROOT, a.out.replace(".json", ".report.json")), "w"), indent=1)
    json.dump({k: v for k, v in ok.items()}, open(os.path.join(ROOT, a.out.replace(".json", ".correct.json")), "w"),
              indent=1, sort_keys=True)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
