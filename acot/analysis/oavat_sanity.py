"""1006 G9(b′) (acot_note/1006_t0_log.md D8): does the OA-VAT instance-prototype matching identify the target on this
task? Over the ticks of the OA-VAT default arm where the target is among the candidates, there are ≥ 2 candidates and
the head computed prototype similarities (trace `oavat_top`), the rate at which the most prototype-similar candidate
is the target (GT only for offline scoring). PASS iff the 95% Wilson lower bound of that rate exceeds the mean chance
level (1 / number of candidates) over the same ticks; otherwise the OA-VAT baseline is REMOVED (user decision).
  python -m acot.analysis.oavat_sanity acot/configs/experiments/t0_grid_devB.yaml --arm oavat_r0p70_t0p05"""
import argparse
import json

from acot.analysis.results import load
from acot.analysis.stats import wilson
from acot.config.resolve import resolve_experiment


def sanity(rows):
    hit = n = 0
    chance = 0.0
    by_mode = {}
    for r in rows.values():
        if r["status"] != "complete":
            continue
        for t in r["ticks"]:
            k = t.get("n", 0)
            if "oavat_top" not in t or t.get("true") is None or t["true"] < 0 or k < 2:
                continue
            ok = int(t["oavat_top"] == t["true"])
            hit, n, chance = hit + ok, n + 1, chance + 1.0 / k
            m = by_mode.setdefault(t.get("oavat_mode", "?"), [0, 0])
            m[0] += ok
            m[1] += 1
    lo, hi = wilson(hit, n)
    c = chance / n if n else float("nan")
    return {"n_ticks": n, "top1_rate": hit / n if n else None, "wilson95": [lo, hi], "mean_chance": c,
            "pass": bool(n and lo > c), "by_mode": {k: {"n": v[1], "rate": v[0] / v[1]} for k, v in by_mode.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--arm", default="oavat_r0p70_t0p05")
    a = ap.parse_args()
    D = load(resolve_experiment(a.experiment))
    print(json.dumps(sanity(D[a.arm]), indent=1))


if __name__ == "__main__":
    main()
