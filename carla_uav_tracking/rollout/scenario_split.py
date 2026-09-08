"""Scenario-conditioned advantage — ZERO new rollouts, reuses per-episode conditional buckets in eval_out.

Each episode logs mis_follow conditioned on scene buckets:
  mis_by_N       (× n_dist)    — # of in-frame distractors (2 / 3 / 5): look-alike DENSITY
  mis_by_central (× c_dist)    — target central vs OFF-center: OFF is the deployment-hard regime
  mis_by_depth   (× d_dist)    — target range bucket
Plus per-episode reacquire_* (loss→re-acq events) → split episodes by re-acquisition LOAD.

We frame-weight-aggregate track_correct (=1−mis) per bucket per arm on the common paired seeds, to find the
scenario where the learned module (TAH+DAgger) wins on MULTIPLE metrics — honestly, not by cherry-pick:
we report ALL buckets, and read where each method leads.

Usage: scenario_split.py  (arms hard-wired to the paper lineup)
"""
import json
import os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "eval_out")
ARMS = [
    ("l5",        "paper_l5_dino_conf_tavg"),
    ("borrow#1",  "paper_ext_reidmotion03"),
    ("DeepSORT",  "paper_ext_assoc_deepsort"),
    ("DAM4SAM",   "paper_ext_dam4sam"),
    ("TAH+DAgger", "paper_ext_tahdag_gt"),
    ("TAHRelM+DAg", "paper_ext_tahrelm_gt"),
]


def load(stem):
    p = os.path.join(ROOT, stem + ".json")
    if not os.path.exists(p):
        return None                                    # arm not yet run (e.g. A1 pending) → skipped
    d = json.load(open(p))
    return {int(e["episode_seed"]): e for e in d["episodes"] if "mis_by_N" in e}


def wmean_bucket(eps, seeds, mis_key, cnt_key, bucket):
    """frame-weighted track_correct (=1-mis) in `bucket` across seeds."""
    num = den = 0.0
    for s in seeds:
        e = eps[s]
        c = e[cnt_key].get(bucket, 0)
        if c and bucket in e[mis_key]:
            num += (1.0 - e[mis_key][bucket]) * c
            den += c
    return (num / den) if den else float("nan"), den


def main():
    loaded = [(lab, load(stem)) for lab, stem in ARMS]
    loaded = [(lab, e) for lab, e in loaded if e]      # drop arms whose json isn't present yet
    common = sorted(set.intersection(*[set(e) for _, e in loaded]))
    print(f"paired seeds n={len(common)}  arms={[l for l,_ in loaded]}\n")

    # ---- 1) look-alike density (mis_by_N) ----
    print("== track_correct by distractor density (frames) ==")
    Ns = ["2", "3", "5"]
    hdr = f"{'arm':<12}" + "".join(f"{'N='+n:>10}" for n in Ns)
    print(hdr); print("-"*len(hdr))
    for lab, eps in loaded:
        row = f"{lab:<12}"
        for n in Ns:
            v, _ = wmean_bucket(eps, common, "mis_by_N", "n_dist", n)
            row += f"{v:>10.3f}" if not np.isnan(v) else f"{'-':>10}"
        print(row)
    # frame counts (context)
    ctx = {n: int(sum(loaded[0][1][s]["n_dist"].get(n, 0) for s in common)) for n in Ns}
    print(f"  [frames: N=2:{ctx['2']}  N=3:{ctx['3']}  N=5:{ctx['5']}]")

    # ---- 2) central vs OFF-center (deployment-hard) ----
    print("\n== track_correct by target framing (central vs OFF-center = deployment-hard) ==")
    hdr = f"{'arm':<12}{'central':>12}{'OFF-center':>12}"
    print(hdr); print("-"*len(hdr))
    for lab, eps in loaded:
        vc, _ = wmean_bucket(eps, common, "mis_by_central", "c_dist", "central")
        vo, _ = wmean_bucket(eps, common, "mis_by_central", "c_dist", "off")
        print(f"{lab:<12}{vc:>12.3f}{vo:>12.3f}")

    # ---- 3) episodes split by re-acquisition LOAD (loss→reacq events) ----
    print("\n== re-acquisition scenario (episodes with reacquire_attempts >= 3) ==")
    base = loaded[0][1]
    hi = [s for s in common if base[s].get("reacquire_attempts", 0) >= 3]
    lo = [s for s in common if base[s].get("reacquire_attempts", 0) < 3]
    print(f"  hi-reacq seeds (n={len(hi)}): {hi}")
    print(f"  lo-reacq seeds (n={len(lo)}): {lo}")
    for tag, sub in [("HI-reacq", hi), ("LO-reacq", lo)]:
        if not sub:
            continue
        print(f"\n  [{tag}] arm: track_correct / max_wrong_mean_s / latch@4s / q_reacq")
        for lab, eps in loaded:
            tc = np.mean([1 - eps[s]["mis_follow_inst"] for s in sub])
            mw = np.mean([eps[s]["max_wrong_run_s"] for s in sub])
            l4 = np.mean([1.0 if eps[s]["max_wrong_run_s"] < 4 else 0.0 for s in sub])
            att = np.array([eps[s]["reacquire_attempts"] for s in sub], float)
            qs = np.array([eps[s]["reacquire_success_rate"] for s in sub])
            q = float((qs*att).sum()/att.sum()) if att.sum() > 0 else float("nan")
            print(f"    {lab:<12} {tc:>6.3f}  {mw:>6.2f}  {l4:>6.3f}  {q:>6.3f}")


if __name__ == "__main__":
    main()
