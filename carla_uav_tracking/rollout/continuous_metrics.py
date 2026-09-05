"""Continuous / segment metrics from existing rollout jsons — reveal method differences that the
binary latch-SR (one ≥2s wrong-window in a 90s episode → permanent fail; SR≈(1−q)^N) hides.

The latch SR saturates near 0 and can't discriminate methods (temporal reid had BETTER identity
yet SR dropped to 0 on one latched window). These metrics measure the SAME behavior with signal:
  track_correct_frac = 1 − mis_follow_inst   (per-frame: % of time on the right car)
  track_fraction     = track_seconds / duration
  q_reacq            = attempts-weighted per-event re-acquisition success (the 1−q in SR≈(1−q)^N)
  max_wrong_run_s    = the longest wrong window (what the latch thresholds at 2s) — graded
  latch@Ts           = frac(max_wrong_run_s < T): T=2 is the strict identity-latch, T→∞ removes it

Usage: continuous_metrics.py a.json b.json [c.json ...]   (paired by episode_seed vs the FIRST file)
"""
import json
import sys
import numpy as np


def load(path):
    d = json.load(open(path))
    return {int(e["episode_seed"]): e for e in d["episodes"]}


def arm_metrics(eps):
    seeds = sorted(eps)
    mis_inst = np.array([eps[s]["mis_follow_inst"] for s in seeds])
    trk_frac = np.array([eps[s]["track_seconds"] / max(eps[s]["duration_seconds"], 1e-6) for s in seeds])
    mwr = np.array([eps[s]["max_wrong_run_s"] for s in seeds])
    att = np.array([eps[s]["reacquire_attempts"] for s in seeds], float)
    qsucc = np.array([eps[s]["reacquire_success_rate"] for s in seeds])
    sr = np.array([1.0 if eps[s]["success"] else 0.0 for s in seeds])
    idsw = np.array([eps[s]["id_switches"] for s in seeds], float)
    q = float((qsucc * att).sum() / att.sum()) if att.sum() > 0 else float("nan")
    return {
        "SR (latch)": sr.mean(),
        "track_correct_frac (1−mis_inst)": (1 - mis_inst).mean(),
        "track_fraction (tracked/dur)": trk_frac.mean(),
        "q_reacq (per-event success)": q,
        "id_switches": idsw.mean(),
        "max_wrong_run_s (mean)": mwr.mean(),
        "max_wrong_run_s (median)": float(np.median(mwr)),
        "latch@2s (frac<2s)": (mwr < 2).mean(),
        "latch@3s": (mwr < 3).mean(),
        "latch@4s": (mwr < 4).mean(),
        "latch@5s": (mwr < 5).mean(),
    }, seeds


def main():
    paths = sys.argv[1:]
    arms = [(p.split("/")[-1].replace(".json", ""), load(p)) for p in paths]
    base_seeds = sorted(arms[0][1])
    common = [s for s in base_seeds if all(s in a[1] for a in arms)]
    print(f"paired seeds (n={len(common)}): {common}\n")
    rows = {}
    for name, eps in arms:
        sub = {s: eps[s] for s in common}
        rows[name], _ = arm_metrics(sub)
    keys = list(rows[list(rows)[0]].keys())
    w = max(len(k) for k in keys)
    hdr = "  ".join(f"{n:>12}" for n, _ in arms)
    print(f"{'metric':<{w}}  {hdr}")
    print("-" * (w + 2 + len(hdr)))
    for k in keys:
        vals = "  ".join(f"{rows[n][k]:>12.3f}" for n, _ in arms)
        print(f"{k:<{w}}  {vals}")
    # paired deltas vs first arm on the discriminating continuous metrics
    print("\nΔ vs base (first arm), paired:")
    base = arms[0][1]
    for name, eps in arms[1:]:
        dmis = np.mean([(1 - eps[s]["mis_follow_inst"]) - (1 - base[s]["mis_follow_inst"]) for s in common])
        dmwr = np.mean([eps[s]["max_wrong_run_s"] - base[s]["max_wrong_run_s"] for s in common])
        dtrk = np.mean([eps[s]["track_seconds"] - base[s]["track_seconds"] for s in common])
        print(f"  {name:>12}: Δtrack_correct_frac={dmis:+.3f}  Δmax_wrong_run_s={dmwr:+.2f}s  Δtrack_s={dtrk:+.1f}")


if __name__ == "__main__":
    main()
