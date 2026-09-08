"""Control-theory framing of the offline→online gap — ZERO new rollouts, reuses eval_out JSONs.

Reframes the same per-episode observables we already log as CLOSED-LOOP STABILITY quantities of the
identity→control feedback loop ("bad frame → mis-ID → fly wrong car → worse frame" = positive feedback):

  error_persistence  = max_wrong_run_s (median)   # settling time: how long the loop stays diverged
  recovery_rate      = q_reacq                     # return-to-track probability per lost-lock event
  oscillation        = id_switches (mean)          # limit-cycle / chatter between look-alikes
  steady_state_track = track_correct_frac          # fraction on the right car at equilibrium

The damping story: undamped grounding (xattn) → + confidence gate / motion consensus (borrow#1) adds
DAMPING (shorter persistence, higher recovery) → consensus gate (borrow#2) OVER-DAMPS (suppresses correct
commits too → collapse). This script emits stability_control.json for the stability-plane figure and
prints the per-arm table. Arms whose json is absent are skipped (tahdag added once B1 lands).
"""
import json
import os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "eval_out")

# (file stem, short label, control-theory role) — order = the damping trajectory
ARMS = [
    ("paper_l1_xattn",              "xattn",        "undamped (language grounding)"),
    ("paper_l3_dino",              "reid",         "appearance re-ID, no gate"),
    ("paper_l4_dino_conf",         "reid+cΤ",      "+ confidence gate (damping)"),
    ("paper_l5_dino_conf_tavg",    "reid+cΤ+EMA",  "+ temporal EMA (filtering)"),
    ("paper_ext_reidmotion03",     "borrow#1",     "+ motion consensus (sweet-spot damping)"),
    ("paper_ext_reidmotion07",     "borrow#1@.7",  "motion gate too tight (over-constrained)"),
    ("paper_ext_reidmotion03_cons1.0", "borrow#2", "+ consensus GATE (OVER-damped → collapse)"),
    ("paper_ext_tah_gt",           "TAH",          "learned assoc, offline-trained"),
    ("paper_ext_tahdag_gt",        "TAH+DAgger",   "learned assoc, on-policy (B1)"),
    ("paper_ext_tahrelm_gt",       "TAHRelM+DAg",  "learned assoc + motion-consensus (A1)"),
]


def load(stem):
    p = os.path.join(ROOT, stem + ".json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {int(e["episode_seed"]): e for e in d["episodes"] if "mis_follow_inst" in e}


def arm_stability(eps, seeds):
    mwr = np.array([eps[s]["max_wrong_run_s"] for s in seeds])
    att = np.array([eps[s]["reacquire_attempts"] for s in seeds], float)
    qs = np.array([eps[s]["reacquire_success_rate"] for s in seeds])
    q = float((qs * att).sum() / att.sum()) if att.sum() > 0 else float("nan")
    return {
        "n": len(seeds),
        "steady_state_track": float(np.mean([1 - eps[s]["mis_follow_inst"] for s in seeds])),
        "recovery_rate_q": q,
        "error_persistence_med_s": float(np.median(mwr)),
        "error_persistence_mean_s": float(mwr.mean()),
        "oscillation_idsw": float(np.mean([eps[s]["id_switches"] for s in seeds])),
        "SR_latch": float(np.mean([1.0 if eps[s]["success"] else 0.0 for s in seeds])),
    }


def main():
    loaded = [(stem, lab, role, load(stem)) for stem, lab, role in ARMS]
    present = [(stem, lab, role, e) for stem, lab, role, e in loaded if e]
    missing = [lab for _, lab, _, e in loaded if e is None]
    # pair on the common scored seed set across all present arms (honest, same-seed)
    common = sorted(set.intersection(*[set(e) for _, _, _, e in present]))
    out = {"paired_seeds": common, "n_paired": len(common), "arms": []}
    for stem, lab, role, e in present:
        m = arm_stability(e, common)
        m.update({"stem": stem, "label": lab, "role": role})
        out["arms"].append(m)
    op = os.path.join(ROOT, "stability_control.json")
    json.dump(out, open(op, "w"), indent=2)
    # table
    print(f"paired scored seeds n={len(common)}"
          + (f"   [missing arms: {missing}]" if missing else "") + "\n")
    cols = ["steady_state_track", "recovery_rate_q", "error_persistence_med_s", "oscillation_idsw", "SR_latch"]
    hdr = f"{'arm':<14}{'role':<40}" + "".join(f"{c.split('_')[0][:7]:>9}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for a in out["arms"]:
        print(f"{a['label']:<14}{a['role']:<40}" + "".join(f"{a[c]:>9.3f}" for c in cols))
    print(f"\n[cols] steady=track_correct_frac  recover=q_reacq  error=persistence(med s, LOWER better)"
          f"  oscill=id_switches  SR=latch\n-> {op}")


if __name__ == "__main__":
    main()
