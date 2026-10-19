"""Closed-loop tracking metrics for PGIF vs baselines (CARLA, same seeds).
Episode-level (from eval_out/*.json, 10 Hz scorer frames): SR, correct-locked frames/seconds/fraction (per-frame pick =
target), committed-correct frames (K-hysteresis), longest wrong run, latch@2/5, id_switches, re-acquisition.
Tick-level (from runs/asd/*.jsonl, 1 tick = 6 frames = 0.6 s): LONGEST CORRECT RUN = consecutive ticks with the target
visible and the FINAL pick (PGIF output, or the raw memory pick when no correction) on the target; an out-of-view tick
ends the run (conservative).  python pgif_track_metrics.py --json eval_out/pgif_track_metrics.json"""
import argparse, json, os
import numpy as np

EO = os.path.join(os.path.dirname(__file__), "eval_out")
LG = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "asd")
ARMS = [  # (label, setting, eval json, jsonl, final-pick field)
    ("TAHRelM 无纠偏", "Town05·TAHRelM", "paper_asd_tah_observe", "tah_observe_91000", "mem"),
    ("TAHRelM + PGIF", "Town05·TAHRelM", "paper_asd_tah_tempered", "tah_tempered_91000", "anc"),
    ("真值漂移检测上界", "Town05·TAHRelM", "paper_ext_tahrelm_oracledrift", None, None),
    ("oracle 记忆上界", "Town05·TAHRelM", "paper_ext_tahrelm_gt", None, None),
    ("TAHRelM 无纠偏", "Town10HD·TAHRelM", "paper_asd_tah_observe_t10", "tah_observe_t10", "mem"),
    ("TAHRelM + PGIF", "Town10HD·TAHRelM", "paper_asd_tah_tempered_t10", "tah_tempered_t10", "anc"),
    ("reid 无纠偏", "Town05·reid", "paper_asd_reid_observe", "reid_observe_91000", "mem"),
    ("reid + PGIF", "Town05·reid", "paper_asd_reid_tempered", "reid_tempered_91000", "anc"),
]


def episodes(stem):
    d = json.load(open(os.path.join(EO, stem + ".json")))
    return [e for e in d["episodes"] if e.get("mis_follow_inst") is not None]


def ep_metrics(E):
    tf = np.array([e["track_frames"] for e in E], float)
    inst = np.array([e["mis_follow_inst"] for e in E]); sus = np.array([e["mis_follow_sustained"] for e in E])
    mw = np.array([e["max_wrong_run_s"] for e in E])
    att = np.array([e["reacquire_attempts"] for e in E], float)
    rs = np.array([e["reacquire_success_rate"] or 0.0 for e in E])
    return {"n": len(E), "SR": float(np.mean([bool(e["success"]) for e in E])),
            "SR_k": int(sum(bool(e["success"]) for e in E)),
            "correct_frac": float(np.mean(1 - inst)),
            "correct_frames_mean": float(np.mean(tf * (1 - inst))),
            "correct_s_mean": float(np.mean(tf * (1 - inst)) / 10),
            "committed_correct_frac": float(np.mean(1 - sus)),
            "track_s_mean": float(tf.mean() / 10),
            "max_wrong_s_mean": float(mw.mean()), "max_wrong_s_median": float(np.median(mw)),
            "latch2": float(np.mean(mw < 2)), "latch5": float(np.mean(mw < 5)),
            "id_switches": float(np.mean([e["id_switches"] for e in E])),
            "q_reacq": float((rs * att).sum() / att.sum()) if att.sum() else float("nan")}


def longest_correct(jsonl, field):
    p = os.path.join(LG, jsonl + ".jsonl")
    if not os.path.exists(p):
        return None
    R = [json.loads(l) for l in open(p)]
    eps = {}
    for r in R:
        eps.setdefault(r["ep"], []).append(r)
    best = []
    for S in eps.values():
        b = cur = 0; t0 = None
        for r in S:
            pick = r[field] if r[field] is not None else r["mem"]
            ok = r["true"] is not None and r["true"] >= 0 and pick == r["true"]
            if ok:
                t0 = r["t"] if cur == 0 else t0; cur += 1
                b = max(b, (r["t"] - t0) / 10 + 0.6)
            else:
                cur = 0
        best.append(b)
    best = np.array(best)
    return {"eps": len(best), "longest_correct_s_mean": float(best.mean()),
            "longest_correct_s_median": float(np.median(best)), "longest_correct_s_max": float(best.max())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    out = []
    for lab, st, stem, jl, fld in ARMS:
        m = ep_metrics(episodes(stem)); m.update({"arm": lab, "setting": st, "stem": stem})
        lc = longest_correct(jl, fld) if jl else None
        if lc:
            m.update(lc)
        out.append(m)
        print(f"{st:17s} {lab:14s} n={m['n']:2d} SR={m['SR']:.3f}({m['SR_k']}) correct={m['correct_frac']:.3f} "
              f"correct_s={m['correct_s_mean']:.1f}/{m['track_s_mean']:.1f}s commit={m['committed_correct_frac']:.3f} "
              f"longest_correct={m.get('longest_correct_s_mean', float('nan')):.1f}s(med {m.get('longest_correct_s_median', float('nan')):.1f}) "
              f"maxwrong={m['max_wrong_s_mean']:.1f}s(med {m['max_wrong_s_median']:.1f}) l2={m['latch2']:.2f} l5={m['latch5']:.2f} "
              f"idsw={m['id_switches']:.2f} qreacq={m['q_reacq']:.2f}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
