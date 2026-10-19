"""Does TAHRelM need retraining? (1004 opening fix follow-up) Error attribution of the FINAL model's closed-loop ticks
(v1 attribute opening + TAHRelM + PGIF tempered/absent-hold, deployable v2: tracker ids + occlusion-aware candidates),
dev 91000–19, vs the old setting V0 (ideal sim ids, all in-frame candidates). Per S2 tick:
  true = target slot among the candidates (−1 = target not a candidate: occluded / out of frame), mem = TAHRelM's raw
  pick, anc = PGIF output (committed pick; None/−1 = absent).
GT is used only to score.   python tah_need.py"""
import collections

import numpy as np

import opening_cl as oc

ARMS = {"v1 开局 + 部署 v2(最终模型)": ["zoa_pgif_trk_a", "zoa_pgif_trk_b", "zoa_pgif_trk_c"],
        "H-A(原开局)+ 部署 v2": ["v2h_pgif_trk"]}


def load_v0():
    import json, os
    LG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs", "asd", "deploy_all", "tahrelm_pgif.jsonl")
    return oc.load([(LG, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_out", "paper_dep_tahrelm_pgif.json"))])


def analyse(name, T):
    c = collections.Counter(); sw = collections.Counter()
    for s, R in T.items():
        R = [r for r in R if "true" in r and "mem" in r]
        prev_vis, prev_tid, since = False, None, 99
        for r in R:
            vis = r["true"] is not None and r["true"] >= 0
            anc = r.get("anc"); out = anc if anc is not None and anc >= 0 and anc < len(r["aidx"]) else None
            if not vis:
                c["tick_absent"] += 1
                c["absent_out_distractor" if out is not None else "absent_out_abstain"] += 1
                prev_vis = False; since = 0; continue
            tid = r["aidx"][r["true"]]
            phase = "reappear(≤5 tick)" if since <= 5 else "steady"
            if prev_vis and prev_tid is not None and tid != prev_tid:
                sw["target tracker-id switch"] += 1
            c[f"vis_{phase}"] += 1
            raw_ok = r["mem"] == r["true"]; out_ok = out == r["true"]
            c[f"raw_ok_{phase}"] += raw_ok; c[f"out_ok_{phase}"] += out_ok
            if not out_ok:
                if out is None:
                    k = "abstain while target visible"
                elif not raw_ok and out == r["mem"]:
                    k = "TAHRelM raw wrong, PGIF follows"
                elif raw_ok:
                    k = "TAHRelM raw RIGHT, PGIF overrides to wrong"
                else:
                    k = "both wrong (different)"
                c["err:" + k + f" [{phase}]"] += 1; c["err_total"] += 1
            prev_vis, prev_tid = True, tid; since += 1
    n_vis = c["vis_steady"] + c["vis_reappear(≤5 tick)"]
    print(f"\n## {name}: {len(T)} episodes, {n_vis + c['tick_absent']} S2 ticks")
    print(f"  target visible ticks {n_vis} | target NOT a candidate (occluded/out of frame) {c['tick_absent']} "
          f"→ output a distractor {c['absent_out_distractor']} / abstain {c['absent_out_abstain']}")
    for ph in ("steady", "reappear(≤5 tick)"):
        n = c[f"vis_{ph}"]
        if n:
            print(f"  [{ph:17s}] n={n:5d} | TAHRelM raw acc {c[f'raw_ok_{ph}'] / n:.3f} | committed (PGIF) acc {c[f'out_ok_{ph}'] / n:.3f}")
    print(f"  wrong committed ticks while target visible: {c['err_total']} — attribution:")
    for k, v in sorted(((k, v) for k, v in c.items() if k.startswith("err:")), key=lambda kv: -kv[1]):
        print(f"     {v:5d}  ({v / max(1, c['err_total']):.0%})  {k[4:]}")
    print(f"  target tracker-id switches between consecutive visible ticks: {sw['target tracker-id switch']}")
    wrong_all = c["err_total"] + c["absent_out_distractor"]
    print(f"  ALL wrong-identity ticks (visible-wrong + absent-but-committed-to-distractor) = {wrong_all}: "
          f"visible & TAHRelM raw wrong = {sum(v for k, v in c.items() if k.startswith('err:TAHRelM raw wrong') or k.startswith('err:both'))} "
          f"| target absent → distractor = {c['absent_out_distractor']}")


def main():
    for name, stems in ARMS.items():
        T = {}
        for s in stems:
            T.update(oc.load(oc.pairs_for(s)))
        analyse(name, T)
    try:
        analyse("V0 旧设定(理想仿真 ID,全部画面内候选，原开局)", load_v0())
    except Exception as e:
        print("V0 not available:", e)


if __name__ == "__main__":
    main()
