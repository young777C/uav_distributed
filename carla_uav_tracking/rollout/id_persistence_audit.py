"""Audit (2026-10-02): how much of PGIF's gain relies on the simulator's PERSISTENT actor ids surviving occlusion?
(1) Open-loop replay on observe logs (memory picks of TAHRelM w/o PGIF): PGIF-tempered fed the true persistent ids vs
    ids that are re-issued whenever an actor RE-ENTERS the candidate set (= a realistic tracker without cross-occlusion
    re-ID). Accuracy = output pick == true slot over target-present ticks; GT used only for scoring.
(2) Closed-loop PGIF log: corrective fires (memory wrong → PGIF output right) split by whether the target re-entered
    within the last R ticks.
  python id_persistence_audit.py"""
import collections
import json
import os

import numpy as np

from asd import ASD

LG = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "asd")


def eps(path):
    E = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line); E[r["ep"]].append(r)
    return [sorted(v, key=lambda r: r["t"]) for v in E.values()]


def reissue_on_reentry(seq, gap=6):
    """ids re-issued when an actor was absent at the previous S2 tick (or the previous tick is > gap steps ago)."""
    lab, nxt, prev_ids, prev_t, out = {}, 10_000, set(), None, []
    for r in seq:
        cont = prev_t is not None and r["t"] - prev_t <= gap
        ids = []
        for a in r["aidx"]:
            if a not in lab or not (cont and a in prev_ids):
                lab[a] = nxt; nxt += 1
            ids.append(lab[a])
        out.append(ids); prev_ids = set(r["aidx"]); prev_t = r["t"]
    return out


def replay(seq, ids_seq):
    f = ASD(mode="filter", filter_kind="tempered", resid_decay=0.98)
    f.reset(); hit = n = 0
    for r, ids in zip(seq, ids_seq):
        if r["mem"] is None or r["mem"] < 0:
            continue
        o = f.step(None, ids, r["mem"], mem_conf=r["conf"])
        out = o["slot"]
        if r["true"] is not None and r["true"] >= 0:
            n += 1; hit += out == r["true"]
    return hit, n


def main():
    print("## (1) open-loop replay: PGIF with persistent ids vs ids re-issued on re-entry")
    for name in ("tah_observe_91000", "tah_observe_t10", "reid_observe_91000"):
        p = os.path.join(LG, name + ".jsonl")
        if not os.path.exists(p):
            continue
        E = eps(p); mem = [0, 0]; per = [0, 0]; reent = [0, 0]
        for seq in E:
            for r in seq:
                if r["true"] is not None and r["true"] >= 0 and r["mem"] is not None:
                    mem[0] += r["mem"] == r["true"]; mem[1] += 1
            h, n = replay(seq, [r["aidx"] for r in seq]); per[0] += h; per[1] += n
            h, n = replay(seq, reissue_on_reentry(seq)); reent[0] += h; reent[1] += n
        a = lambda x: x[0] / max(x[1], 1)
        print(f"{name:20s} eps={len(E):2d} ticks={mem[1]} | memory raw {a(mem):.3f} | PGIF persistent ids {a(per):.3f} "
              f"(+{a(per)-a(mem):.3f}) | PGIF ids re-issued on re-entry {a(reent):.3f} (+{a(reent)-a(mem):.3f}) | "
              f"share of gain kept {((a(reent)-a(mem))/(a(per)-a(mem)) if a(per)>a(mem) else float('nan')):.2f}")
    print("\n## (2) closed-loop PGIF (deploy_all/tahrelm_pgif.jsonl): corrective vs harmful fires by target re-entry recency")
    p = os.path.join(LG, "deploy_all", "tahrelm_pgif.jsonl")
    cnt = collections.Counter()
    for seq in eps(p):
        last_seen_t, prev_t, absent_since = None, None, None
        for r in seq:
            tv = r["true"] is not None and r["true"] >= 0
            gap_before = (prev_t is None) or (r["t"] - prev_t > 6)
            if tv:
                if last_seen_t is None or gap_before or r["t"] - last_seen_t > 6:
                    reentry_t = r["t"]
                    cur_reentry = reentry_t
                last_seen_t = r["t"]
            prev_t = r["t"]
            if not tv or r.get("anc") is None:
                continue
            recent = (r["t"] - cur_reentry) <= 30          # within 5 S2 ticks (3 s) of the target (re)appearing
            first = cur_reentry == seq[0]["t"]              # the episode's first appearance is not a re-entry
            kind = "corrective" if (r["mem"] != r["true"] and r["anc"] == r["true"]) else \
                   ("harmful" if (r["mem"] == r["true"] and r["anc"] != r["true"]) else None)
            if kind:
                cnt[(kind, "after_reentry≤3s" if (recent and not first) else "steady")] += 1
    for k in ("corrective", "harmful"):
        a, b = cnt[(k, "after_reentry≤3s")], cnt[(k, "steady")]
        print(f"{k:10s} fires: {a+b:4d} | within 3 s after target re-entry {a:4d} ({a/max(a+b,1):.0%}) | steady tracking {b:4d}")


if __name__ == "__main__":
    main()
