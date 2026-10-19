"""Association quality of the GT-free tracker (1004 deployable v2), scored OFFLINE against the simulator actor index.
Input: <ASD_LOG>.trk.jsonl written by policy.py (per S2 tick: tracker ids `tid` and simulator ids `gt` per candidate).
Per run: fragmentation (a GT actor's tracker id changes between consecutive observations), purity (share of a tracker
id's observations that belong to its majority actor), and target-specific: distinct ids given to the target per
episode, target id changes, and ticks where the target's current id previously belonged to another actor (swap).
  python trk_quality.py runs/asd/deploy_v2/<name>.trk.jsonl [...]"""
import collections
import json
import sys


def run(path):
    E = collections.defaultdict(list)
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:                       # truncated / NUL line (e.g. host reboot mid-write) → skip
            continue
        E[r["ep"]].append(r)
    frag = obs_pairs = 0; tgt_ids = []; tgt_changes = 0; tgt_swaps = 0; pur_num = pur_den = 0
    for recs in E.values():
        recs.sort(key=lambda r: r["k"])
        last = {}                                   # gt actor -> last tracker id
        owner = collections.defaultdict(collections.Counter)   # tracker id -> Counter(gt actor)
        seen_tgt = set()
        for r in recs:
            for t, g in zip(r["tid"], r["gt"]):
                owner[t][g] += 1
                if g in last:
                    obs_pairs += 1
                    if last[g] != t:
                        frag += 1
                        if g == -1:
                            tgt_changes += 1
                if g == -1:
                    seen_tgt.add(t)
                    if any(gg != -1 for gg in owner[t]):   # this id also carried another actor at some point
                        tgt_swaps += 1
                last[g] = t
        tgt_ids.append(len(seen_tgt))
        for t, cnt in owner.items():
            pur_num += cnt.most_common(1)[0][1]; pur_den += sum(cnt.values())
    n_ep = len(E)
    return {"episodes": n_ep, "frag_rate_per_obs": frag / max(obs_pairs, 1), "purity": pur_num / max(pur_den, 1),
            "target_ids_per_ep": sum(tgt_ids) / max(n_ep, 1), "target_id_changes_per_ep": tgt_changes / max(n_ep, 1),
            "target_ticks_on_shared_id": tgt_swaps}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in run(p).items()}))
