"""The ONE definition of every reported metric (framework §12.3). Episode metrics read the scorer summary
(rollout/metrics.py:RolloutScorer.finalize); the opening lock reads S2 tick records."""
import math

EPISODE = {   # name → (label, f(episode summary))
    "correct": ("正确锁定比例(承诺)", lambda e: 1 - e["mis_follow_sustained"]),          # primary
    "correct_l": ("正确比例(宽松·弃权算对)", lambda e: 1 - e["mis_follow_inst"]),
    "correct_s": ("正确比例(严格·弃权算错)", None),
    "longC": ("最长连续正确(s)", lambda e: e["max_correct_run_s"]),
    "longW": ("最长锁错(s)", lambda e: e["max_wrong_run_s"]),
    # 1006 §4: the policy's internal commitment (what WHERE follows) — acot runtime only, None otherwise
    "pc_correct": ("承诺正确比例(policy 承诺)", lambda e: e.get("pc_correct")),
    "tr90": ("固定时域正确跟踪率 TR90", lambda e: e.get("tr90")),
    "pc_longC": ("最长连续正确(policy 承诺,s)", lambda e: e.get("pc_max_correct_run_s")),
    "pc_longW": ("最长锁错(policy 承诺,s)", lambda e: e.get("pc_max_wrong_run_s")),
    "auc_sr": ("AUC-SR(τ ∈ [0.5, 20] s)", lambda e: auc_sr(e)),
}

SR_TAUS = (1.0, 2.0, 5.0, 10.0)
AUC_TAUS = tuple(0.5 + 0.1 * i for i in range(196))          # 0.5 … 20.0 s, step 0.1


def sr(e, tau):
    """SR(τ) of one episode: no policy-commitment wrong run ≥ τ seconds and not truncated by target loss (1006 §4.2)."""
    w = e.get("pc_max_wrong_run_s")
    if w is None:
        return None
    return float(w < tau and not e.get("truncated", False))


def auc_sr(e):
    """mean of SR(τ) over τ ∈ [0.5, 20] s (step 0.1) = the area under the per-episode SR curve, normalised."""
    v = [sr(e, t) for t in AUC_TAUS]
    return None if v[0] is None else sum(v) / len(v)


def strict(e):
    """per-frame pick == target over target-present frames; an abstention (no pick) counts as NOT correct."""
    nd, bn = e.get("n_dist") or {}, e.get("mis_by_N") or {}
    t = sum(nd.values())
    return 1 - sum(bn[k] * nd[k] for k in nd if k in bn) / t if t else float("nan")


def value(name, e):
    if name.startswith("sr@"):
        return sr(e, float(name[3:]))
    return strict(e) if name == "correct_s" else EPISODE[name][1](e)


def label(name):
    if name.startswith("sr@"):
        return f"SR(τ = {float(name[3:]):g} s)"
    return EPISODE[name][0] if name in EPISODE else name


# ------------------------------------------------------------------------------------------------------- opening lock
def out_id(r):
    """identity output of an ASD tick record: actor / tracklet id of PGIF's pick (None = absent / no pick)."""
    s = r.get("anc") if "anc" in r else r.get("mem")
    if s is None or s < 0 or s >= len(r["aidx"]):
        return None
    return r["aidx"][s]


def opening_lock(records, lock=12):
    """ASD tick records of one episode → True/False (identity output == target at the first S2 tick with the target
    among the candidates and t ≥ lock), None if no such tick. lock = s2_period × (opening window − 1)."""
    P = [r for r in records if r.get("true") is not None and r["true"] >= 0]
    rl = next((r for r in P if r["t"] >= lock), None)
    if rl is None:
        return None
    return out_id(rl) == rl["aidx"][rl["true"]]


def opening_lock_ticks(ticks, lock=12):
    """the same from acot wrapper tick records (ticks.jsonl: keys / true / pred) — for arms without ASD logs."""
    P = [r for r in ticks if r.get("true") is not None and r["true"] >= 0]
    rl = next((r for r in P if r["t"] >= lock), None)
    if rl is None:
        return None
    return rl["pred"] >= 0 and rl["keys"][rl["pred"]] == rl["keys"][rl["true"]]


def finite(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))
