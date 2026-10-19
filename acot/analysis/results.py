"""Read acot results (framework §8.3): <results_root>/<experiment>/<arm>/seeds/<seed>/{status,episode,scenario,
gt_access}.json + ticks.jsonl; legacy ASD logs (logs/asd_<tag>.jsonl) are keyed to seeds EXACTLY via logmap_<tag>.json."""
import glob
import json
import os

from acot import ROOT


def exp_dir(R):
    return os.path.join(ROOT, R["infra"]["results_root"], R["name"])


def _read(path, default=None):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return default


def asd_ticks(arm_dir):
    """{seed: ASD tick records sorted by t} from logs/asd_<tag>.jsonl + logmap_<tag>.json (asd_ep → seed)."""
    out = {}
    for lm in glob.glob(os.path.join(arm_dir, "logmap_*.json")):
        tag = os.path.basename(lm)[len("logmap_"):-len(".json")]
        ep2seed = {m["asd_ep"]: m["seed"] for m in _read(lm, []) if m.get("asd_ep") is not None}
        log = os.path.join(arm_dir, "logs", f"asd_{tag}.jsonl")
        if not os.path.exists(log):
            continue
        for line in open(log):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            s = ep2seed.get(r.get("ep"))
            if s is not None:
                out.setdefault(s, []).append(r)
    return {s: sorted(R, key=lambda r: r["t"]) for s, R in out.items()}


def load(R):
    """→ {arm: {seed: {status, reason, episode, scenario, gt_access, ticks, asd}}} for the experiment's seed list."""
    base, data = exp_dir(R), {}
    for arm in R["arms"]:
        d = os.path.join(base, arm)
        asd = asd_ticks(d)
        rows = {}
        for s in R["seeds"]:
            sd = os.path.join(d, "seeds", str(s["seed"]))
            st = _read(os.path.join(sd, "status.json"), {}) or {}
            ticks = []
            if os.path.exists(os.path.join(sd, "ticks.jsonl")):
                ticks = [json.loads(x) for x in open(os.path.join(sd, "ticks.jsonl")) if x.strip()]
            rows[s["seed"]] = {"status": st.get("status", "missing"), "reason": st.get("reason"),
                               "episode": _read(os.path.join(sd, "episode.json")),
                               "scenario": _read(os.path.join(sd, "scenario.json")),
                               "gt_access": _read(os.path.join(sd, "gt_access.json")),
                               "ticks": ticks, "asd": asd.get(s["seed"], [])}
        data[arm] = rows
    return data


def completeness(R, data):
    """fail-closed check (framework §12.1) → (ok, [problem rows])."""
    c = R["completeness"]
    allowed = set(c.get("allow_skipped_reasons") or []) if c.get("require") == "allow_skipped" else set()
    probs = []
    for arm, rows in data.items():
        for s, r in rows.items():
            if r["status"] == "complete":
                continue
            if r["status"] == "skipped" and (r["reason"] in allowed or "*" in allowed):
                continue
            probs.append({"arm": arm, "seed": s, "status": r["status"], "reason": r["reason"]})
    return not probs, probs
