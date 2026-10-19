"""L3 live parity (framework §9): the same seeds run through a LEGACY launcher (eval_asd.sh with DUMP_LABELS) and through
`acot eval` must agree on the seed → scenario mapping (target blueprint / colour, language, distractor set), on the skip
set, and on the scorer configuration; trajectories / metric values may differ (CARLA world history, DiT sampling).
  python -m acot.eval.l3_compare --arm-dir <exp>/<arm> --legacy-json <eval_out/x.json> --legacy-labels <dir>
P2 (acot runtime vs the P1 path, both `acot eval` runs): --arm-dir <exp A>/<arm> --other-arm-dir <exp B>/<arm> compares
the per-seed status / skip reason, the scenario fingerprints, the scorer parameters of the client echoes and the GT
access verdicts."""
import argparse
import glob
import json
import os


def fp_new(d):
    sc = json.load(open(os.path.join(d, "scenario.json")))
    return {"target_bp": sc.get("target_blueprint"), "target_color": sc.get("target_color"),
            "language": sc.get("language"),
            "distractors": [(m.get("bp"), m.get("color"), m.get("cls")) for m in (sc.get("distractors") or [])]}


def fp_legacy(path):
    d = json.load(open(path))
    return {"target_bp": d.get("target_bp"), "target_color": d.get("target_color"), "language": d.get("language"),
            "distractors": [(m.get("bp"), m.get("color"), m.get("cls")) for m in d.get("distractors", [])]}


def _echo(arm_dir):
    out = {}
    for p in sorted(glob.glob(os.path.join(arm_dir, "client_echo_*.json"))):
        e = json.load(open(p))
        out = {k: e.get(k) for k in ("scorer", "env", "fps", "resolution", "fov_deg", "steps", "blank", "town")}
    return out


def compare_runs(a_dir, b_dir):
    """two acot runs of the same arm configuration on the same seeds → {ok, rows, echo_equal}."""
    rows, ok = [], True
    sa = {int(os.path.basename(d)) for d in glob.glob(os.path.join(a_dir, "seeds", "*"))}
    sb = {int(os.path.basename(d)) for d in glob.glob(os.path.join(b_dir, "seeds", "*"))}
    for s in sorted(sa | sb):
        da, db = (os.path.join(x, "seeds", str(s)) for x in (a_dir, b_dir))
        st = [json.load(open(os.path.join(d, "status.json"))) if os.path.exists(os.path.join(d, "status.json")) else {}
              for d in (da, db)]
        fp = [fp_new(d) if os.path.exists(os.path.join(d, "scenario.json")) else None for d in (da, db)]
        ga = [json.load(open(os.path.join(d, "gt_access.json"))) if os.path.exists(os.path.join(d, "gt_access.json"))
              else None for d in (da, db)]
        row = {"seed": s, "status": [x.get("status") for x in st], "reason": [x.get("reason") for x in st],
               "scenario_equal": None if None in fp else fp[0] == fp[1],
               "gt_clean": [None if g is None else not (g["violations"] or g["unknown_sites"]) for g in ga]}
        row["status_equal"] = row["status"][0] == row["status"][1] and (
            row["status"][0] != "skipped" or row["reason"][0] == row["reason"][1])
        if all(x.get("status") == "complete" for x in st):
            eps = [json.load(open(os.path.join(d, "episode.json"))) for d in (da, db)]
            row["committed_correct"] = [round(1 - e["mis_follow_sustained"], 3) for e in eps]
        ok &= row["status_equal"] and row["scenario_equal"] is not False and False not in row["gt_clean"]
        rows.append(row)
    ea, eb = _echo(a_dir), _echo(b_dir)
    return {"ok": ok and ea == eb, "echo_equal": ea == eb, "echo_diff": {k: (ea.get(k), eb.get(k)) for k in
                                                                       set(ea) | set(eb) if ea.get(k) != eb.get(k)},
            "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", required=True)
    ap.add_argument("--legacy-json")
    ap.add_argument("--legacy-labels")
    ap.add_argument("--other-arm-dir", help="P2: a second acot run (e.g. legacy runtime) of the same arm and seeds")
    a = ap.parse_args()
    if a.other_arm_dir:
        r = compare_runs(a.arm_dir, a.other_arm_dir)
        print(json.dumps(r, indent=1))
        return 0 if r["ok"] else 1
    legacy = {int(e["episode_seed"]): e for e in json.load(open(a.legacy_json))["episodes"]}
    rows, ok = [], True
    for sd in sorted(glob.glob(os.path.join(a.arm_dir, "seeds", "*"))):
        s = int(os.path.basename(sd))
        st = json.load(open(os.path.join(sd, "status.json")))
        e = legacy.get(s)
        l_skip = None if e is None else bool(e.get("skipped"))
        n_skip = st["status"] == "skipped"
        lab = os.path.join(a.legacy_labels, f"{s}.json")
        same_fp = None
        if os.path.exists(lab) and os.path.exists(os.path.join(sd, "scenario.json")):
            same_fp = fp_new(sd) == fp_legacy(lab)
        row = {"seed": s, "new_status": st["status"], "legacy_skipped": l_skip, "skip_equal": l_skip == n_skip,
               "scenario_equal": same_fp}
        if e is not None and st["status"] == "complete":
            ne = json.load(open(os.path.join(sd, "episode.json")))
            row["committed_correct"] = {"legacy": 1 - e["mis_follow_sustained"], "acot": 1 - ne["mis_follow_sustained"]}
        ok &= bool(row["skip_equal"]) and same_fp is not False
        rows.append(row)
    print(json.dumps({"ok": ok, "rows": rows}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
