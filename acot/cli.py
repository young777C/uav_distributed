"""python -m acot <command> … — the single entry point (framework §4).

  lint [--infra] [--strict-legacy]                 validate every config; legacy-surface coverage; legacy findings
  show <experiment.yaml> [--arm A] [--diff] [--legacy] [key=value …]
  eval <experiment.yaml> [--arms a,b] [--dry-run] [--allow-test] [key=value …]
  analyze <experiment.yaml> [--force]                 report (fails closed on incomplete results)
  audit-replay <experiment.yaml> --arm A --seed S --gpu G   validate the GT audit on a recorded episode
  replay-parity <experiment.yaml> --record <act_record.pkl> --gpu G [--arms a,b]   acot vs legacy runtime decisions
  results import-legacy <arm …>                       import legacy closed-loop results (deterministic precedence)
  artefacts import|ls|verify [names]                 artefact store (framework §10)
  data split <dataset ref> --name N --rule naive|town_heldout · data gen <datagen.yaml> [--dry-run] · data qc
  prep <prep.yaml> [--gpus 1,2] [--dry-run]           tah_seq / vlm_ctx / dagger caches (content-addressed)
  train <recipe.yaml> [--gpu G] [seed=N] [--dry-run]  tah_relm / opening_attr_lin / opening_attr_ft
  surface                                            regenerate acot_note/1004_legacy_surface.md
"""
import argparse
import json
import os
import sys

from acot import ROOT


def _overrides(items):
    from acot.config.load import parse_override
    return [parse_override(s) for s in items]


def cmd_lint(a):
    from acot import lint
    return lint.run(strict_legacy=a.strict_legacy, infra=a.infra)


def cmd_show(a):
    from acot.config.resolve import diff_vs_base, dump_yaml, resolve_experiment
    R = resolve_experiment(a.experiment, _overrides(a.overrides), allow_test=a.allow_test,
                           arms=a.arm.split(",") if a.arm else None)
    print(f"# experiment {R['name']} ({R['source']}) protocol {R['protocol_ref']}"
          f"{' [AD HOC]' if R['ad_hoc'] else ''}")
    towns = sorted({s['town'] for s in R['seeds']})
    print(f"# seeds: {len(R['seeds'])} × {towns} ({R['seed_purpose']}), hash {R['seeds_hash']}")
    for name, arm in R["arms"].items():
        print(f"\n## arm {name}: method {arm['method']['name']} (chain {' → '.join(arm['method_chain'])}) role "
              f"{arm['role']} config_hash {arm['config_hash']}")
        print(f"#   uses_gt: {arm['uses_gt'] or 'none (deployable under the protocol)'}; "
              f"privileged: {arm['protocol']['privileged_perception']}")
        if arm["override"]:
            print(f"#   override: {arm['override']}")
        if arm["provenance_gaps"]:
            print(f"#   accepted provenance gaps: {arm['provenance_gaps']}")
        if a.diff:
            for k, (b, m) in diff_vs_base(arm).items():
                print(f"#   {k}: {b} → {m}")
        elif not a.legacy:
            print(dump_yaml({"protocol": arm["protocol"], "method": arm["method"]}))
        if a.legacy:
            from acot.config.legacy import translate
            inv = translate(R, name, worker=None)
            print(json.dumps({k: v for k, v in inv.items() if k != "resolved"}, indent=1, ensure_ascii=False))
    return 0


def cmd_eval(a):
    if a.dry_run:
        from acot.eval.dry_run import dry_run
        return dry_run(a)
    from acot.eval.scheduler import run_experiment
    return run_experiment(a)


def cmd_results(a):
    from acot.analysis import legacy_import as L
    rep = {}
    for n in a.names:
        sc, st, warn = L.episodes(n)
        T, how = L.ticks(L.ticks_for(n), warn)
        files, _ = L.files_for(n)
        rep[n] = {"files": [os.path.relpath(f, ROOT) for f in files], "n_scored": len(sc),
                  "status": {k: sorted(s for s, v in st.items() if v["status"] == k) for k in ("complete", "skipped", "failed")},
                  "ticks": {"n": len(T), "alignment": sorted(set(how.values()))}, "warnings": warn,
                  "protocol_tag": L.arm_map().get(n, {}).get("protocol_tag", "unknown")}
        print(f"{n}: {len(sc)} scored, skipped {rep[n]['status']['skipped']}, ticks {len(T)} {sorted(set(how.values()))}"
              + (f", {len(warn)} warning(s)" if warn else ""))
    out = os.path.join(ROOT, "runs", "acot_logs", "import_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=1)
    print(f"wrote {os.path.relpath(out, ROOT)}")
    return 0


def cmd_artefacts(a):
    from acot import artefacts as A
    if a.action == "import":
        from acot.legacy.import_artefacts import import_all
        import_all(a.names or None)
        return 0
    ms = [m for m in A.ls(incomplete=True) if not a.names or m["name"] in a.names]
    bad = 0
    for m in ms:
        if m.get("incomplete"):
            print(f"{m['kind']:12s} {m['name']:26s} @{m['id']}  INCOMPLETE (no manifest — in progress or aborted)")
            continue
        line = f"{m['kind']:12s} {m['name']:26s} @{m['id']}  compat {json.dumps(m['compat'], ensure_ascii=False)}"
        if a.action == "verify":
            probs = A.verify(m)
            bad += bool(probs)
            line += "  OK" if not probs else f"  PROBLEMS {probs[:3]}"
        print(line)
    return 1 if bad else 0


def cmd_data(a):
    import yaml
    if a.action == "split":
        from acot.data.datasets import make_split
        params = {k: yaml.safe_load(v) for k, v in (x.split("=", 1) for x in a.param)}
        m, sp = make_split(a.spec, a.name, a.rule, params)
        print(f"split/{m['name']}@{m['id']}: " + ", ".join(f"{k} {len(v)}" for k, v in sp.items()))
        return 0
    spec = yaml.safe_load(open(a.spec))
    if a.action == "qc":
        from acot.data.gen import qc_dir
        q = qc_dir(spec, a.files)
        print(json.dumps(q, indent=1))
        return 0 if all(x["ok"] for x in q) else 1
    from acot.data.gen import run
    print(json.dumps(run(spec, dry_run=a.dry_run), indent=1, default=str)[:4000])
    return 0


def cmd_prep(a):
    import yaml
    spec = yaml.safe_load(open(a.spec))
    gpus = [int(g) for g in a.gpus.split(",")]
    if spec["kind"] == "tah_seq":
        from acot.prep import tah_seq
        if a.dry_run:
            _, _, inputs, compat, aid, d, _ = tah_seq.plan(spec)
            print(json.dumps({"id": aid, "dir": d, "inputs": inputs, "compat": compat}, indent=1))
            return 0
        tah_seq.build(spec, gpus)
    elif spec["kind"] == "vlm_ctx":
        from acot.prep import vlm_ctx
        r = vlm_ctx.build(spec, gpus, dry_run=a.dry_run)
        if a.dry_run:
            print(json.dumps(r, indent=1))
    elif spec["kind"] == "dagger":
        from acot.prep import merge
        merge.build(spec, gpus[0])
    else:
        raise SystemExit(f"unknown prep kind {spec['kind']}")
    return 0


def cmd_train(a):
    import yaml
    from acot.config.load import parse_override, set_dotted
    r = yaml.safe_load(open(a.recipe))
    for k, v in (parse_override(x) for x in a.overrides):
        set_dotted(r, k, v)
    if r["trainer"] == "tah_relm":
        from acot.train.tah_relm import train
        if a.dry_run:
            from acot import artefacts as A
            refs = r["inputs"] + list(r.get("val_inputs") or [])
            print(json.dumps({"inputs": r["inputs"], "val_inputs": r.get("val_inputs"),
                              "b1_problems": A.check_inputs(refs, waive=tuple(r.get("waivers") or {}))}, indent=1))
            return 0
        train(r, a.gpu, register_only=a.register_only)
    elif r["trainer"] in ("opening_attr_lin", "opening_attr_ft"):
        from acot.train.opening_attr import train
        res = train(r, a.gpu, dry_run=a.dry_run)
        if a.dry_run:
            print(json.dumps(res, indent=1))
    else:
        raise SystemExit(f"unknown trainer {r['trainer']}")
    return 0


def cmd_surface(a):
    from acot.tools import legacy_surface
    out = os.path.join(ROOT, "acot_note", "1004_legacy_surface.md")
    with open(out, "w") as f:
        sys_stdout, sys.stdout = sys.stdout, f
        try:
            legacy_surface.main()
        finally:
            sys.stdout = sys_stdout
    print(f"wrote {os.path.relpath(out, ROOT)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m acot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lint")
    p.add_argument("--infra", action="store_true")
    p.add_argument("--strict-legacy", action="store_true")
    p.set_defaults(fn=cmd_lint)
    p = sub.add_parser("show")
    p.add_argument("experiment")
    p.add_argument("--arm", default=None, help="comma-separated arm names (default: all)")
    p.add_argument("--diff", action="store_true", help="only the differences from methods/_base")
    p.add_argument("--legacy", action="store_true", help="the translated legacy invocation")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("overrides", nargs="*", help="key=value (infra.* free; anything else marks the run ad hoc)")
    p.set_defaults(fn=cmd_show)
    p = sub.add_parser("eval")
    p.add_argument("experiment")
    p.add_argument("--arms", default=None)
    p.add_argument("--seeds", default=None, help="restrict to these seeds, e.g. 91000-91001,91005")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--carla-ports", default=None, help="use these CARLA servers, e.g. 2050,2060")
    p.add_argument("--gpus", default=None, help="policy GPUs, e.g. 1,2")
    p.add_argument("overrides", nargs="*")
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("analyze")
    p.add_argument("experiment")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--force", action="store_true", help="report an INCOMPLETE experiment anyway (marked as such)")
    p.set_defaults(fn=lambda a: __import__("acot.analysis.report", fromlist=["cmd"]).cmd(a))
    p = sub.add_parser("audit-replay")
    p.add_argument("experiment")
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.set_defaults(fn=lambda a: __import__("acot.eval.scheduler", fromlist=["run_audit_replay"]).run_audit_replay(a))
    p = sub.add_parser("replay-parity")
    p.add_argument("experiment")
    p.add_argument("--record", required=True, help="act_record.pkl of a recorded episode (collect record_act)")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--arms", default=None)
    p.add_argument("--limit", type=int, default=0, help="replay only the first N act messages")
    p.add_argument("--out", default=None)
    p.add_argument("overrides", nargs="*")
    p.set_defaults(fn=lambda a: __import__("acot.eval.scheduler", fromlist=["run_replay_parity"]).run_replay_parity(a))
    p = sub.add_parser("results")
    p.add_argument("action", choices=["import-legacy"])
    p.add_argument("names", nargs="+", help="legacy arm names (acot/legacy/arm_map.yaml) or result stems")
    p.set_defaults(fn=cmd_results)
    p = sub.add_parser("artefacts")
    p.add_argument("action", choices=["import", "ls", "verify"])
    p.add_argument("names", nargs="*")
    p.set_defaults(fn=cmd_artefacts)
    p = sub.add_parser("data")
    p.add_argument("action", choices=["split", "gen", "qc"])
    p.add_argument("spec", help="split: <dataset ref>; gen: datagen yaml; qc: datagen yaml")
    p.add_argument("--name", default=None)
    p.add_argument("--rule", default="naive")
    p.add_argument("--param", nargs="*", default=[], help="split rule params key=value")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--files", nargs="*", default=[], help="qc: episode files")
    p.set_defaults(fn=cmd_data)
    p = sub.add_parser("prep")
    p.add_argument("spec")
    p.add_argument("--gpus", default="1")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_prep)
    p = sub.add_parser("train")
    p.add_argument("recipe")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--register-only", action="store_true",
                   help="register the outputs of a training run that finished after its host process died")
    p.add_argument("overrides", nargs="*", help="seed=N …")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("surface")
    p.set_defaults(fn=cmd_surface)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
