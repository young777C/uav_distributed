"""`acot eval --dry-run` (framework §9 L0): resolve the experiment and print, per arm, the resolved identity (config hash,
role, GT channels) and the invocation the scheduler will drive — the legacy-equivalent one (infra.runtime=legacy) or the
acot runtime server reading the resolved config (infra.runtime=acot; arms it does not implement are listed) — nothing
is started."""
from acot.config.legacy import shell_join, translate
from acot.config.resolve import resolve_experiment


def seed_blocks(seeds):
    """consecutive runs of the same town → [(town, [seed dicts])] (one CARLA client process per block)."""
    out = []
    for s in seeds:
        if out and out[-1][0] == s["town"] and out[-1][1][-1]["seed"] + 1 == s["seed"]:
            out[-1][1].append(s)
        else:
            out.append((s["town"], [s]))
    return out


def dry_run(a):
    from acot.cli import _overrides
    R = resolve_experiment(a.experiment, _overrides(a.overrides), allow_test=a.allow_test,
                           arms=a.arms.split(",") if a.arms else None)
    print(f"# {R['name']}: protocol {R['protocol_ref']}, {len(R['seeds'])} seeds ({R['seed_purpose']}), "
          f"collect {R['collect']}{'  [AD HOC]' if R['ad_hoc'] else ''}")
    img, net = R["infra"]["docker"]["policy_image"], R["infra"]["docker"]["network"]
    runtime = R["infra"].get("runtime") or "legacy"
    for name, arm in R["arms"].items():
        inv = translate(R, name)
        print(f"\n## arm {name} — method {arm['method']['name']}, role {arm['role']}, config_hash {arm['config_hash']}, "
              f"uses_gt {arm['uses_gt'] or '[]'}, runtime {runtime}")
        from acot.eval.scheduler import runtime_problems
        bad = runtime_problems({"arms": {name: arm}, "collect": R["collect"]}, runtime).get(name)
        if bad:
            print(f"  NOT RUNNABLE on the {runtime} runtime: " + "; ".join(bad))
            continue
        if runtime == "acot":
            print(f"docker run -d --network {net} --name {inv['worker']['container']} "
                  f"{R['infra']['docker']['policy_image_acot']} \\\n  python -m acot.runtime.server "
                  f"--spec <results>/{R['name']}/{name}/resolved_spec.json --worker w0 --port {R['infra']['policy_port']}")
            for town, block in seed_blocks(R["seeds"]):
                print(f"docker exec cyh-carla python /workspace/acot_client/episode.py --client <part>.client.json  "
                      f"# {town} seeds {block[0]['seed']}–{block[-1]['seed']}")
            continue
        envs = " ".join(f"-e {k}={v!r}" for k, v in inv["policy_env"].items() if v not in ("", "0", "0.9", "head"))
        print(f"docker run -d --network {net} --name {inv['worker']['container']} {envs} {img} \\\n"
              f"  bash -c \"pip install -q {inv['pip_preinstall']}; python carla_uav_tracking/rollout/policy_server.py "
              f"{shell_join(inv['policy_argv'])}\"")
        for town, block in seed_blocks(R["seeds"]):
            b = translate(R, name, seeds=block)
            print(f"docker exec cyh-carla python /workspace/rollout/run_rollout.py {shell_join(b['client']['argv'])}")
    return 0
