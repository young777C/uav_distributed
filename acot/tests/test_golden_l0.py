"""L0 invocation parity (framework §9): for every golden legacy launcher, the invocations captured with the fake docker
must be SEMANTICALLY IDENTICAL (argparse-parsed policy / client args, effective env incl. JSON kwargs merged with the
constructor defaults, collection presence) to what the translator produces from the matching acot experiment arm."""
import os

import pytest

from acot import CONFIGS
from acot.config.legacy import semantic, semantic_of, translate
from acot.config.resolve import resolve_experiment
from acot.tests.golden.harness import cleanup, invocations, run_launcher

EXP = os.path.join(CONFIGS, "experiments", "legacy")
V1 = '{"window":3,"ft":0.5}'
V3 = '{"window":3,"ft":0.5,"agg":"last","final_reset":true,"reset_asd":true}'
V4 = '{"window":3,"ft":0.5,"agg":"last","force_final":true,"seed_asd":3.0}'
# (case id, launcher, args, extra env, experiment, arm, expected [(seed_base, episodes, town)])
CASES = [
    ("chains_attr_a", "runs/opening_dump/chains_attr.sh", ["a"], {}, "chains_attr", "zoa", [(91000, 7, "Town05")]),
    ("chains_attr_c", "runs/opening_dump/chains_attr.sh", ["c"], {}, "chains_attr", "zoa", [(91014, 6, "Town05")]),
    ("openval_v1", "runs/opening_dump/run_openval.sh", ["v1", "0", "2050", "1", "93100", "140", "attr", V1], {},
     "openval", "v1", [(93100, 140, "Town05")]),
    ("openval_head", "runs/opening_dump/run_openval.sh", ["hd", "1", "2060", "1", "93100", "140", "head", ""], {},
     "openval", "head", [(93100, 140, "Town05")]),
    ("openval_v3", "runs/opening_dump/run_openval.sh", ["v3", "2", "2070", "2", "93380", "140", "attr", V3], {},
     "openval", "v3", [(93380, 140, "Town05")]),
    ("openval_v4", "runs/opening_dump/run_openval.sh", ["v4", "3", "2080", "2", "93520", "140", "attr", V4], {},
     "openval", "v4", [(93520, 140, "Town05")]),
    ("f_v4", "runs/opening_dump/f_v4.sh", [], {}, "f_v4", "zo4", [(91000 + 4 * i, 4, "Town05") for i in range(5)]),
    ("oracle_absent", "runs/opening_dump/f_oracle_absent.sh", [], {}, "oracle_absent", "zoo",
     [(91000 + 4 * i, 4, "Town05") for i in range(5)]),
    ("dagger_v2", "runs/tah_v90/collect_dagger.sh", [], {}, "dagger_v2", "dag",
     [(97000 + 15 * i, 15, t) for i, t in enumerate(["Town05"] * 4 + ["Town01", "Town02", "Town04"])]),
    ("ext_dam4sam", "carla_uav_tracking/rollout/eval_ext_dam4sam.sh", [], {}, "ext_dam4sam", "dam4sam",
     [(91000, 20, "Town05")]),
    ("ext_assoc", "carla_uav_tracking/rollout/eval_ext_assoc.sh", [], {}, "ext_assoc", ["motion", "deepsort"],
     [(91000, 20, "Town05"), (91000, 20, "Town05")]),
    ("launch_wd_resume", "runs/opening_dump/chains_attr.sh", ["b"], {"START_PART": "3"}, "chains_attr", "zoa",
     [(91007, 7, "Town05")]),
]
INFRA = {"image": "acot-uav-train:cu118", "network": "cyh-carla-net", "shm": "16g", "pip": "timm==1.0.29"}


def _block(R, base, n, town):
    seeds = [s for s in R["seeds"] if s["town"] == town and base <= s["seed"] < base + n]
    assert len(seeds) == n, (base, n, town, len(seeds))
    return seeds


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_l0_semantic_parity(case):
    cid, launcher, args, env, exp, arms, blocks = case
    recs, repo, out = run_launcher(launcher, args, env)
    try:
        inv = invocations(recs, repo)
        assert len(inv) == len(blocks), f"{cid}: captured {len(inv)} invocations\n{out[-2000:]}"
        R = resolve_experiment(os.path.join(EXP, f"{exp}.yaml"))
        arms = arms if isinstance(arms, list) else [arms] * len(blocks)
        if len(set(blocks)) == len(blocks):          # parallel launchers: match captured invocations to blocks by seeds
            key = {}
            for g in inv:
                a = g["client_argv"]
                key[(int(a[a.index("--seed-base") + 1]), int(a[a.index("--episodes") + 1]), a[a.index("--town") + 1])] = g
            assert sorted(key) == sorted(blocks), (cid, sorted(key))
            inv = [key[b] for b in blocks]
        for got, arm, (base, n, town) in zip(inv, arms, blocks):
            want = translate(R, arm, seeds=_block(R, base, n, town))
            g = semantic(got["policy_argv"], got["policy_env"], got["client_argv"], got["client_env"])
            w = semantic_of(want)
            assert g == w, f"{cid}/{arm}: legacy ≠ acot\n" + "\n".join(
                f"  {k}: legacy {g.get(k)!r}\n  {' ' * len(k)}  acot   {w.get(k)!r}" for k in sorted(set(g) | set(w))
                if g.get(k) != w.get(k))
            for k, v in INFRA.items():
                assert got["docker"][k] == v, (cid, k, got["docker"][k])
            assert got["docker"]["workdir"].endswith("uav-acot-track")
    finally:
        cleanup(repo)


def test_semantic_detects_a_drift():
    """the comparison is not vacuous: one changed knob (conf_tau) and one changed JSON key (absent_hold) are caught."""
    R = resolve_experiment(os.path.join(EXP, "chains_attr.yaml"))
    inv = translate(R, "zoa")
    base = semantic_of(inv)
    argv = list(inv["policy_argv"])
    argv[argv.index("--conf-tau") + 1] = "0.4"
    assert semantic(argv, inv["policy_env"], inv["client"]["argv"], inv["client"]["env"])["policy"] != base["policy"]
    env = dict(inv["policy_env"], ASD_CFG='{"filter_kind":"tempered"}')
    assert semantic(inv["policy_argv"], env, inv["client"]["argv"], inv["client"]["env"])["asd_kwargs"] != base["asd_kwargs"]


def test_dagger_dir_presence_semantics():
    """policy.py switches the grounding-DAgger buffer on with `DAGGER_DIR is not None` → an EMPTY value is not 'unset'
    (found by the P1 GT audit in the first smoke): the translator omits it and the normaliser tells them apart."""
    R = resolve_experiment(os.path.join(EXP, "chains_attr.yaml"))
    inv = translate(R, "zoa")
    assert "DAGGER_DIR" not in inv["policy_env"]
    env = dict(inv["policy_env"], DAGGER_DIR="")
    assert "DAGGER_DIR" in semantic(inv["policy_argv"], env, inv["client"]["argv"], inv["client"]["env"])["collect"]
    assert "DAGGER_DIR" not in semantic_of(inv)["collect"]
