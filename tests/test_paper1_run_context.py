from __future__ import annotations

from uavlab.common.config import load_resolved_config, _deep_merge
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.runner.run_context import paper1_run_context_label


def test_run_context_label_coupling_case():
    base = load_resolved_config("configs/base.yaml")
    case = load_resolved_config("configs/experiments/paper1/cases/c2_g2_m2.yaml")
    sys = load_resolved_config(
        "configs/experiments/paper1/system/coupling_periodic_goal.yaml"
    )
    cfg = apply_experiment_presets(_deep_merge(_deep_merge(base, case), sys))
    c = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    label = paper1_run_context_label(cfg, contract=c)
    assert "scene=g2_cluster_m2" in label
    assert "case=C2_G2_M2" in label
    assert "model=fdlc" in label
    assert "periodic_goal" in label
