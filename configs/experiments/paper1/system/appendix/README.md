# Paper 1 — appendix / engineering coupling

Not part of the main **§4.2.2** coupling sweep (`configs/sweeps/paper1_axis_grid_coupling.yaml`, three profiles only).

These YAMLs replay **legacy** ablation stems (`coupling_no_*`, old `coupling_full` run directory names) for debugging or comparing against historical `runs/sweeps/test_couplling/` results.

| File | `experiment.paper1.coupling` | Notes |
|------|------------------------------|--------|
| `coupling_no_feedback.yaml` | `no_feedback` | Event replan + fast mode switch; no fast→slow feedback |
| `coupling_no_fast_switching.yaml` | `no_fast_switching` | Event-driven slow; fixed fast upload |
| `coupling_no_replan.yaml` | `no_replan` | `advance_on_poi_done`; blocks event replan |
| `coupling_v2_ablation_fixed05.yaml` | — | event + `fixed_send_ratio=0.5` (v1 对照) |
| `coupling_v2_ablation_periodic_ts40.yaml` | — | periodic at Ts=40, fixed 0.2 (v1 对照) |

Main §4.2.2 profiles live in `../coupling_periodic_goal.yaml`, `../coupling_event_driven_goal.yaml`, `../coupling_full_coupling.yaml` (coupling_v2 parameters).

Presets: `COUPLING_APPENDIX_PRESETS` in `src/uavlab/experiments/presets.py`.
