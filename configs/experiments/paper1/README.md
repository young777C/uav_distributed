# Layer-3 (Paper 1): experiments

This folder is organized to make scan experiments easy and non-redundant.

## Structure

- `system/`: **system-axis** definitions (struct/modeling/coupling/algorithm). No scene/comm here.
- `cases/`: **environment-axis** definitions (C×G×M) via `extends` comm profiles + `scene_file`.
- `runs/`: runnable entrypoints that combine **one** `system/*` + **one** `cases/*`.

Top-level `struct_*.yaml` (and other entry files) are convenience wrappers used by
existing sweep plans; they extend from `runs/`.

## Paper-1 preset keys: paper symbols vs legacy

`apply_experiment_presets` (`src/uavlab/experiments/presets.py`) normalizes legacy strings; prefer the **paper symbols** in new YAML.

| Axis | New key (preferred) | Legacy key → normalized |
|------|---------------------|-------------------------|
| `experiment.paper1.struct` | `cdsl` | `b1_centralized_single_loop` → `cdsl` |
| | `wcdl` | `b3_decouple_dual_loop` → `wcdl` |
| | `fdlc` | `full_architecture` → `fdlc` |
| `experiment.paper1.modeling` | `comm_aware_decision` | `task_comm` → `comm_aware_decision` |
| | `energy_aware_decision` | `task_energy` → `energy_aware_decision` |
| | `comm_energy_aware_decision` | `full_model` → `comm_energy_aware_decision` |
| `experiment.paper1.coupling` | `periodic_goal` | `no_feedback` → `periodic_goal` (in contract) |
| | `event_driven_goal` | `no_fast_switching` → `event_driven_goal` (in contract) |
| | `full_coupling` | (unchanged) |

**Main modeling sweep (§4 information utilization)** — `configs/sweeps/paper1_axis_grid_modelling.yaml`:

- `system/modelling_comm_aware_decision.yaml`
- `system/modelling_energy_aware_decision.yaml`
- `system/modelling_comm_energy_aware_decision.yaml`

Fixed when sweeping modeling: `struct=fdlc`, `coupling=full_coupling`. Each profile sets
`paper1_loops.semantics` (`use_comm_in_*` / `use_energy_in_*`) for **both** slow and fast loops.

**Main coupling sweep (§4.2.2 only)** — v2 (recommended):

- `configs/sweeps/paper1_axis_grid_coupling_v2.yaml` — `c2_g2_m2` + `c2_hard_g2_m2` × three profiles
- Smoke: `configs/sweeps/paper1_axis_grid_coupling_v2_smoke.yaml`
- Legacy v1 diag: `configs/sweeps/paper1_axis_grid_coupling_diag_c2g2.yaml`

System YAML:

- `system/coupling_periodic_goal.yaml` (Ts=200 via `experiment.sweep`, fixed 0.2)
- `system/coupling_event_driven_goal.yaml` (hybrid + P0, fixed 0.2)
- `system/coupling_full_coupling.yaml` (hybrid + P0, policy FSM)

**Appendix / engineering** (not in main v2 sweep): `system/appendix/` — includes `coupling_v2_ablation_*.yaml`. See `appendix/README.md`.

