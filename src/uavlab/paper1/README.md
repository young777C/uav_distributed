# `uavlab.paper1`: lightweight task-level simulator (Paper 1)

This package implements a **standalone, lightweight** task-level simulation platform
that follows the models and loop decomposition in `paper1.pdf`:

- **Slow loop (GCS)**: §3.4 task-level planning / selection.
- **Fast loop (UAV)**: §3.5 local motion + communication FSM.
- **Models**: link state, key return, completion, energy (see `uavlab.related_models`).

Design goals:

- **No coupling** to the legacy `uavlab.tasks.taskA` environment.
- **No PX4/Gazebo** high-fidelity flight simulation.
- Deterministic, testable components with explicit inputs/outputs.

Entry points will live under `uavlab.paper1.runner` (to be added).

