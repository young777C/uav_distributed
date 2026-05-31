"""
Paper 1 implementation package.

This namespace intentionally stays independent from the legacy `uavlab.tasks.*`
environments to avoid any new/old-code conflicts.

Suggested layering:
- `uavlab.paper1.sim`: lightweight task-level environment + scene/config adapters
- `uavlab.paper1.loops`: fast/slow loop implementations
- `uavlab.paper1.runner`: CLI entry points compatible with sweep runner
"""

