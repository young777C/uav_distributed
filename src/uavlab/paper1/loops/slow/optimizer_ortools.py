from __future__ import annotations

from dataclasses import dataclass
from typing import List

from uavlab.paper1.loops.slow.planner import SlowWindow


@dataclass(frozen=True)
class SlowOptResult:
    sequence_poi_ids: List[int]
    objective_value: float
    status: str


def _require_ortools():
    try:
        from ortools.sat.python import cp_model  # type: ignore

        return cp_model
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "OR-Tools is required for slow-loop CP-SAT optimization. "
            "Install via `pip install ortools`."
        ) from e


def solve_short_horizon_tour(
    *,
    window: SlowWindow,
    horizon_h: int,
    lambda_q: float,
    mu_dist: float,
    # beta_ret 表示目标函数中数据回传适宜度的权重参数。
    # 在慢环排序优化目标中，通常将每个候选点“覆盖完成后是否具备有效同步/回传链路”作为评分项之一，
    # beta_ret 控制该项（数据回传适宜性）的贡献大小。三种架构对应不同权重，一般CDSL最大、WCDL适中、FDLC最小。
    # 其值越大，慢环规划排序时越倾向优先选择通信/回传条件更佳的POI，反之则弱化这方面约束，允许更灵活推进（尤其分布式时）。
    beta_ret: float,
    comm_objective: bool,
    energy_budget_constraint: bool,
    energy_objective_weight: float,
    energy_budget: float,
    time_limit_s: float = 0.5,
) -> SlowOptResult:
    """
    Short-horizon POI ordering on ``window`` (node 0 = UAV, 1..n = POIs).

    Optional-node ATSP: ``x[i,i]=1`` skips node i (self-loop). Depot 0 is active: ``x[0,0]=0``.
    """

    cp_model = _require_ortools()

    n_pois = len(window.poi_ids)
    n_nodes = 1 + n_pois
    if n_nodes <= 1:
        return SlowOptResult(sequence_poi_ids=[], objective_value=0.0, status="empty_window")

    H = int(horizon_h)
    visit_cap = n_pois if H <= 0 else min(n_pois, H)

    SCALE = 1000

    def cint(x: float) -> int:
        return int(round(float(x) * SCALE))

    model = cp_model.CpModel()
    x = [[model.NewBoolVar(f"x_{i}_{j}") for j in range(n_nodes)] for i in range(n_nodes)]

    for i in range(n_nodes):
        model.Add(sum(x[i][j] for j in range(n_nodes)) == 1)
    for j in range(n_nodes):
        model.Add(sum(x[i][j] for i in range(n_nodes)) == 1)

    arcs = [(i, j, x[i][j]) for i in range(n_nodes) for j in range(n_nodes)]
    model.AddCircuit(arcs)

    model.Add(x[0][0] == 0)

    visit: List = []
    for i in range(1, n_nodes):
        vi = model.NewBoolVar(f"v_{i}")
        model.Add(vi == 1 - x[i][i])
        visit.append(vi)
    model.Add(sum(visit) <= visit_cap)

    fly_terms = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            fly_terms.append(cint(window.e_fly_ij[i][j]) * x[i][j])
    hover_terms = [cint(window.e_hover_i[i]) * visit[i - 1] for i in range(1, n_nodes)]
    energy_lin = sum(fly_terms) + sum(hover_terms)

    if energy_budget_constraint:
        model.Add(energy_lin <= cint(max(0.0, float(energy_budget))))

    arc_obj_terms = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            coeff = float(mu_dist) * float(window.d_ij[i][j])
            if bool(comm_objective) and j >= 1:
                coeff -= float(lambda_q) * float(window.q_poi_i[j])
            if float(beta_ret) > 0.0 and j >= 1:
                coeff -= float(beta_ret) * float(window.r_ret_i[j])
            if float(energy_objective_weight) > 0.0:
                coeff += float(energy_objective_weight) * float(window.e_fly_ij[i][j])
            arc_obj_terms.append(cint(coeff) * x[i][j])

    hover_obj = []
    if float(energy_objective_weight) > 0.0:
        for i in range(1, n_nodes):
            hover_obj.append(cint(float(energy_objective_weight) * float(window.e_hover_i[i])) * visit[i - 1])

    model.Minimize(sum(arc_obj_terms) + sum(hover_obj))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    status = solver.Solve(model)
    st = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return SlowOptResult(sequence_poi_ids=[], objective_value=float("inf"), status=st)

    seq: List[int] = []
    cur = 0
    for _guard in range(n_nodes + 5):
        nxt = None
        for j in range(n_nodes):
            if int(solver.Value(x[cur][j])) == 1:
                nxt = j
                break
        if nxt is None:
            break
        if nxt == 0:
            break
        if 1 <= nxt < n_nodes:
            seq.append(int(window.poi_ids[nxt - 1]))
        cur = nxt
        if cur == 0:
            break

    obj_val = float(solver.ObjectiveValue()) / float(SCALE)
    return SlowOptResult(sequence_poi_ids=seq, objective_value=obj_val, status=st)
