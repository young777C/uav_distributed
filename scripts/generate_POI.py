from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml



Point = Tuple[float, float]

_bh_extra_loss_cached: float | None = None


def _blackhole_extra_loss_from_c2_yaml() -> float:
    """Match `configs/comm_profiles/c2.yaml` → comm.blackhole_extra_loss for POI q̂ proxy."""
    global _bh_extra_loss_cached
    if _bh_extra_loss_cached is None:
        root = Path(__file__).resolve().parents[1]
        p = root / "configs" / "comm_profiles" / "c2.yaml"
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        _bh_extra_loss_cached = float((data.get("comm") or {}).get("blackhole_extra_loss", 0.48))
    return _bh_extra_loss_cached


# =============================================================================
# Tunable metrics (nearest-neighbor distance family)
# =============================================================================
#
# For a point set P = {p_1,...,p_N}:
#   d_i^nn = min_{j!=i} ||p_i - p_j||
#   \bar d^nn = (1/N) sum_i d_i^nn
# Let A be the area of the region, define the normalized uniformity index
#   U = \bar d^nn / (0.5 * sqrt(A / N)).
#
# Interpretation:
# - U ≈ 1 : close to random-uniform
# - U < 1 : more clustered
# - U > 1 : more regular than random
#
# G1: stratified jittered grid — even spatial coverage with bounded randomness (not i.i.d. uniform).
G1_GRID_JITTER_FRAC: float = 0.38  # max offset as fraction of cell half-extent (within cell)

# G2: cap full-set U so scatter resampling can introduce visible non-uniformity vs G1 grid
U_G2_MAX: float = 0.94
NN_MIN_DIST_M: float = 10.0  # hard minimum nearest-neighbor distance to avoid duplicates

# G2: multi-site bias — more clustered mass than G1 for clearer contrast
G2_N_MICRO_MIN: int = 4
G2_N_MICRO_MAX: int = 6
G2_MICRO_RING_R_M: float = 58.0  # spread micro-sites across more of the map
G2_MICRO_DISK_R_M: float = 17.0  # slightly tighter local disks vs G1 cell spacing
G2_MICRO_ANG_JITTER_RAD: float = 0.45
G2_N_CLUSTER_POI: int = 52  # more POIs in sites + fewer scatter → stronger “cluster” read vs G1





@dataclass(frozen=True)
class ConflictThresholds:
    qstrong: float = 0.7  # q_strong in Eq. (50)

    # Eq. (49)–(51) + thresholds listed under M0/M1/M2 in paper1.pdf §5.2.3 (iii)
    m0_qc_min: float = 0.7
    m0_eta_min: float = 0.7

    m1_qc_min: float = 0.4
    m1_qc_max: float = 0.7
    m1_eta_min: float = 0.3
    m1_eta_max: float = 0.7

    m2_qc_max: float = 0.4
    m2_eta_max: float = 0.3


def _nn_uniformity_index(points: List[Point], bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Compute (\bar d^nn, U) as described in the nearest-neighbor metric snippet.
    Returns (dbar_nn_m, U).
    """

    n = len(points)
    if n <= 1:
        return 0.0, 0.0
    dmins: List[float] = []
    for i, (xi, yi) in enumerate(points):
        best = float("inf")
        for j, (xj, yj) in enumerate(points):
            if i == j:
                continue
            d = math.hypot(xi - xj, yi - yj)
            if d < best:
                best = d
        dmins.append(float(best))
    dbar = float(sum(dmins) / n)
    n_min, n_max, e_min, e_max = bounds
    area = float((n_max - n_min) * (e_max - e_min))
    ref = 0.5 * math.sqrt(area / float(n))
    u = float(dbar / (ref + 1e-9))
    return dbar, u


def _min_nn_dist(points: List[Point]) -> float:
    if len(points) <= 1:
        return 0.0
    best_all = float("inf")
    for i, (xi, yi) in enumerate(points):
        best = float("inf")
        for j, (xj, yj) in enumerate(points):
            if i == j:
                continue
            d = math.hypot(xi - xj, yi - yj)
            if d < best:
                best = d
        best_all = min(best_all, best)
    return float(best_all)


def load_yaml(p: Path) -> Dict[str, Any]:
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be dict: {p}")
    return data


def dump_scene(path: Path, *, poi_list: List[Point], header: str) -> None:
    # Write a minimal scene YAML that extends the shared geometry base.
    doc: Dict[str, Any] = {
        # keep comment header as a YAML comment string at the top
        "_header": header,
        "extends": "./base.yaml",
        "poi_list": [[float(round(x, 1)), float(round(y, 1))] for x, y in poi_list],
    }

    # custom dump to preserve header as comments (remove the _header key from YAML)
    header_lines = [f"# {line}".rstrip() for line in header.strip().splitlines()]
    yaml_body = yaml.safe_dump(
        {k: v for k, v in doc.items() if k != "_header"},
        sort_keys=False,
        allow_unicode=True,
    ).rstrip() + "\n"
    path.write_text("\n".join(header_lines) + "\n" + yaml_body, encoding="utf-8")


def in_any_circle(pt: Point, circles: List[List[float]]) -> bool:
    x, y = pt
    for cx, cy, r in circles:
        if (x - cx) ** 2 + (y - cy) ** 2 <= (r ** 2):
            return True
    return False


def qhat_at_point(
    *,
    pt: Point,
    gcs: Point,
    blackholes: List[List[float]],
    d_max: float,
) -> float:
    """
    A lightweight proxy for paper's \hat q(p) used in Eq. (50).

    We map link loss to a [0,1] quality score:
      qhat = 1 - loss_p
    where loss_p follows the paper's link model structure:
      - smooth distance effect via normalized distance
      - + blackhole penalty inside degradation zones
    """

    x, y = pt
    gx, gy = gcs
    d = float(math.hypot(x - gx, y - gy))
    d_bar = min(d / max(d_max, 1e-9), 1.0)

    # smooth distance-decay loss (keep within [0.05, 0.60])
    loss_min, loss_max = 0.05, 0.60
    loss = loss_min + (loss_max - loss_min) * (d_bar ** 1.0)

    # blackhole penalty (F2-style) inside degradation zones (aligned with c2.yaml)
    if in_any_circle((x, y), blackholes):
        loss = min(1.0, loss + _blackhole_extra_loss_from_c2_yaml())

    loss = max(0.0, min(loss, 1.0))
    return float(max(0.0, min(1.0 - loss, 1.0)))


def compute_cluster_conflict(
    *,
    cluster_points: List[Point],
    gcs: Point,
    blackholes: List[List[float]],
    d_max: float,
    thr: ConflictThresholds,
) -> Tuple[float, float]:
    """
    Paper1 §5.2.3 (iii):
      Qbar_C = mean_i \hat q_poi_i
      eta_C  = mean_i 1(p_i in Omega_strong)
      Omega_strong = { p | \hat q(p) >= qstrong }
    """

    q_vals = [qhat_at_point(pt=p, gcs=gcs, blackholes=blackholes, d_max=d_max) for p in cluster_points]
    qc = float(sum(q_vals) / max(1, len(q_vals)))
    eta = float(sum(1.0 for q in q_vals if q >= thr.qstrong) / max(1, len(q_vals)))
    return qc, eta


def meets_level(level: str, qc: float, eta: float, thr: ConflictThresholds) -> bool:
    if level == "M0":
        return bool(qc >= thr.m0_qc_min and eta >= thr.m0_eta_min)
    if level == "M1":
        return bool(thr.m1_qc_min <= qc < thr.m1_qc_max and thr.m1_eta_min <= eta < thr.m1_eta_max)
    if level == "M2":
        return bool(qc < thr.m2_qc_max and eta < thr.m2_eta_max)
    raise ValueError(level)


def sample_uniform_points(
    *,
    rng: random.Random,
    n: int,
    bounds: Tuple[float, float, float, float],
    margin: float = 10.0,
) -> List[Point]:
    n_min, n_max, e_min, e_max = bounds
    pts: List[Point] = []
    for _ in range(n):
        x = rng.uniform(n_min + margin, n_max - margin)
        y = rng.uniform(e_min + margin, e_max - margin)
        pts.append((x, y))
    return pts


def _grid_dims_for_n(n: int, width: float, height: float) -> Tuple[int, int]:
    """gn*ge >= n with aspect ratio roughly width/height."""
    aspect = width / max(height, 1e-9)
    gn = max(1, int(round(math.sqrt(n * aspect))))
    ge = int(math.ceil(n / gn))
    while gn * ge < n:
        ge += 1
    return gn, ge


def sample_stratified_jittered_grid(
    *,
    rng: random.Random,
    n: int,
    bounds: Tuple[float, float, float, float],
    margin: float,
    jitter_frac: float,
) -> List[Point]:
    """
    One POI per grid cell (stratification) + independent uniform jitter inside each cell.
    Covers the map evenly like an “average” layout while remaining stochastic.
    """
    n_min, n_max, e_min, e_max = bounds
    w = n_max - n_min - 2.0 * margin
    h = e_max - e_min - 2.0 * margin
    gn, ge = _grid_dims_for_n(n, w, h)
    cell_w = w / gn
    cell_h = h / ge
    cells = [(i, j) for j in range(ge) for i in range(gn)]
    rng.shuffle(cells)
    pts: List[Point] = []
    for (i, j) in cells[:n]:
        cx = n_min + margin + (i + 0.5) * cell_w
        cy = e_min + margin + (j + 0.5) * cell_h
        half_jx = jitter_frac * 0.5 * cell_w
        half_jy = jitter_frac * 0.5 * cell_h
        x = min(max(cx + rng.uniform(-half_jx, half_jx), n_min + margin), n_max - margin)
        y = min(max(cy + rng.uniform(-half_jy, half_jy), e_min + margin), e_max - margin)
        pts.append((x, y))
    return pts


def _split_counts(total: int, k: int) -> List[int]:
    """Split `total` into k nonnegative integers that sum to `total` (as even as possible)."""
    if k <= 0:
        raise ValueError("k must be positive")
    q, r = divmod(total, k)
    return [q + (1 if i < r else 0) for i in range(k)]


def micro_centers_around_anchor(
    rng: random.Random,
    anchor: Point,
    k: int,
    *,
    ring_r: float,
    ang_jitter: float,
    bounds: Tuple[float, float, float, float],
    margin: float,
) -> List[Point]:
    """Place k micro-cluster centers on a ring around `anchor`, with angular jitter."""
    n_min, n_max, e_min, e_max = bounds
    ax, ay = anchor
    base_phase = rng.uniform(0.0, 2.0 * math.pi)
    out: List[Point] = []
    for i in range(k):
        ang = base_phase + (2.0 * math.pi * i / k) + rng.uniform(-ang_jitter, ang_jitter)
        rad = ring_r * rng.uniform(0.82, 1.03)
        mx = ax + rad * math.cos(ang)
        my = ay + rad * math.sin(ang)
        mx = min(max(mx, n_min + margin), n_max - margin)
        my = min(max(my, e_min + margin), e_max - margin)
        out.append((mx, my))
    return out


def _nearest_center_dist_m(pt: Point, centers: List[Point]) -> float:
    x, y = pt
    return min(math.hypot(x - cx, y - cy) for cx, cy in centers)


def sample_cluster_points(
    *,
    rng: random.Random,
    center: Point,
    n: int,
    sigma_m: float,
    bounds: Tuple[float, float, float, float],
    margin: float = 10.0,
) -> List[Point]:
    n_min, n_max, e_min, e_max = bounds
    pts: List[Point] = []
    cx, cy = center
    for _ in range(n):
        for _try in range(2000):
            x = rng.gauss(cx, sigma_m)
            y = rng.gauss(cy, sigma_m)
            if (n_min + margin) <= x <= (n_max - margin) and (e_min + margin) <= y <= (e_max - margin):
                pts.append((x, y))
                break
        else:
            raise RuntimeError("Failed to sample cluster point within bounds")
    return pts


def sample_uniform_disk_points(
    *,
    rng: random.Random,
    center: Point,
    n: int,
    disk_r_m: float,
    bounds: Tuple[float, float, float, float],
    margin: float = 10.0,
) -> List[Point]:
    """Uniform draws inside a disk (no central peak); milder visual clustering than Gaussian."""
    n_min, n_max, e_min, e_max = bounds
    cx, cy = center
    pts: List[Point] = []
    attempts = max(8000, n * 500)
    while len(pts) < n and attempts > 0:
        attempts -= 1
        ang = rng.uniform(0.0, 2.0 * math.pi)
        rad = disk_r_m * math.sqrt(rng.random())
        x = cx + rad * math.cos(ang)
        y = cy + rad * math.sin(ang)
        if (n_min + margin) <= x <= (n_max - margin) and (e_min + margin) <= y <= (e_max - margin):
            pts.append((x, y))
    short = n - len(pts)
    if short > 0:
        pts.extend(
            sample_cluster_points(
                rng=rng,
                center=center,
                n=short,
                sigma_m=disk_r_m * 0.5,
                bounds=bounds,
                margin=margin,
            )
        )
    return pts


def choose_micro_layout_for_level(
    *,
    level: str,
    rng: random.Random,
    bounds: Tuple[float, float, float, float],
    gcs: Point,
    blackholes: List[List[float]],
    d_max: float,
    thr: ConflictThresholds,
    cluster_disk_r_m: float,
    n_cluster: int,
    k_min: int,
    k_max: int,
    ring_r: float,
) -> Tuple[List[Point], List[Point], float, float]:
    """
    Search anchor + ring-placed micro-centers; pool all micro-cluster samples and
    evaluate (Qbar_C, eta_C) on that pooled set for M0/M1/M2 acceptance.

    Returns (micro_centers, pooled_cluster_points, qc, eta); the pooled list is the
    accepted cluster geometry (not resampled) so header metrics match the YAML.
    """
    n_min, n_max, e_min, e_max = bounds
    for _ in range(14000):
        k = rng.randint(k_min, k_max)
        ax = rng.uniform(n_min + 28.0, n_max - 28.0)
        ay = rng.uniform(e_min + 28.0, e_max - 28.0)
        anchor = (ax, ay)
        micro = micro_centers_around_anchor(
            rng,
            anchor,
            k,
            ring_r=ring_r,
            ang_jitter=G2_MICRO_ANG_JITTER_RAD,
            bounds=bounds,
            margin=16.0,
        )
        counts = _split_counts(n_cluster, k)
        pooled: List[Point] = []
        for center, ni in zip(micro, counts):
            local_rng = random.Random(rng.random())
            pooled.extend(
                sample_uniform_disk_points(
                    rng=local_rng,
                    center=center,
                    n=ni,
                    disk_r_m=cluster_disk_r_m,
                    bounds=bounds,
                    margin=12.0,
                )
            )
        qc, eta = compute_cluster_conflict(
            cluster_points=pooled,
            gcs=gcs,
            blackholes=blackholes,
            d_max=d_max,
            thr=thr,
        )
        if meets_level(level, qc, eta, thr):
            return micro, pooled, qc, eta
    raise RuntimeError(f"Could not find a micro-cluster layout for {level} within search budget")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base_scene = load_yaml(root / "configs/scenes/base.yaml")

    bounds = (
        float(base_scene["n_min"]),
        float(base_scene["n_max"]),
        float(base_scene["e_min"]),
        float(base_scene["e_max"]),
    )
    gcs = tuple(map(float, base_scene.get("gcs_ne", base_scene["start_ne"])))  # type: ignore[assignment]
    blackholes = [list(map(float, c)) for c in list(base_scene.get("communication_blackholes") or [])]

    # parameters (paper-aligned defaults; adjust as needed)
    d_max = 250.0
    thr = ConflictThresholds()

    # POI counts
    total_poi = 120
    n_cluster = G2_N_CLUSTER_POI
    n_scatter = total_poi - n_cluster
    cluster_disk_r = G2_MICRO_DISK_R_M
    # Only exclude scatter very near a micro-center so gaps between sites can fill with scatter
    scatter_excl_m = cluster_disk_r + 12.0

    # G1: stratified jittered grid (even coverage + randomness), not i.i.d. uniform
    rng_g1 = random.Random(2027)
    g1: List[Point] = []
    for jitter in (G1_GRID_JITTER_FRAC, 0.30, 0.22, 0.16):
        cand = sample_stratified_jittered_grid(
            rng=rng_g1,
            n=total_poi,
            bounds=bounds,
            margin=12.0,
            jitter_frac=jitter,
        )
        if _min_nn_dist(cand) >= NN_MIN_DIST_M:
            g1 = cand
            break
    if not g1:
        g1 = sample_stratified_jittered_grid(
            rng=rng_g1,
            n=total_poi,
            bounds=bounds,
            margin=12.0,
            jitter_frac=0.12,
        )
    g1_dbar, g1_u = _nn_uniformity_index(g1, bounds)
    dump_scene(
        root / "configs/scenes/g1_uniform.yaml",
        poi_list=g1,
        header=(
            f"Layer-1: Task distribution G1 (stratified jittered grid), {total_poi} POIs — "
            "one sample per cell for even coverage, jitter adds randomness.\n"
            f"Nearest-neighbor: dbar_nn={g1_dbar:.2f}m, U={g1_u:.3f} (reference U for random-uniform ≈1)."
        ),
    )

    # G2 + M levels: several loose on-disk micro-sites + many scatter POIs
    for level, seed in [("M0", 2100), ("M1", 2101), ("M2", 2102)]:
        rng = random.Random(seed)
        micro_centers, cluster, qc, eta = choose_micro_layout_for_level(
            level=level,
            rng=rng,
            bounds=bounds,
            gcs=gcs,
            blackholes=blackholes,
            d_max=d_max,
            thr=thr,
            cluster_disk_r_m=cluster_disk_r,
            n_cluster=n_cluster,
            k_min=G2_N_MICRO_MIN,
            k_max=G2_N_MICRO_MAX,
            ring_r=G2_MICRO_RING_R_M,
        )
        k_micro = len(micro_centers)
        # scattered points: keep away from every micro-cluster center
        scatter: List[Point] = []
        scat_rng = random.Random(seed + 1999)
        while len(scatter) < n_scatter:
            x = scat_rng.uniform(bounds[0] + 12.0, bounds[1] - 12.0)
            y = scat_rng.uniform(bounds[2] + 12.0, bounds[3] - 12.0)
            if _nearest_center_dist_m((x, y), micro_centers) < scatter_excl_m:
                continue
            scatter.append((x, y))
        pts = cluster + scatter
        # Nearest-neighbor metrics for the full POI set (clustered should have U < 1)
        dbar_nn, u_nn = _nn_uniformity_index(pts, bounds)
        # If too uniform (U too high) or too dense (min nn too low), resample scatter until OK.
        if _min_nn_dist(pts) < NN_MIN_DIST_M or u_nn > U_G2_MAX:
            best_pts = pts
            best_dbar, best_u = dbar_nn, u_nn
            for k in range(800):
                scatter = []
                scat_rng = random.Random(seed + 1999 + k)
                while len(scatter) < n_scatter:
                    x = scat_rng.uniform(bounds[0] + 12.0, bounds[1] - 12.0)
                    y = scat_rng.uniform(bounds[2] + 12.0, bounds[3] - 12.0)
                    if _nearest_center_dist_m((x, y), micro_centers) < scatter_excl_m:
                        continue
                    scatter.append((x, y))
                cand = cluster + scatter
                dbar_nn, u_nn = _nn_uniformity_index(cand, bounds)
                if u_nn <= U_G2_MAX and _min_nn_dist(cand) >= NN_MIN_DIST_M:
                    best_pts, best_dbar, best_u = cand, dbar_nn, u_nn
                    break
                if u_nn < best_u and _min_nn_dist(cand) >= NN_MIN_DIST_M:
                    best_pts, best_dbar, best_u = cand, dbar_nn, u_nn
            pts, dbar_nn, u_nn = best_pts, best_dbar, best_u

        header = (
            f"Layer-1: Task distribution G2 ({k_micro} loose disk sites, {n_cluster} clustered + {n_scatter} scatter) "
            f"+ conflict level {level}, {total_poi} POIs.\n"
            f"Computed by paper1.pdf §5.2.3 (iii): Qbar_C={qc:.3f}, eta_C={eta:.3f}, micro_centers={micro_centers}.\n"
            f"Nearest-neighbor: dbar_nn={dbar_nn:.2f}m, U={u_nn:.3f} (target U<= {U_G2_MAX})."
        )
        out_name = {
            "M0": "g2_cluster_m0.yaml",
            "M1": "g2_cluster_m1.yaml",
            "M2": "g2_cluster_m2.yaml",
        }[level]
        dump_scene(root / f"configs/scenes/{out_name}", poi_list=pts, header=header)

    print(f"Generated scenes under configs/scenes/: g1_uniform + g2_cluster_m0/m1/m2 ({total_poi} POIs each).")


if __name__ == "__main__":
    main()

