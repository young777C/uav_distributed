"""WM M0 annotation — append road-network prediction labels to existing episodes.

Pure ANNOTATION pass (no re-render, no re-sim): reads the already-stored
target/{tx,ty,tz,tspeed,tyaw} + queries the CARLA map, and appends three fields
into each episode's existing `annotation/` group:

  annotation/road_pred          (T, 4, 3) float32  lane-traversal prediction @ {1,2,4,6}s
  annotation/target_is_junction (T,)      uint8     map.get_waypoint(pos).is_junction
  annotation/target_lane_id     (T,)      int32     road_id*100 + lane_id

Per acot_note/wm-m0-data-spec.md §2. Groups episodes by town (one load_world per
town). Idempotent (overwrites the three fields). RGB / everything else untouched.

  # inside the cyh-carla container:
  python scripts/annotate_road_pred.py --dir /data/mvp_full_v5 --port 2001
"""
from __future__ import annotations

import argparse
import glob
import math
from collections import defaultdict

import carla
import h5py
import numpy as np

HORIZONS = [1.0, 2.0, 4.0, 6.0]   # seconds; MUST match config_v5 waypoint.offsets_s
STEP = 2.0                        # lane-traversal step (m)
FPS = 10


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _pick_straightest(nxts, ref_yaw):
    """Successor whose heading best matches ref_yaw (straightest branch at a junction)."""
    return min(nxts, key=lambda w: abs(_wrap180(w.transform.rotation.yaw - ref_yaw)))


def road_pred_for_frame(cmap, p, speed, ref_yaw):
    """(4,3) predicted world positions at {1,2,4,6}s along the lane network."""
    wp = cmap.get_waypoint(carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                           project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return None, 0, 0                      # degenerate: not on a driving lane
    is_junction = int(wp.is_junction)
    lane_id = int(wp.road_id) * 100 + int(wp.lane_id)
    out = np.zeros((len(HORIZONS), 3), np.float32)
    # cumulative single pass to the largest horizon, recording at each horizon's arc length
    targets = [speed * off for off in HORIZONS]
    cur, ry, travelled, ki = wp, float(ref_yaw), 0.0, 0
    max_arc = targets[-1]
    while ki < len(targets):
        # record any horizons already reached at the current position
        while ki < len(targets) and travelled >= targets[ki] - 1e-3:
            loc = cur.transform.location
            out[ki] = (loc.x, loc.y, loc.z); ki += 1
        if ki >= len(targets) or travelled >= max_arc:
            break
        d = min(STEP, targets[-1] - travelled)
        nxts = cur.next(d)
        if not nxts:                            # dead-end → stop advancing
            loc = cur.transform.location
            for kk in range(ki, len(targets)):
                out[kk] = (loc.x, loc.y, loc.z)
            ki = len(targets); break
        cur = _pick_straightest(nxts, ry)
        ry = cur.transform.rotation.yaw
        travelled += d
    return out, is_junction, lane_id


def annotate_episode(ep, cmap):
    with h5py.File(ep, "r+") as f:
        for k in ("target/tx", "target/ty", "target/tz", "target/tspeed", "target/tyaw"):
            if k not in f:
                return {"skip": f"missing {k}"}
        tx, ty, tz = f["target/tx"][:], f["target/ty"][:], f["target/tz"][:]
        spd, tyaw = f["target/tspeed"][:], f["target/tyaw"][:]
        T = f["rgb"].shape[0]
        rp = np.zeros((T, len(HORIZONS), 3), np.float32)
        isj = np.zeros(T, np.uint8)
        lid = np.zeros(T, np.int32)
        degen = 0
        for i in range(T):
            r, j, l = road_pred_for_frame(cmap, (tx[i], ty[i], tz[i]), float(spd[i]), float(tyaw[i]))
            if r is None:
                rp[i] = np.array([tx[i], ty[i], tz[i]], np.float32)   # degenerate: hold position
                degen += 1
            else:
                rp[i], isj[i], lid[i] = r, j, l
        g = f.require_group("annotation")
        for name, data in (("road_pred", rp), ("target_is_junction", isj), ("target_lane_id", lid)):
            if name in g:
                del g[name]
            g.create_dataset(name, data=data, compression="gzip", compression_opts=4)
        # QC: road_pred[:,0] (1s) vs actual position 1s (=FPS frames) later
        h = FPS
        if T > h:
            act = np.stack([tx[h:], ty[h:], tz[h:]], -1)
            err = np.linalg.norm(rp[:-h, 0] - act, axis=1)
            med1s = float(np.median(err))
        else:
            med1s = float("nan")
        return {"T": T, "junction_frac": float(isj.mean()), "degen": degen,
                "med_1s_err_m": med1s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2001)
    a = ap.parse_args()

    eps = sorted(glob.glob(f"{a.dir}/episode_*.h5"))
    by_town = defaultdict(list)
    bad = []
    for ep in eps:
        try:
            with h5py.File(ep, "r") as f:
                by_town[f.attrs.get("town", "Town01")].append(ep)
        except Exception as e:
            bad.append((ep, str(e)[:50]))

    client = carla.Client(a.host, a.port); client.set_timeout(60.0)
    reports = []
    for town, town_eps in sorted(by_town.items()):
        print(f"[town] {town}: {len(town_eps)} episodes — loading map...", flush=True)
        cur = client.get_world().get_map().name.split("/")[-1]
        if cur != town:
            client.load_world(town)
        cmap = client.get_world().get_map()
        for ep in town_eps:
            name = ep.split("/")[-1]
            try:
                r = annotate_episode(ep, cmap)
                r["ep"] = name; r["town"] = town; reports.append(r)
                print(f"  {name}: T={r.get('T')} junc={r.get('junction_frac',0):.2f} "
                      f"degen={r.get('degen')} 1s_err={r.get('med_1s_err_m',float('nan')):.1f}m "
                      f"{r.get('skip','')}", flush=True)
            except Exception as e:
                bad.append((ep, str(e)[:60])); print(f"  {name}: FAILED {e}", flush=True)

    # aggregate QC
    ok = [r for r in reports if "skip" not in r and not math.isnan(r.get("med_1s_err_m", float("nan")))]
    print("\n=== QC 汇总 ===")
    print(f"处理成功: {len(ok)} 集 | 坏/跳过: {len(bad)}")
    if ok:
        errs = [r["med_1s_err_m"] for r in ok]
        juncs = [r["junction_frac"] for r in ok]
        degens = sum(r["degen"] for r in ok)
        Ttot = sum(r["T"] for r in ok)
        print(f"1s road_pred 误差(各集中位)的中位: {np.median(errs):.2f}m  (QC: <5m 合格)")
        print(f"路口帧占比: 均值 {np.mean(juncs)*100:.1f}%  (QC: 5-25% 合理)")
        print(f"退化帧(不在车道): {degens}/{Ttot} = {100*degens/max(Ttot,1):.1f}%")
    for ep, err in bad:
        print(f"  坏/跳过: {ep.split('/')[-1]} — {err}")


if __name__ == "__main__":
    main()
