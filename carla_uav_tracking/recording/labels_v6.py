"""v6 per-actor labels (acot_note/1003_data-regen-v6-spec.md).

Live (recorder, per frame): amodal 2D box from the actor's 3D bounding box, structural-occlusion
raycast, and pixel-level visibility from the instance-segmentation camera. CARLA 0.9.15 instance
ids are unique and stable per object but are NOT actor ids, so blobs are associated to scene
actors by overlap with the projected amodal box.
Offline (postprocess): tracklet ids that BREAK after a visibility gap, and target re-entry events.
"""
from __future__ import annotations

import numpy as np

VEHICLE_TAGS = (14, 15, 16, 18, 19)     # Car, Truck, Bus, Motorcycle, Bicycle (CARLA 0.9.14+ palette)
PEDESTRIAN_TAG = 12
MASK_OTHER_VEHICLE = 254                # instance_mask value for non-scene vehicles
MASK_PEDESTRIAN = 253
MAX_TRAFFIC = 32                        # padded per-frame list of non-scene vehicle blobs
VIEW_BLOCK_TAGS = (0, 3, 4)             # Unlabeled, Building, Wall — camera inside / pressed against geometry
NEAR = 0.1                              # near plane (m) for amodal-box clipping


def actor_handle(obj):
    """carla.Actor behind a scene target/distractor wrapper (VehicleTarget / PedestrianTarget)."""
    return getattr(obj, "_vehicle", None) or getattr(obj, "_walker", None)


def focal(W: int, fov_deg: float) -> float:
    return W / (2.0 * np.tan(np.radians(fov_deg) / 2.0))


def project_world(points: np.ndarray, cam_inv: np.ndarray, W: int, H: int, f: float):
    """points (N,3) world → (u, v, depth) arrays; CARLA camera frame x fwd, y right, z up."""
    P = np.concatenate([points, np.ones((len(points), 1))], 1) @ cam_inv.T
    d = P[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = f * (P[:, 1] / d) + W / 2.0
        v = f * (-P[:, 2] / d) + H / 2.0
    return u, v, d


def amodal_box(actor, cam_inv, W, H, f):
    """Tight 2D box of the projected 3D bounding box, clipped to the image.
    Returns ((x0, y0, x1, y1) or None, center_in_frame, depth_of_center)."""
    tf = actor.get_transform()
    verts = actor.bounding_box.get_world_vertices(tf)
    pts = np.array([[v.x, v.y, v.z] for v in verts], np.float64)
    return amodal_box_from_vertices(pts, cam_inv, W, H, f)


def amodal_box_from_vertices(pts, cam_inv, W, H, f):
    """Same as amodal_box but from the 8 world vertices (offline re-projection). Vertices behind the
    near plane are replaced by the intersections of every front/behind vertex pair with the plane, so
    the box of a car crossing the image border/near plane does not shrink abruptly."""
    P = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ cam_inv.T          # camera frame (x fwd)
    c = P.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cu, cv = f * (c[1] / c[0]) + W / 2.0, f * (-c[2] / c[0]) + H / 2.0
    cd = c[0]
    center_in = bool(cd > NEAR and 0 <= cu < W and 0 <= cv < H)
    front = P[:, 0] > NEAR
    if not front.any():
        return None, False, float(cd)
    Q = [P[front, :3]]
    for i in np.where(front)[0]:
        for j in np.where(~front)[0]:
            t = (P[i, 0] - NEAR) / (P[i, 0] - P[j, 0])
            Q.append((P[i, :3] + t * (P[j, :3] - P[i, :3]))[None])
    Q = np.concatenate(Q, 0)
    u = f * (Q[:, 1] / Q[:, 0]) + W / 2.0
    v = f * (-Q[:, 2] / Q[:, 0]) + H / 2.0
    x0, x1 = np.clip(u.min(), 0, W - 1), np.clip(u.max(), 0, W - 1)
    y0, y1 = np.clip(v.min(), 0, H - 1), np.clip(v.max(), 0, H - 1)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None, center_in, float(cd)
    return (float(x0), float(y0), float(x1), float(y1)), center_in, float(cd)


def decode_instance(raw: bytes, W: int, H: int):
    """CARLA instance-seg BGRA → (tag (H,W) uint8, iid (H,W) int32). R = semantic tag, G|B<<8 = id."""
    a = np.frombuffer(raw, np.uint8).reshape(H, W, 4)
    tag = a[..., 2].copy()
    iid = a[..., 1].astype(np.int32) | (a[..., 0].astype(np.int32) << 8)
    return tag, iid


def view_block_frac(tag) -> float:
    """Fraction of the image tagged Unlabeled/Building/Wall — ≈1 when the camera is inside or pressed
    against geometry (garbage frame)."""
    return float(np.isin(tag, VIEW_BLOCK_TAGS).mean())


def associate(boxes, tag, iid, min_inside: float = 0.5, prev=None):
    """Assign instance blobs to scene actors.

    boxes: list (len A) of amodal boxes (x0,y0,x1,y1) or None.
    Returns per-actor arrays (iid, vis_px, modal box) + uint8 instance mask (0 bg, 1..A scene actors,
    254 other vehicles, 253 pedestrians) + padded traffic table (MAX_TRAFFIC, 7):
    [iid, tag, x0, y0, x1, y1, px] for unmatched vehicle blobs (iid = -1 padding).
    Greedy by overlap; a blob is accepted only if >= min_inside of its pixels lie inside the box
    (so a neighbouring occluder is not taken for the actor). `prev` (A,) = last frame's iid per actor:
    a candidate that keeps the actor's previous iid gets its overlap doubled (temporal consistency)."""
    H, W = tag.shape
    A = len(boxes)
    veh = np.isin(tag, VEHICLE_TAGS)
    out_iid = np.full(A, -1, np.int32)
    out_px = np.zeros(A, np.int32)
    out_box = np.full((A, 4), -1.0, np.float32)
    mask = np.zeros((H, W), np.uint8)
    mask[tag == PEDESTRIAN_TAG] = MASK_PEDESTRIAN
    if veh.any():
        ids, counts = np.unique(iid[veh], return_counts=True)
        total = dict(zip(ids.tolist(), counts.tolist()))
    else:
        total = {}
    cands = []                                   # (overlap, actor, iid)
    for a, b in enumerate(boxes):
        if b is None:
            continue
        x0, y0, x1, y1 = (int(np.floor(b[0])), int(np.floor(b[1])), int(np.ceil(b[2])) + 1, int(np.ceil(b[3])) + 1)
        sub_v = veh[y0:y1, x0:x1]
        if not sub_v.any():
            continue
        ii, cc = np.unique(iid[y0:y1, x0:x1][sub_v], return_counts=True)
        for i_, c_ in zip(ii.tolist(), cc.tolist()):
            if c_ >= min_inside * total.get(i_, 1):
                keep = prev is not None and prev[a] == i_
                cands.append((c_ * (2 if keep else 1), a, i_))
    used_a, used_i = set(), set()
    for c_, a, i_ in sorted(cands, reverse=True):
        if a in used_a or i_ in used_i:
            continue
        used_a.add(a); used_i.add(i_)
        m = veh & (iid == i_)
        ys, xs = np.nonzero(m)
        out_iid[a], out_px[a] = i_, len(xs)
        out_box[a] = (xs.min(), ys.min(), xs.max(), ys.max())
        mask[m] = a + 1
    traffic = np.full((MAX_TRAFFIC, 7), -1.0, np.float32)
    k = 0
    for i_, c_ in sorted(total.items(), key=lambda t: -t[1]):
        if i_ in used_i:
            continue
        m = veh & (iid == i_)
        mask[m] = MASK_OTHER_VEHICLE
        if k < MAX_TRAFFIC:
            ys, xs = np.nonzero(m)
            traffic[k] = (i_, float(np.bincount(tag[m]).argmax()), xs.min(), ys.min(), xs.max(), ys.max(), c_)
            k += 1
    return out_iid, out_px, out_box, mask, traffic


# ---------------------------------------------------------------- offline (postprocess)

def tracklet_ids(visible: np.ndarray, gap_frames: int) -> np.ndarray:
    """visible (T, A) bool → tracklet id (T, A) int32, -1 where invisible. A new id is issued when an
    actor becomes visible after being invisible for >= gap_frames (shorter gaps keep the id).
    Ids are unique within the episode."""
    T, A = visible.shape
    out = np.full((T, A), -1, np.int32)
    nxt = 0
    for a in range(A):
        cur, gap = -1, 10 ** 9
        for t in range(T):
            if visible[t, a]:
                if cur < 0 or gap >= gap_frames:
                    cur = nxt; nxt += 1
                out[t, a] = cur
                gap = 0
            else:
                gap += 1
    return out


def reentry_events(visible_t: np.ndarray, off_t: np.ndarray, n_lookalike_vis: np.ndarray,
                   n_vehicle_vis: np.ndarray, min_gap: int) -> np.ndarray:
    """Target re-entry events → (K, 5) int32: [frame, gap_frames, cause (0 off-screen, 1 occlusion,
    2 mixed), n_lookalikes_visible, n_scene_vehicles_visible] at the first visible frame after an
    invisible run of >= min_gap frames."""
    ev, run, n_off = [], 0, 0
    for t in range(len(visible_t)):
        if not visible_t[t]:
            run += 1; n_off += int(off_t[t])
            continue
        if run >= min_gap:
            cause = 0 if n_off == run else (1 if n_off == 0 else 2)
            ev.append((t, run, cause, int(n_lookalike_vis[t]), int(n_vehicle_vis[t])))
        run, n_off = 0, 0
    return np.array(ev, np.int32).reshape(-1, 5)
