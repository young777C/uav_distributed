"""SimProxyPerception (framework §7.2, P2): the protocol's idealised perception, run in the policy process at S2 ticks.
Turns the simulator state of an act message into Detections (what a method sees) + ScoringInfo (labels), exactly as
the legacy Policy.act did inline (carla_uav_tracking/rollout/policy.py):
  candidates  build_candidates: GT-projected pseudo-boxes of the in-frame task vehicles, shuffled by the episode RNG
              (seeded with control.shuffle_seed at every reset), VLM grid pooled in each box;
  occlusion   raycast → boxes of vehicles the env's ray-cast marks occluded are dropped;
  identity    tracker → crop-DINOv2 features + TrackletAssigner ids, greedy assignment (the simulator index stays in
              ScoringInfo.gt_idx);
  WHERE       gt_matched_box → each box carries the world position of the vehicle it was projected from.
Only this configuration (sim_proxy_v2) is implemented; anything else is refused at construction."""
import numpy as np

from acot.runtime import support
from acot.runtime.io import Detection, ScoringInfo


class SimProxyPerception:
    def __init__(self, protocol, image_cfg, shuffle_seed, dino, device):
        import rollout.tracklets as T
        # sim_proxy_v2 tracker = GREEDY assignment by default: every legacy deploy-v2 run had no scipy in the policy
        # image, so TrackletAssigner fell back to greedy (it uses Hungarian only if scipy imports) — pinned, not left
        # to the image. perception.tracker.assign = hungarian (ablation) uses the numpy solver acot/runtime/lsa.py.
        assign = protocol["perception"]["tracker"].get("assign", "greedy")
        if assign == "hungarian":
            from acot.runtime.lsa import linear_sum_assignment
            T._lsa = linear_sum_assignment
        else:
            T._lsa = None
        TrackletAssigner = T.TrackletAssigner
        bad = support.perception(protocol)
        if bad:
            raise ValueError("SimProxyPerception: " + "; ".join(bad))
        pc, cam = protocol["perception"], protocol["camera"]
        geo = {"width": cam["width"], "height": cam["height"], "fov_deg": cam["fov_deg"],
               "vehicle_size_m": pc["candidates"]["vehicle_size_m"], "box_min_px": pc["candidates"]["box_min_px"],
               "box_max_px": pc["candidates"]["box_max_px"]}
        diff = {k: (image_cfg.get(k), v) for k, v in geo.items() if float(image_cfg.get(k, -1)) != float(v)}
        if diff:
            raise ValueError(f"model config image block differs from the protocol geometry: {diff}")
        self.image_cfg = image_cfg
        self.occlusion = pc["occlusion"] == "raycast"
        self.shuffle_seed = int(shuffle_seed)
        self.tracker = TrackletAssigner(**{k: v for k, v in pc["tracker"].items() if k != "assign"})
        self.dino, self.device = dino, device
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.shuffle_seed)
        self.frame_rng = np.random.default_rng([self.shuffle_seed, 1016])   # frame_detections only (own stream)
        self.tracker.reset()

    def crop_feats(self, rgb, boxes):
        from train.reid_resolution_probe import _dino_feats
        W, H = self.image_cfg["width"], self.image_cfg["height"]
        f = _dino_feats(self.dino, rgb, boxes, W, H, 224, 1.3, self.device).astype(np.float64)
        return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)

    def process(self, step, gt_candidates, grid_last, rgb, target_color="", target_bp=""):
        """→ (detections tuple | None, ScoringInfo)."""
        from rollout.candidates import build_candidates
        tpos, dstates, campose = gt_candidates[:3]
        cset = build_candidates(grid_last, tpos, dstates, campose, self.image_cfg, self.rng)
        vis = gt_candidates[4] if len(gt_candidates) > 4 else None
        if cset is not None and self.occlusion and vis is not None:
            t_vis, d_vis = vis
            keep = [i for i, c in enumerate(cset.cands)
                    if (t_vis if c.is_target else (c.idx < len(d_vis) and d_vis[c.idx]))]
            if not keep:
                cset = None
            else:
                cands = [cset.cands[i] for i in keep]
                cset = type(cset)(feats=cset.feats[keep], cands=cands,
                                  true_idx=next((j for j, c in enumerate(cands) if c.is_target), -1),
                                  mask=cset.mask[keep])
        if cset is None:
            return None, ScoringInfo(step=step, true_idx=-1, is_target=(), gt_idx=(), target_color=target_color,
                                     target_bp=target_bp, vis=vis)
        ft = self.crop_feats(rgb, cset.cands)
        keys = self.tracker.update([(c.u, c.v) for c in cset.cands], [c.w for c in cset.cands], ft)
        ds = np.asarray(dstates)
        dets = []
        for i, (c, k) in enumerate(zip(cset.cands, keys)):
            gi = int(c.idx)
            if gi < 0:
                wx = tuple(np.asarray(tpos, np.float64).tolist())
            else:
                wx = tuple(np.asarray(ds[gi][:3], np.float64).tolist()) if gi < ds.shape[0] else None
            dets.append(Detection(key=int(k), u=c.u, v=c.v, w=c.w, h=c.h, depth_m=c.depth, world_xyz=wx,
                                  vlm_feat=cset.feats[i], crop_feat=ft[i]))
        info = ScoringInfo(step=step, true_idx=int(cset.true_idx), is_target=tuple(bool(c.is_target) for c in cset.cands),
                           gt_idx=tuple(int(c.idx) for c in cset.cands), target_color=target_color,
                           target_bp=target_bp, vis=vis)
        return tuple(dets), info

    def frame_detections(self, gt_candidates, rgb, feats):
        """1016 every-frame MOT baselines (DeepSORT / OC-SORT at the camera rate): this frame's candidate boxes by the same
        projection, box and occlusion rules as process(), WITHOUT the shared tracker (key −1), the VLM pooling (dummy
        grid; vlm_feat unused) or the S2 shuffle stream (own stream frame_rng, reset per episode — S2 ticks unchanged).
        feats: also the crop-DINOv2 (DeepSORT's appearance). → tuple of Detections | None (nothing visible)."""
        from rollout.candidates import build_candidates
        tpos, dstates, campose = gt_candidates[:3]
        cset = build_candidates(np.zeros((1, 1, 1), np.float32), tpos, dstates, campose, self.image_cfg,
                                self.frame_rng)
        if cset is None:
            return None
        cands = list(cset.cands)
        vis = gt_candidates[4] if len(gt_candidates) > 4 else None
        if self.occlusion and vis is not None:                   # the occlusion rule of process()
            t_vis, d_vis = vis
            cands = [c for c in cands if (t_vis if c.is_target else (c.idx < len(d_vis) and d_vis[c.idx]))]
        if not cands:
            return None
        ft = self.crop_feats(rgb, cands) if feats else [None] * len(cands)
        v0 = np.zeros(1, np.float32)
        return tuple(Detection(key=-1, u=c.u, v=c.v, w=c.w, h=c.h, depth_m=c.depth, world_xyz=None, vlm_feat=v0,
                               crop_feat=f) for c, f in zip(cands, ft))
