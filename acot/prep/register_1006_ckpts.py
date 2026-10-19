"""1006 T0: register the TAHRelM checkpoints trained in T0 (G13 recipe replicates, DINOv2 / DINOv3 offline-only) as
tah_ckpt artefacts. Inputs = the recipe stored in each checkpoint + best epoch / val acc; compat = the training data's:
cache_dag (64 offline + 23 DAgger) → same compat as tah_relm_dag; offline-only caches → tick 0.2 s; DINOv3 cache →
its feature id / dim; the DINOv3 full recipe (64 offline + a @3 DAgger round) → tick [0.2, 0.6].
  python -m acot.prep.register_1006_ckpts [--only NAME …]"""
import argparse
import os

import torch

from acot import ROOT
from acot import artefacts as A

V2 = "dino_s14_lvd142m_cls384_c224_p1.3"
V3 = "dinov3_vitb16_lvd1689m_cls768_c224_p1.3"
SPECS = [  # name, file, compat
    ("tah_relm_1006_s0", "runs/tah_1006/tah_relm_s0.pt", {"d_feat": 384, "feature_id": V2, "fov_deg": "70img/90proj+90", "tick_s": [0.2, 0.6]}),
    ("tah_relm_1006_s1", "runs/tah_1006/tah_relm_s1.pt", {"d_feat": 384, "feature_id": V2, "fov_deg": "70img/90proj+90", "tick_s": [0.2, 0.6]}),
    ("tah_relm_1006_s2", "runs/tah_1006/tah_relm_s2.pt", {"d_feat": 384, "feature_id": V2, "fov_deg": "70img/90proj+90", "tick_s": [0.2, 0.6]}),
    ("tah_relm_1006_v2off_s0", "runs/tah_1006/tah_relm_v2off_s0.pt", {"d_feat": 384, "feature_id": V2, "fov_deg": "70img/90proj", "tick_s": 0.2}),
    ("tah_relm_1006_v3off_s0", "runs/tah_1006/tah_relm_v3off_s0.pt", {"d_feat": 768, "feature_id": V3, "fov_deg": "70img/90proj", "tick_s": 0.2}),
    ("tah_relm_1006_v3full_s0", "runs/tah_1006/tah_relm_v3full_s0.pt", {"d_feat": 768, "feature_id": V3, "fov_deg": "70img/90proj+90", "tick_s": [0.2, 0.6]}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+")
    a = ap.parse_args()
    for name, f, comp in SPECS:
        if a.only and name not in a.only:
            continue
        ck = torch.load(os.path.join(ROOT, f), map_location="cpu", weights_only=False)
        inputs = {"recipe": ck.get("recipe", {}), "best_epoch": ck.get("best_epoch"), "best_val_acc": ck.get("best_val_acc"),
                  "n_train_eps": ck.get("n_train_eps"), "n_val_eps": ck.get("n_val_eps")}
        compat = dict({"arch": ck.get("arch"), "depth_scale": 100.0}, **comp)
        m = A.register("tah_ckpt", name, files=[os.path.join(ROOT, f)], inputs=inputs, compat=compat,
                       producer={"tool": "1006 T0", "script": "train/tah_train.py"},
                       provenance_gaps=["fov_deg: 70° images projected with a 90° camera (inherited from the caches)"])
        print(f"{name:26s} @{m['id']}  {compat['feature_id']}  d_feat={compat['d_feat']}")


if __name__ == "__main__":
    main()
