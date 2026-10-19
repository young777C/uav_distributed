"""The frontend every arm shares (protocol.frontend, framework §5.2): the frozen VLM (OnlineVLM, language mode), the
Stage-2 modules of frontend.vlm.stage2_ckpt (EAR / IAR / DiT / grounding head), the crop feature extractor
(frontend.crop_features) and the opening scorer (frontend.opening). Built from the resolved protocol only — no
environment variables, no constructor defaults, checkpoints by explicit path."""
import os

import torch
import yaml

from acot import ROOT
from acot.runtime import support

def attr_opening(ckpt_lin, ckpt_ft, device, wm, wc, wq, ft, vlm=None):
    """AttrScorer (carla_uav_tracking/rollout/opening_zoom.py) with BOTH checkpoints given explicitly (the legacy
    constructor reads the fine-tuned one from a fixed eval_out path)."""
    import timm
    from rollout.opening_zoom import AttrScorer, Dino, ZoomVerifier
    s = AttrScorer.__new__(AttrScorer)
    ck = torch.load(ckpt_lin, map_location=device, weights_only=False)
    s.ft = float(ft)
    s.ftnet = None
    if s.ft:
        fk = torch.load(ckpt_ft, map_location=device, weights_only=False)
        b = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0, img_size=224)
        net = torch.nn.ModuleDict({"b": b, "hm": torch.nn.Linear(768, len(fk["cls_m"])),
                                   "hc": torch.nn.Linear(768, len(fk["cls_c"]))})
        net.load_state_dict(fk["state"])
        s.ftnet = net.to(device).eval()
        s.ft_idx = ({c: i for i, c in enumerate(fk["cls_m"])}, {c: i for i, c in enumerate(fk["cls_c"])})
        s.ft_scale = float(fk["test_scale"])
    s.dino, s.dev = Dino(device), device
    s.scales = tuple(ck["scales"])
    s.heads = {}
    for k in ("m", "c"):
        sd = ck[f"W{k}"]
        W = torch.nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0]).to(device)
        W.load_state_dict(sd)
        W.eval()
        s.heads[k] = (W, {c: i for i, c in enumerate(ck[f"cls_{k}"])}, float(ck[f"T{k}"]))
    s.wm, s.wc, s.wq = float(wm), float(wc), float(wq)
    s.zv = None
    if s.wq:
        if vlm is None:
            raise ValueError("opening attr with wq > 0 needs the VLM (Qwen yes/no colour term)")
        s.zv = ZoomVerifier(vlm.bk.model, vlm.bk.processor, device, scale=2.0, spotlight=False)
    return s


class Frontend:
    def __init__(self, protocol, device, vlm=None, log=print):
        from rollout.online_vlm import OnlineVLM
        from rollout.policy import load_policy_modules
        from train.reid_resolution_probe import _load_dino
        bad = support.frontend(protocol)
        if bad:
            raise ValueError("Frontend: " + "; ".join(bad))
        fr = protocol["frontend"]
        self.device = device
        self.cfg = yaml.safe_load(open(os.path.join(ROOT, fr["vlm"]["model_config"])))
        if vlm is None:
            vlm = OnlineVLM(self.cfg, device=device, language_mode=fr["vlm"]["language_mode"])
        elif vlm.language_mode != fr["vlm"]["language_mode"]:
            raise ValueError(f"shared VLM has language_mode {vlm.language_mode}, protocol {fr['vlm']['language_mode']}")
        self.vlm = vlm
        self.ear, self.iar, self.dit, self.tid, d, self.ckpt_meta = load_policy_modules(
            os.path.join(ROOT, fr["vlm"]["stage2_ckpt"]), self.cfg, device)
        self.K, self.H, self.A = int(d["K"]), int(d["H"]), int(d["A"])
        self.a_scale = float(self.cfg["action"]["scale"])
        self.steps = int(self.cfg["flow"]["sample_steps"])
        self.image_cfg = self.cfg["image"]
        self.dino = _load_dino(support.CROP_FEATURES[fr["crop_features"]["id"]], device)
        o = fr["opening"]
        self.opening = None
        if o["module"] == "attr":
            a = o["attr"]
            self.opening = attr_opening(os.path.join(ROOT, a["ckpt_lin"]), os.path.join(ROOT, a["ckpt_ft"]), device,
                                        a["wm"], a["wc"], a["wq"], a["ft"], vlm=vlm)
        log(f"[acot-frontend] stage-2 {fr['vlm']['stage2_ckpt']} (epoch {self.ckpt_meta.get('epoch')}), "
            f"language_mode={fr['vlm']['language_mode']}, opening={o['module']}")

    def encode(self, rgb, language):
        """→ (ctx_layers [(M,C)], ctx (M,C), grid_last (Gh,Gw,C) numpy)."""
        return self.vlm.encode(rgb, language)
