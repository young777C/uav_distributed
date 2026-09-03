"""Step-1 baseline: measure the CURRENT (no-WM) model's out-of-frame prediction.

The action eval teacher-forces DiT on GT waypoints, so the intercept bottleneck is
EAR's OWN prediction. This script samples EAR's waypoints and reports FDE (meters)
split by VISIBLE vs LOSS frames × horizon, plus a velocity-free 'persistence'
baseline (predict future = the 1s waypoint). If EAR's LOSS-frame FDE is large / no
better than persistence -> the reactive model can't predict out-of-frame -> WM justified.
"""
from __future__ import annotations
import argparse
import numpy as np, torch, yaml
from torch.utils.data import DataLoader

from train.dataset_stage2 import RealStage2Dataset, collate_stage2
from train.ear import EAR
from train.iar import IAR
from train.dit import DiT
from train.stage2 import action_sample
from train.flow_matching import sample as ear_sample


def _cos(a, b, eps=1e-6):
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config_v5.yaml")
    ap.add_argument("--ckpt", default="runs/stage2_v5/stage2_best.pt")
    ap.add_argument("--device", default="cuda:2")
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(a.config))
    ck = torch.load(a.ckpt, map_location="cpu")
    K = ck["dims"]["K"]; cond_dim = ck["dims"]["cond_dim"]; pdim = ck["dims"]["pdim"]
    m = cfg["model"]; wp_scale = cfg["waypoint"]["scale"]
    offs = cfg["waypoint"]["offsets_s"]
    steps = cfg["flow"]["sample_steps"]

    H = ck["dims"]["H"]; A = ck["dims"]["A"]; nL = ck["dims"]["n_layers_in"]
    ear = EAR(cond_dim, k=K, d_model=m["ear"]["d_model"], n_layers=m["ear"]["n_layers"],
              n_heads=m["ear"]["n_heads"], proprio_dim=pdim).to(dev)
    ear.load_state_dict(ck["ear"]); ear.eval()
    iar = IAR(cond_dim, nL, d=m["iar"]["d"], n_im=m["iar"]["n_im"], n_heads=m["iar"]["n_heads"]).to(dev)
    iar.load_state_dict(ck["iar"]); iar.eval()
    dit = DiT(A, H, cond_dim, d_ex=3, d_im=m["iar"]["d"], proprio_dim=pdim,
              d=m["dit"]["d_model"], n_layers=m["dit"]["n_layers"], n_heads=m["dit"]["n_heads"]).to(dev)
    dit.load_state_dict(ck["dit"]); dit.eval()

    ds = RealStage2Dataset(cfg, "val")
    dl = DataLoader(ds, batch_size=64, collate_fn=collate_stage2, num_workers=4)
    icos = {"vis": [], "loss": []}    # 拦截朝向 cos:模型自预测动作 vs 专家动作

    # accumulate FDE per horizon, split by visible/loss; plus depth(forward,dim0) vs
    # lateral(image-plane,dims1:2) decomposition — is the EAR error monocular-depth or planar?
    agg = {("vis", k): [] for k in range(K)}; agg.update({("loss", k): [] for k in range(K)})
    pers = {("vis", k): [] for k in range(K)}; pers.update({("loss", k): [] for k in range(K)})
    dep = {("vis", k): [] for k in range(K)}; dep.update({("loss", k): [] for k in range(K)})
    lat = {("vis", k): [] for k in range(K)}; lat.update({("loss", k): [] for k in range(K)})
    nvis = nloss = 0
    with torch.no_grad():
        for b in dl:
            ctx = b["vlm_ctx"].to(dev); cm = b["ctx_mask"].to(dev); prop = b["proprio"].to(dev)
            gt = b["waypoint"].to(dev)                                  # (B,K,3) cam-frame /scale
            pred = ear_sample(ear, ctx, cm, K, steps, proprio=prop)     # (B,K,3) EAR 自预测
            fde = (pred - gt).norm(dim=-1) * wp_scale                   # (B,K) meters
            depe = (pred[..., 0] - gt[..., 0]).abs() * wp_scale         # (B,K) depth (forward axis)
            late = (pred[..., 1:] - gt[..., 1:]).norm(dim=-1) * wp_scale  # (B,K) lateral (image plane)
            persist = (gt[:, 0:1, :] - gt).norm(dim=-1) * wp_scale      # predict future = 1s wp
            vis = b["visible"].to(dev) > 0.5                            # (B,)
            # 拦截朝向 cos:用【EAR 自预测航点】驱动 DiT(非 teacher-forcing),对比专家动作方向
            zim, _ = iar([x.to(dev) for x in b["vlm_ctx_layers"]], cm)
            act = action_sample(dit, pred, zim, ctx, prop, cm, H, A, steps)   # (B,H,A) 模型动作
            md = act[:, :, :3].sum(1)                                   # 净位移方向(模型)
            ed = b["action"][:, :, :3].to(dev).sum(1)                   # 净位移方向(专家)
            c = _cos(md, ed)                                            # (B,)
            for grp, mask in [("vis", vis), ("loss", ~vis)]:
                if mask.any():
                    icos[grp].append(c[mask].cpu().numpy())
                    for k in range(K):
                        agg[(grp, k)].append(fde[mask, k].cpu().numpy())
                        pers[(grp, k)].append(persist[mask, k].cpu().numpy())
                        dep[(grp, k)].append(depe[mask, k].cpu().numpy())
                        lat[(grp, k)].append(late[mask, k].cpu().numpy())
            nvis += int(vis.sum()); nloss += int((~vis).sum())

    def mean(d, grp, k):
        arr = np.concatenate(d[(grp, k)]) if d[(grp, k)] else np.array([0.0])
        return arr.mean()

    print(f"[intercept-eval] ckpt={a.ckpt}  val帧: 可见={nvis} 丢失={nloss}")
    print(f"  waypoint 视界(s): {offs}")
    print(f"  === EAR 预测 FDE(米,越低越好) ===")
    for grp, name in [("vis", "可见帧"), ("loss", "丢失帧")]:
        ear_f = [mean(agg, grp, k) for k in range(K)]
        per_f = [mean(pers, grp, k) for k in range(K)]
        dep_f = [mean(dep, grp, k) for k in range(K)]
        lat_f = [mean(lat, grp, k) for k in range(K)]
        print(f"  {name}:  EAR   " + "  ".join(f"{offs[k]}s={ear_f[k]:5.1f}" for k in range(K)))
        print(f"  {name}:   ├深度 " + "  ".join(f"{offs[k]}s={dep_f[k]:5.1f}" for k in range(K))
              + "   ← 沿光轴(单目深度歧义)")
        print(f"  {name}:   └横向 " + "  ".join(f"{offs[k]}s={lat_f[k]:5.1f}" for k in range(K))
              + "   ← 像平面内(跟踪/居中相关)")
        print(f"  {name}:  persist " + "  ".join(f"{offs[k]}s={per_f[k]:5.1f}" for k in range(K))
              + "   ← 速度无关基线(预测未来=1s位置)")
    print("  判读: 丢失帧 EAR FDE 若 >> 可见帧、或 不优于 persist → 反应式模型出画预测失败 → WM 有据")
    print("  判读2: 可见帧 EAR 若[深度]>>[横向] → 14m 主要是单目深度歧义(可观测性限制),非跟踪失败")
    print(f"  === 拦截朝向 cos(模型自预测动作 vs 专家动作, 越接近1越好; 专家自身≈1.0) ===")
    for grp, name in [("vis", "可见帧"), ("loss", "丢失帧")]:
        arr = np.concatenate(icos[grp]) if icos[grp] else np.array([0.0])
        print(f"  {name}: cos 中位={np.median(arr):.2f}  均值={arr.mean():.2f}  cos>0.5占比={100*(arr>0.5).mean():.0f}%")
    print("  判读: 丢失帧 cos 高→模型直飞目标(拦截行为在); 低→乱飞/没学到拦截")


if __name__ == "__main__":
    main()
