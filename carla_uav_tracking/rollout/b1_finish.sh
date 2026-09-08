#!/usr/bin/env bash
# B1 auto-chain: wait DAgger collect done -> merge offline+DAgger caches -> retrain TAHRel (GPU3)
# -> re-eval closed-loop gt-seeded (GPU2) vs borrow#1.
set -uo pipefail
REPO=/nvidia/hque/code/cyhe/uav-acot-track; cd "$REPO"
say(){ echo "[b1 $(date +%H:%M:%S)] $*"; }
say "waiting for DAgger collect (=== DONE collect)..."
until grep -q "=== DONE collect" runs/tahdag_collect.log 2>/dev/null; do sleep 60; done
say "collect done: $(ls runs/tahdag/*.pt 2>/dev/null | wc -l) dagger episodes"
# 1) merge offline + DAgger caches
docker run --rm -v "$REPO":"$REPO" -w "$REPO" acot-uav-train:cu118 python -c "
import torch,glob
off=torch.load('runs/tah_cache.pt',weights_only=False)
eps=list(off['episodes'])
for f in sorted(glob.glob('runs/tahdag/*.pt')):
    d=torch.load(f,weights_only=False); eps+=d['episodes']
torch.save({'episodes':eps,'d_feat':384},'runs/tah_cache_dag.pt')
nf=sum(len(e) for e in eps); print(f'[merge] {len(eps)} eps ({len(off[\"episodes\"])} offline + rest DAgger), {nf} frames -> tah_cache_dag.pt')
" 2>&1 | grep -E 'merge'
# 2) retrain TAHRel on merged (GPU3)
say "retrain TAHRel on offline+DAgger (GPU3)"
docker run --rm --gpus '"device=3"' -v "$REPO":"$REPO" -w "$REPO" acot-uav-train:cu118 \
  python train/tah_train.py --cache runs/tah_cache_dag.pt --out runs/tah_rel_dag.pt --rel --aug --epochs 60 --wd 3e-4 --device cuda:0 2>&1 | grep -E 'params|ep60/60|done. best' | tail -3
# 3) re-eval closed-loop gt-seeded with the DAgger-retrained ckpt (GPU2), vs borrow#1
say "re-eval closed-loop gt-seeded (DAgger ckpt) vs borrow#1"
docker rm -f acot-policy-server >/dev/null 2>&1 || true
GSEED=gt CONF=0.5 EPISODES=20 SEED_BASE=91000 STEPS=900 GPU=2 \
  TAHCK=runs/tah_rel_dag.pt bash -c '
    sed "s#runs/tah_rel.pt#runs/tah_rel_dag.pt#; s#paper_ext_tah_\${GSEED}#paper_ext_tahdag_\${GSEED}#" \
      carla_uav_tracking/rollout/eval_tah.sh > /tmp/eval_tahdag.sh
    bash /tmp/eval_tahdag.sh' 2>&1 | grep -E 'SR |track_correct|max_wrong|q_reacq|DONE tah|metric' | tail -10
say "=== B1 DONE ==="
