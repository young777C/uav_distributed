# acot_probe — layer-16 grounding probe (VLM backbone selection gate)

Implements the **§3.4 selection验证** from `acot-uav-design.md`: before committing a
frozen VLM backbone, measure whether its **layer-16** representation already lets a
tiny head **select the language-referred target vehicle among visually-similar
distractors**. Compares **PaliGemma-2-3B** (18-layer, truncation-friendly) vs
**Qwen3-VL-4B** (deep 36-layer) on your own CARLA data.

Metric: **Target-Selection Accuracy (TSA)** = P(argmax score == target), per layer.
Chance = mean(1 / #candidates). A backbone "passes" if TSA@16 ≫ chance and ≥ the
`pass_tsa_at_16` threshold. Expected pattern (the whole point): PaliGemma passes at
16; Qwen is weak at 16 but climbs by ~20–24 → confirms the truncation trade-off.

## How it works

```
episode.h5 (target/distractor 3D + cam pose)        [no CARLA needed]
   └─ projection.py   → project each vehicle to a candidate box in the image
   └─ backbones.py    → frozen VLM, ONE forward, per-layer image grid + text feat
   └─ data.py         → pool grid inside each box → cache rows (feat, group, is_target)
   └─ probe.py        → tiny selection head per (backbone,layer) → TSA vs chance
   └─ run_probe.py    → orchestrate + backbone×layer table + verdict
```

## Quick start

```bash
cd uav-acot-track
pip install -r acot_probe/requirements.txt

# 0) plumbing check — no models, no CARLA, no GPU:
python -m acot_probe.smoke_test

# 1) generate a SMALL similar-distractor probe set (needs CARLA), e.g. ~15 episodes
#    of the urban_hard_similar scenario, into data/probe/ :
python carla_uav_tracking/scripts/generate_batch.py \
    --scenario carla_uav_tracking/config/scenarios/urban_hard_similar.yaml \
    --n 15 --out data/probe        # (flags per your generator; goal: data/probe/episode_*.h5)

# 2) VERIFY projection lands boxes on the right cars (do not skip):
python -m acot_probe.visualize --config acot_probe/config.yaml \
    --episode data/probe/episode_000000.h5 --frames 8 --out runs/proj_check.png

# 3) run the comparison (PaliGemma on cuda:0, Qwen3-VL on cuda:1):
python -m acot_probe.run_probe --config acot_probe/config.yaml
```

Output: a `backbone × layer` TSA table + a PASS/WEAK verdict at layer 16, and
`runs/probe_out/probe_results.json`.

## 4×A40 notes

- Probe is **frozen-backbone + tiny head** → memory-light; a 3–4B model in bf16 is
  ~7–9 GB, fits one A40 (48 GB). Config puts PaliGemma on `cuda:0`, Qwen on `cuda:1`
  so both cache in parallel. The probe head trains on CPU (`probe.head_device`).
- Only ~10–20 episodes are needed for a selection signal; keep it small and fast.

## Projection caveat (read before trusting numbers)

The recorder stores 3D vehicle **centers + camera pose**, not true 2D boxes, so
`projection.py` reprojects offline using CARLA's camera convention and draws a
**distance-scaled pseudo-box**. Step (2) `visualize.py` exists to confirm the boxes
land on the vehicles. If they don't, the cleanest fix is to record **true 2D bboxes
for target + every distractor at generation time** (CARLA `actor.bounding_box`
projected with the real camera matrix) into `annotation/` — this also feeds the
Stage-2 `L_target_id` head (design §4.3), so it's worth doing once.

## Backbone token-grid mapping

`backbones.py` maps image tokens → 2D grid from processor outputs (PaliGemma:
`sqrt(image_seq_length)`; Qwen: `image_grid_thw // spatial_merge_size`). These are
version-sensitive; `visualize.py` + a real-image check is the guard. Qwen prompt
puts **text before image** so image tokens can attend to the instruction under
causal attention (keeps the fused features language-aware for a fair comparison).

## Files

| file | role |
|---|---|
| `projection.py` | CARLA-consistent 3D→pixel projection + candidate extraction |
| `backbones.py` | frozen PaliGemma-2-3B / Qwen3-VL-4B wrappers + box pooling |
| `data.py` | per-frame feature caching to `.npz` |
| `probe.py` | selection head + group-wise train/eval (TSA) |
| `run_probe.py` | orchestrator + report |
| `visualize.py` | projection sanity overlay |
| `smoke_test.py` | dependency-free end-to-end plumbing test |
| `config.yaml` | backbones, layers, paths, thresholds |
```
