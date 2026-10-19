VERDICT: MAJOR-REVISIONS

## MAJOR

1. §5 mapping is not complete enough to be a “complete收编清单”.
Problem: several behavior-affecting flags/env/JSON keys/constants are omitted or only hidden under broad buckets, so a translator could silently miss behavior.
Evidence: `policy_server.py` exposes `--ablate`, `--device`, `--reid-feature`, `--reid-bank`, `--reid-topm`, `--reid-tavg`, `--reid-motion`, `--reid-motion-gate-px`, `--reanchor-patience`, `--reanchor-tavg`, `--assoc-mode`, `--assoc-lambda`, `--dam4sam-port` at `carla_uav_tracking/rollout/policy_server.py:101-162`; `run_rollout.py` has `--policy-host`, `--policy-port`, `--carla-port` at `carla_uav_tracking/rollout/run_rollout.py:82-84`; policy has a non-exposed `shuffle_seed` reset every episode at `carla_uav_tracking/rollout/policy.py:99-120` and `carla_uav_tracking/rollout/policy.py:280-286`.
Evidence: JSON key surfaces are larger than §5 shows: ASD has `learned_ckpt`, `anchor_mode`, `eps_switch`, `q_lo/q_hi`, `alpha`, `naive`, `gate_tah`, `gate_dam`, etc. at `carla_uav_tracking/rollout/asd.py:23-30`; TRK_CFG includes `vel_beta` and `app_alpha` at `carla_uav_tracking/rollout/tracklets.py:27-29`; OPENING_CFG includes `agg`, `final_reset`, `reset_asd`, `force_final`, `seed_asd` at `carla_uav_tracking/rollout/policy.py:196-209`, plus `ckpt/wm/wc/wq/ft` at `carla_uav_tracking/rollout/opening_zoom.py:116-137`.
Evidence: launcher knobs include `HF_CACHE`, `NET`, `CONFIG`, `S2_PERIOD`, `SRV`, `NTHR`, `DUMP_LABELS`, `TM_PORT`, `VIS_RAYCAST`, `HIRES` in `carla_uav_tracking/rollout/eval_asd.sh:7-27`, and watchdog knobs/adapter map in `runs/asd/iter/launch_wd.sh:6-10`.
Required fix: add an exhaustive generated inventory table: every argparse flag, env read/write, JSON key, shell default, and hard-coded behavior constant. Mark each as scientific config, diagnostic, collection/logging, or infra; unknown legacy env/JSON keys should fail in P2.

2. GT fencing is directionally right but not mechanically safe in P0/P1.
Problem: §6 says logging-only GT will be isolated later, but the current wire protocol still sends broad GT to policy, and several GT channels need explicit classification now.
Evidence: env sends `gt_candidates=(tpos, dstates, campose, road_pred_world, vis)` to the policy at `carla_uav_tracking/rollout/env.py:291-299`; remote bridge sends `target_color`, `target_bp`, and `gt_candidates` at `carla_uav_tracking/rollout/remote_policy.py:27-34`; `attrbind` consumes target color/blueprint directly at `carla_uav_tracking/rollout/policy.py:694-699`; road oracle updates from true target every tick at `carla_uav_tracking/rollout/env.py:283-287`.
Evidence: collection/logging GT is spread across DAgger, TAH DAgger, opening dumps, and labels at `carla_uav_tracking/rollout/policy.py:1057-1060`, `carla_uav_tracking/rollout/policy.py:919-923`, `carla_uav_tracking/rollout/policy.py:729-744`, and `carla_uav_tracking/rollout/env.py:177-186`.
Required fix: define the fence as code-level interfaces earlier: `PolicyInput` must exclude `is_target/true_idx/gt_idx/target_color/target_bp` unless the arm is diagnostic; `ScoringInfo`/collector objects carry them. Add tests that diagnostic arms are labeled and deployable arms cannot access these fields.

3. Strangler parity criteria are too weak for “no silent behavior change”.
Problem: P1 accepts “first decision same” on 2 seeds, but current behavior depends on skipped episodes, tick-log ordering, reset semantics, partial files, and scorer coupling.
Evidence: P1 criterion is only first decision equality at `acot_note/1004_code_framework.md:311-313`; current reset carries no seed at `carla_uav_tracking/rollout/policy_server.py:65-67` and `carla_uav_tracking/rollout/remote_policy.py:23-25`; tick logs are aligned to seeds by order because skipped episodes do not reset policy at `carla_uav_tracking/rollout/tick_align.py:1-3`; `opening_cl.py` also zips sorted log episode ids to non-skipped scorer rows at `carla_uav_tracking/rollout/opening_cl.py:36-51`.
Evidence: scorer `commit_k` defaults independently at `carla_uav_tracking/rollout/metrics.py:35-41`, while policy `--commit-k` is separate at `carla_uav_tracking/rollout/policy_server.py:111`.
Required fix: parity should include dry-run exact legacy command/env equality, resolved-config hash echo, per-tick seed-bearing logs, same skipped/failed seed set, scorer/policy `commit_k` binding, and metric equality on imported legacy outputs. Live CARLA parity can tolerate trajectory nondeterminism, but not invocation, seed, or scoring drift.

4. Offline plan addresses known failures, but compatibility metadata is still under-specified.
Problem: §8 catches FOV, splits, output-dir conflicts, and tick mismatch, but the current cache/checkpoint surfaces need stronger manifests than described for Stage-2/context/DINO/opening/DAgger.
Evidence: missing split currently falls back to naive train/val and empty test at `train/dataset.py:25-39`; `config_v5.yaml` and `config_v5_fov90.yaml` both use `stage2_ckpt_dir: runs/stage2_v5` at `train/config_v5.yaml:49-55` and `train/config_v5_fov90.yaml:49-55`; TAH cache stride defaults to 2 at `train/tah_cache.py:26-31`, while closed-loop S2 period defaults to 6 in `carla_uav_tracking/rollout/eval_asd.sh:8-9`.
Evidence: TAH cache only saves `d_feat`, `order`, and `fov`, not feature id/tick_s/input hashes, at `train/tah_cache.py:67`; TAH training only asserts `d_feat` compatibility at `train/tah_train.py:79-82`; context caches skip existing files without config/hash validation at `train/backbone_kv.py:63-72` and `train/backbone_kv.py:106-116`.
Required fix: require manifests for all offline artifacts, including context caches, DINO crops, TAH caches, DAgger merges, opening snapshots/classifiers, and Stage-2 ckpts. Compat must compare `fov_deg`, `tick_s`, crop size/pad, DINO model, depth scale, language mode, split manifest hash, and dataset manifest hash.

5. Results/analysis design needs stricter completeness and partial-file rules.
Problem: §7/§9 moves in the right direction, but legacy import and analysis can still silently pair different seed sets or consume partial outputs inconsistently.
Evidence: `iter_gate.py` drops skipped/unscored episodes and first occurrence wins at `carla_uav_tracking/rollout/iter_gate.py:74-88`; parts prefer final over partial at `carla_uav_tracking/rollout/iter_gate.py:91-107`; `opening_cl.py` selects `js[-1]` from `*.json*`, which may prefer `.partial` by sort order at `carla_uav_tracking/rollout/opening_cl.py:26-33`; watchdog merges whatever part files exist at `runs/asd/iter/launch_wd.sh:60-71`.
Required fix: new analysis should fail closed unless an experiment explicitly allows partial/skipped seeds. Store per-seed status `{complete, skipped, failed, partial}`; compare only preregistered seed sets by default; import legacy results with a deterministic final-vs-partial precedence and warnings.

6. Infra/runtime constraints are not explicit enough.
Problem: the design mentions py3.8 and coordination, but it does not make the existing Docker/network/service constraints enforceable.
Evidence: current policy launch requires `--network cyh-carla-net` at `carla_uav_tracking/rollout/policy_server.py:7-9` and `carla_uav_tracking/rollout/eval_asd.sh:7-13`; DAM4SAM service is manually launched with model/port only at `carla_uav_tracking/rollout/dam4sam_service.py:11` and consumed via host/port flags at `carla_uav_tracking/rollout/policy_server.py:160-162`; code comments still mention py3.6 CARLA env and py3.11 policy bridge at `carla_uav_tracking/rollout/env.py:17-19` and `carla_uav_tracking/rollout/bridge.py:1-8`.
Required fix: add infra schema fields for existing network only, container/image names, GPU ownership, port ownership, DAM4SAM lifecycle/health checks, and CARLA client Python target. Add py3.8 syntax checks for CARLA-side modules and keep config package 3.8-compatible.

## MINOR

1. Effort estimate is optimistic. The doc estimates P0-P3 at 5-6 days at `acot_note/1004_code_framework.md:308-317`, but the actual surface spans policy/env/scorer, launchers, offline caches, DAM4SAM, and multiple analysis importers. Split P0 into inventory/schema and translator golden tests.

2. Seed registry should support arbitrary seed lists, not only continuous blocks. Current launchers use blocks, but retries/substitutes already exist in data generation scripts such as `runs/tah_v90/gen_fov90b.sh:2-5`.

3. “Deployable v2” wording should be softened. The protocol still allows sim-projected candidate boxes, GT depth, raycast occlusion, and matched GT world localization in `acot_note/1004_code_framework.md:130-145`; call it “sim-proxy protocol” unless there is a real detector/localizer path.

4. `acot data gen` is underspecified. It needs a frozen seed→town→scenario plan, retry/substitution policy, CARLA port/TM allocation, and QC gate, not just a dataset YAML output.

## Suggestions

Add one generated `legacy_surface.md` from small introspection scripts before implementing the schema. That gives the translator a target and makes “complete mapping” testable instead of aspirational.