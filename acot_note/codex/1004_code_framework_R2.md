VERDICT: MAJOR-REVISIONS

## R1 items

M1: PARTIALLY — inventory/map plan is much better, but the generator is not exhaustive and its output misses real `train/` flags and launcher fixed flags.

M2: PARTIALLY — P2 `PolicyInput`/`ScoringInfo` split is sound, but P1 audit does not mechanically fence raw `gt_candidates` (`tpos`/`dstates`) still passed to `Policy.act`.

M3: PARTIALLY — L0-L3 parity is substantially improved; however control-field ownership and `commit_k` schema placement are internally inconsistent.

M4: RESOLVED — offline artifact manifest/compatibility coverage now addresses the known cache, FOV, split, tick, and provenance failures.

M5: RESOLVED — per-seed states, fail-closed defaults, and deterministic legacy final/partial precedence are adequate.

M6: RESOLVED — network, container, Python, GPU/port leases, and DAM4SAM single-client handling are now enforceable in the design.

m1: RESOLVED — effort is split into P0a/P0b/P1 and expanded to realistic staged milestones.

m2: RESOLVED — seed registry supports ranges, explicit lists, list files, excludes, and substitutes.

m3: RESOLVED — `deploy_v2` wording is replaced by `sim_proxy_v2`.

m4: RESOLVED — datagen now has frozen seed-town plan, retry/substitution, leases, QC, staging, and manifest flow.

## MAJOR

1. Problem: Control/scientific fields are split inconsistently between protocol and method, making existing arm differences either illegal or ambiguous.
Evidence: §5.1 says arm overrides can only cover `method.*` and protocol overrides are mostly rejected (`acot_note/1004_code_framework.md:179-184`), and says policy `commit_k` is `method.control.commit_k` (`acot_note/1004_code_framework.md:188-191`). But the protocol example owns `control: {ex_source, frame_gain, commit_k, shuffle_seed, ablate}` (`acot_note/1004_code_framework.md:228`), while §6.3 maps `--ex-source`, `--frame-gain`, `--ablate`, `shuffle_seed` to `protocol.control.*` and `--commit-k` to `method.control.commit_k` (`acot_note/1004_code_framework.md:445-446`). These are real behavior knobs in `policy_server.py:102-111,125-128` and `policy.py:120-129,644-657,1063-1084`.
Required fix: choose one ownership model. Put per-arm policy behavior knobs under `method.control.*`/method head configs, keep protocol for shared sim/perception/scoring invariants, and update schema, legacy map, examples, and hash rules consistently.

2. Problem: The generated inventory is not exhaustive enough to support “every behavior-affecting channel” tests.
Evidence: `tools/legacy_surface.py` hard-codes a short `ARGPARSE_FILES` list (`tools/legacy_surface.py:20-23`) while the repository has behavior-affecting `train/` CLIs outside that list, e.g. `train/bc_center_train.py:178-192`, `train/reid_metric_train.py:171-185`, `train/qwen_lora_grounding.py:188-200`, `train/stage1_ear.py:103-105`, and `train/make_split.py:87-90`. Launcher extraction also omits fixed flags such as `--config`, `--ckpt`, `--language-mode`, `--port`, `--policy-host`, even though scripts pass them (`carla_uav_tracking/rollout/eval_asd.sh:17-27`); the inventory row for `eval_asd.sh` lists only a subset (`acot_note/1004_legacy_surface.md:292`).
Required fix: make inventory discovery glob/AST-based for all scoped `train/*.py` and rollout analysis/prep scripts, or explicitly mark excluded files with reasons. For launchers, capture actual argv/env via shell/dry-run or a broader parser, not a small regex allowlist.

3. Problem: P1 GT audit does not fence raw GT arrays, so P1 can still miss decision-time use of `tpos`/`dstates`.
Evidence: §7.3 P1 audits `CandidateSet.true_idx`, `Candidate.is_target`, `Candidate.gt_idx`, and `Observation.target_color/target_bp` only (`acot_note/1004_code_framework.md:527-533`). But policy unpacks raw `tpos, dstates, campose` from `gt_candidates` (`carla_uav_tracking/rollout/policy.py:653-657`), uses them for WHERE/localization (`carla_uav_tracking/rollout/policy.py:1080-1084`), and non-track/ex-source paths directly update from true `tpos` (`carla_uav_tracking/rollout/policy.py:1092-1104`). P1 wire blanking leaves `gt_candidates=(tpos,dstates,campose,road_pred_world,vis)` on the wire (`carla_uav_tracking/rollout/env.py:291-298`, `remote_policy.py:27-34`).
Required fix: in P1, wrap/audit `gt_candidates` slots too, or explicitly state P1 only detects label/property leaks and cannot prove raw-GT nonuse until P2. Whitelist `build_candidates` and approved sim-proxy localization call sites; fail any other raw `tpos`/`dstates` reads in deployable arms.

## MINOR

1. `reply.cand_lite` inventory should call out the current missing `gt_idx`. `_cand_lite` sends only `is_target`, `idx`, `depth`, `true_idx` (`policy_server.py:38-46`), while scoring needs `gt_idx` when tracker rekeys identities (`metrics.py:29-32`). The design fixes this in P2 (`acot_note/1004_code_framework.md:523`), but the inventory row (`acot_note/1004_legacy_surface.md:276`) should mark it as a known current gap.

2. Reset-seed parity is feasible, but the wording should be tightened. Current `RemotePolicy.reset()` sends no seed (`remote_policy.py:23-25`), `Policy.reset()` takes no seed (`policy.py:279`), and RNG is reseeded from fixed `_shuffle_seed` every reset (`policy.py:286`). Adding a seed-only log field is decision-neutral if no extra reset call is introduced and the seed is not fed into `_rng`.

3. Launcher inventory classifies several output paths as `infra` even when they identify scientific artifacts/caches, e.g. `--out`, `--cache`, `--save-ckpt` in training scripts. Treat artifact identity/output destinations as recipe/artefact fields when they determine reuse or downstream inputs; reserve `infra` for host placement only.

## Suggestions

Run `tools/legacy_surface.py` against a denylist-free glob first, then explicitly suppress irrelevant scripts with comments. That makes omissions auditable.

Add one P1 test with an intentional direct `tpos` read in a deployable arm; it should fail before P2, or the design should stop claiming P1 mechanically fences that class of leak.