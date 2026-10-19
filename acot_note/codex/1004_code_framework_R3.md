VERDICT: APPROVE-WITH-MINOR

## R2 items
M1: RESOLVED — ownership is now consistent: `control` is method-owned in §5.1/§5.3 and mappings (`acot_note/1004_code_framework.md:201`, `:267`, `:459-462`, `:504`), while scorer `commit_k` stays protocol-owned (`:258`, `:483`).

M2: RESOLVED — generator is glob-based over in-scope Python/shell (`acot/tools/legacy_surface.py:26-28`), now captures prior missing train flags (`acot_note/1004_legacy_surface.md:267-280`, `:339-351`, `:359-373`, `:406`) and launcher fixed flags/constants (`:615`, `:727`).

M3: RESOLVED — P1 now audits raw `tpos`/`dstates`/`vis`, has site categories, positive controls, and `gt_fence` fallback (`acot_note/1004_code_framework.md:607-638`), matching current uses in `policy.py:657`, `policy.py:661-663`, `policy.py:1080`, `policy.py:1094`.

m1: RESOLVED — `cand_lite` missing `gt_idx` is explicitly called out (`acot_note/1004_legacy_surface.md:599`; `acot_note/1004_code_framework.md:883`) and matches current `_cand_lite` (`policy_server.py:38-46`) vs scorer expectation (`metrics.py:29-32`).

m2: RESOLVED — reset seed is now specified as log-only and not fed to `_rng` (`acot_note/1004_code_framework.md:708-710`), matching current no-seed reset and fixed shuffle reseed (`remote_policy.py:23-25`, `policy.py:279-286`).

m3: RESOLVED — output/cache/checkpoint paths are artefact-managed, not infra (`acot_note/1004_code_framework.md:419-420`, `:520`; inventory examples `1004_legacy_surface.md:270-271`, `:350-351`, `:361`).

## MAJOR
None.

## MINOR
1. Problem: “everything not scanned is listed” is still slightly overstated.
Evidence file:line: generator scans only declared scopes (`acot/tools/legacy_surface.py:26-28`) and excludes some roots (`:29-41`), but unscanned `acot_note/figs/*.py` exists and is not listed in the exclusion table.
Required fix: add explicit exclusions for `acot_note/**`/paper-result/report-only roots, or change the wording to “in-scope omissions are listed.”

2. Problem: `dam4sam_service.py --model` is captured but not called out in §6.3 mappings.
Evidence file:line: inventory marks `--model` scientific (`acot_note/1004_legacy_surface.md:52`); infra hardcodes service model in `start_cmd` (`acot_note/1004_code_framework.md:405`) while §6.3 only maps host/port/every-frame/cooldown (`:511`).
Required fix: map service model explicitly, either as frozen protocol/method identity field or as an infra-pinned service version with manifest echo.

## Suggestions
Keep the P1 audit whitelist keyed to normalized source plus function, as written, but make `AuditedTuple` recursively wrap `vis[1]` so future code cannot unpack once and read the distractor visibility list invisibly.