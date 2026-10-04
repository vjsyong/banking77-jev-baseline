# Stage C operational notes and deviations

Date: 4 October 2026. Applies to the CLINC150 confirmation campaign (Stage C) of
`docs/frontier_guided_discovery_experiment_brief.md`. This file records operations-level
facts and deviations from the D-1 sequence; it does not alter the frozen protocol
(`discovery/fe_discovery/frozen_protocol.json`), the registered contrasts, the per-run mechanics, or any
teacher packet content.

## 1. Per-job memory cap discovery (pre-campaign, resolved)

Hermes background jobs run inside per-session systemd scopes with
`MemoryMax=4G` (`hermes-worker-proc_*.scope`). Five development/calibration jobs
were OOM-killed inside their own scopes (exit 137) while the machine itself was
not under pressure. Mitigation for campaign jobs (user-space, reversible):

- `systemctl --user set-property <scope> MemoryMax=24G`
- `oom_score_adj` 200 -> 0 for the job process.

No experiment data was affected (dev batteries and the first calibration only).

## 2. Hot-path measurements (serial campaign, U seed 11)

Per round, wall-time share (5-round run, ~45 min total):

- teacher call: 100-300 s (30-40% of wall) — network-bound; GPU idle.
- extraction of new probes: ~170 s (25%) — GPU burst.
- selector evaluations: ~90-150 s (20%) — CPU (7.5 s/eval, ~13-21 evals/round).
- remainder: parse, ledger, accounting (<10%).

## 3. Deviation D1 — 2-way parallel pool (active)

The registered campaign ran arms sequentially. From the switchover (4 Oct ~15:40 UTC),
arm-runs execute as independent subprocesses through `discovery/fe_discovery/stage_c_pool.py`
(concurrency 2, `FE_POOL_WORKERS`). Scientific rationale for acceptability:

- each arm run is an independent unit (own teacher conversation, own budget
  accounting, own artifacts); the registered paired comparisons are unchanged;
- seed setups remain strictly serial;
- no teacher packet content, prompt, parameter, or learner setting changes;
- latency measurements may carry mild GPU-contention noise; budgets are loose
  relative to realized costs (bank 20-40 ms vs limits 82/163/326 ms per text),
  and the final banks are re-measured wherever the protocol calls for it.

Run order queue: seed-major, arms U/R/E/F; completed runs (final.json present)
are skipped on resume. R seed 11 (partial under the serial process) restarts
from round 1 under the pool (D3).

## 4. Deviation D2 — operational accelerators (value-identical)

- FoldTFIDF writes a per-seed disk cache (`seed_S/tfidf_cache.pkl`, keyed by
  text/fold hash). Values identical to in-memory fits (reload-equality check
  passed). Correction to an earlier estimate: a fold fit is only ~4 s, so the
  cache is a small belt-and-braces saving, not a major hot path; the setup
  block's ~583 s was dominated by the one-time 150-option reference extraction
  (kept serial per seed).
- Pool workers: `FE_LR_JOBS=4` (OvR thread count; affects wall time only),
  `OMP/MKL_NUM_THREADS=6`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- The model, dtype, extractor, schema, selector, learner, and prompt templates
  are untouched by these changes.

## 4b. Concurrency control — cross-process eval gate (finding + fix)

First pool attempt exposed the campaign's true hidden hot path: with two workers,
the OvR-liblinear CV evaluations (150 binary fits x 5 folds per eval, ~7.5 s each
with 4 threads) spawned ~7 worker processes eating ~19 cores whenever two
selectors overlapped. That starved extraction's CPU-side tokenization: observed
extraction throughput dropped ~10x (store growth ~65 rows/s vs ~670 rows/s) and
the GPU sat at 0% utilisation waiting for data. Root cause: selector CPU storms
overlapping across processes, not GPU contention.

Fix: `Selector.eval()` now acquires a cross-process `flock` gate
(`stage_c/eval.lock`, via `eval_lock_path`); CV fits from different pool workers
never overlap, while teacher waits, extraction, and measurements still run
concurrently. Verified with a two-process lock test (second competitor blocks
until first releases). Worker env adjusted: `FE_LR_JOBS=6`, OMP/MKL=4.

No scientific impact: evaluation results are deterministic given the frozen
learner; the gate changes scheduling only. Latency noise caveat (D1) unchanged.

## 5. Pre-switchover crash log (no confirmation access)

Three startup crashes occurred during first launches, all before any run
completed a first round: protocol key names (`ceilings` nesting), option cap
(150-option reference question vs 6-option probe cap), intent-label list type.
Each was fixed and relaunched; U seed 11 completed under the serial runner
before the switchover. No confirmation-split data has been touched by any run
or by this file's authors.
