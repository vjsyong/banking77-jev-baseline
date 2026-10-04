# Run notes — BANKING77 Jev + classical baseline (2026-10-04)

## Origin
- Received `Banking77_Jev_Baseline_Bundle.zip` via chat 2026-10-04; imported here as commit `b254ba8` (unmodified).
- Related prior work: `~/tinyjev-spike/` (Oct 2–3 TinyJev spike) — not modified by this run.
- The bundle's Ollama/`parable/tinyjev` route was NOT needed: a local TinyJev install
  already exists on this machine (`tinyjev` 0.1.3 package + `AnkitAI/TinyJev-0.6B`
  cached, HF snapshot `c559c2f7ea95`).

## Environment
- Bundle venv (`venv/`): Python 3.12.3, datasets 3.6.0, numpy 2.5.3, scikit-learn 1.9.1.
- `HF_DATASETS_TRUST_REMOTE_CODE=1` is required in this environment: datasets 3.x
  refuses the legacy BANKING77 loading script non-interactively otherwise. The bundle
  code was left unmodified (env var only).
- Serve venv (`venv-serve/`): tinyjev 0.1.3, torch 2.14.1+cu130, transformers 5.18.0.
  Fresh venv because the spike venv is deliberately CPU-only (torch 2.14.1+cpu) and
  this box has a usable RTX 3090 (GPU0 is in an error state; GPU1 healthy, driver 590.48).

## Jev endpoint
- Served with: `tinyjev serve TinyJev-0.6B --backend torch --device cuda` →
  `http://127.0.0.1:8077/v1/systemone` (System One shape matches the bundle's parser
  exactly: flat `answers` map, `noul` floats, `choice` + `probabilities`).
- HTTP path smoke-validated with the bundle's own `--max-rows-per-split 2` (4/4 OK).

## Extraction (the full run)
- Workload: 13,083 texts (10,003 train + 3,080 test), 17 questions per text
  (77-way direct choice + 16 noul probes), one request per text.
- Steady-state single-stream latency ≈ **0.50 s/text** on the 3090 (measured).
- Parallelism experiment (for the record): 4 concurrent in-process workers made
  per-text latency ~1.9 s and aggregate throughput *worse* than single-stream
  (~0.9–1.0/s vs ~2/s) — the model's forwards serialize on the GPU and 4 CUDA
  contexts add switching overhead. Single-stream is optimal for this workload.
- Main extraction therefore runs the bundle's own `extract-jev` (workers=1) against
  the local endpoint; the SQLite cache is resumable (first ~640 texts were seeded
  by shard workers, whose output was verified equivalent to the HTTP path:
  `tools/compare_caches.py` — max noul diff 0, max prob diff 1.2e-7, choices equal).
- ETA ≈ 115 min from 2026-10-04 ~03:52 HKT.

## Tooling added (this dir, tracked in git)
- `tools/shard_extract.py` — sharded in-process extractor (used for seeding/validation;
  writes the bundle's cache format via the bundle's own functions).
- `tools/compare_caches.py` — cache equivalence checker (HTTP vs in-process).
- `tools/latency_probe.py` — single-stream latency probe for the endpoint.

## Update — extraction method switched to the batched path (final)

Mid-run, extraction was switched from the single-record HTTP path (~0.50 s/text) to
the productionized batched extractor (`tools/batched_extract.py`: width-grouped,
cross-record, in-process; validated 0/256 flips + 7.98x on a 256-text slice). All
13,083 texts were then re-extracted in one consistent pass in **896 s (68.5 ms/text,
7.3x)**, followed by `extract-jev` reconciliation (0 fetches), `evaluate-jev`, and a
full cross-check vs the single-path backup (`runs/banking77/jev_cache.single_path_backup.sqlite`):

- 3,547 common rows: **4 choice flips (0.11%; near-ties, all train rows)**
- noul: max |Δ| 3.5e-3, p95 1.5e-3, mean 6.1e-4, **none > 5e-3** (56,752 comparisons)
- deterministic: a full rerun reproduced identical flips/deltas

## Results (final, official test split)
- Classical: LR macro-F1 0.9094 / acc 0.9091; LinearSVC 0.9087 / 0.9084; CNB 0.8068 / 0.8143.
- Jev direct choice (TinyJev-0.6B): macro-F1 **0.7574**, acc **0.7633**, top-3 acc **0.9078**;
  batched per-text cost p50 **67.9 ms** (p95 86.6 ms).
- Jev semantic-feature models (16 probes): LR 0.4038; LinearSVC 0.4307; ExtraTrees **0.5707** (macro-F1).
  => the classifier-on-probes hypothesis fails for the initial 16-probe bank (as the plan's
  limitations section anticipated: 16 broad signals do not encode 77 intent distinctions).

## Caveats
- GPU0 (`0000:01:00.0`) is in an "Unknown Error" state on this box; all inference ran
  on GPU1. fp16 as shipped, no quantization.
- Batched extraction introduces <=0.11% near-tie choice flips vs the single-record
  path (4/3,547 compared rows, all train; drift <=3.5e-3 on noul / probabilities;
  fp16 kernel-shape noise). Reported latencies are true batched values (~68 ms/text).
- Results are local research data (text + labels retained per the bundle's own
  instructions; keep private per dataset terms).

## Batching probe (2026-10-04, mid-run)
Measured on a quiet GPU (extraction paused): multi-state batching gives ZERO speedup.
- separate calls: 0.494 s/text
- one call x4 states: 0.497 s/text | x8: 0.494 | x16: 0.496
Reason: `Agent._run()` loops over states internally (one forward per text); no shared work.
Per-text cost is fixed overhead (padding-waste in the 93-row batch, CPU hidden-state
copies, kernel-launch latency), not GPU FLOPs. Reproducer: `tools/batch_probe.py`.
