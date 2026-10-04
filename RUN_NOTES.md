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

## Results
- Classical (official test split): to fill from `runs/banking77/metrics.json`.
- Jev (direct + semantic-feature models): to fill after `evaluate-jev`.

## Caveats
- GPU0 (`0000:01:00.0`) is in an "Unknown Error" state on this box; all inference ran
  on GPU1. fp16 as shipped, no quantization.
- The manifest's latency stats blend ~640 in-process shard entries with the HTTP
  extraction entries; treat the latency summary as indicative, not a clean HTTP-only
  measurement.
- Results are local research data (text + labels retained per the bundle's own
  instructions; keep private per dataset terms).

## Batching probe (2026-10-04, mid-run)
Measured on a quiet GPU (extraction paused): multi-state batching gives ZERO speedup.
- separate calls: 0.494 s/text
- one call x4 states: 0.497 s/text | x8: 0.494 | x16: 0.496
Reason: `Agent._run()` loops over states internally (one forward per text); no shared work.
Per-text cost is fixed overhead (padding-waste in the 93-row batch, CPU hidden-state
copies, kernel-launch latency), not GPU FLOPs. Reproducer: `tools/batch_probe.py`.
