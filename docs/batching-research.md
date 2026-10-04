# Batching / speedup research — TinyJev on BANKING77 (2026-10-04)

Status: research only (CPU side; no GPU experiments run — per instruction).
Reproduce: `tools/token_accounting.py` (tokenizer + family, CPU-only),
`tools/batch_probe.py` (the earlier GPU measurement).

## TL;DR

Yes — there is a viable path, and it is **algorithmic, not concurrency**. The current
execution spends ~90% of its FLOPs on *padding*, not on real tokens. Eliminating the
padding (split rows by width class + batch rows across texts) is worth roughly
**5–10×** by FLOP accounting, with the reference implementation (`kev`) already
containing every piece needed to port. Estimated full extraction: **~15–45 min
instead of ~110 min**.

## Measured facts (all on this box)

- Current extraction: **0.495 s/text** steady state. Perfectly linear in call size —
  one call with 4/8/16 states costs exactly 0.494/0.494/0.496 s per text (no
  amortization; `_run` loops per state internally).
- 4 concurrent replicas made it *worse* (1.9 s/text per replica, ~0.94 texts/s
  aggregate vs ~2 texts/s single-stream).
- ⇒ The workload is **compute-bound at ~34 TFLOPS effective** (~85% of RTX 3090
  fp16 peak). No kernel-efficiency headroom, no concurrency win. The only lever is
  **eliminating FLOPs**.

## Root cause (code-level)

Prompt layout (`tinyjev/families/pointer.py`): **one row per question**.
Our payload = 17 rows: 1 choice row over 77 options + 16 noul rows.

Real token sizes per text (400-text sample, `tools/token_accounting.py`):

| component | tokens (mean) |
|---|---:|
| state prefix | **13** |
| choice row (77 options) | **823** |
| noul row (max of 16) | **33** |

But `hidden_rows` (`tinyjev/backends/torch_backend.py`) pads **all 17 rows to the
widest row** → grid = 17 × (13 + 823) ≈ **14,212 positions/text**
≈ 16.9 TFLOPs/text ≈ **exactly the measured 0.495 s**. It all adds up: the box is
running at peak, on 90% padding.

Two dead ends confirmed by the same accounting:
- tinyjev's shared-prefix path only triggers at prefix ≥ 96 tokens; **0/400** texts
  qualify (B77 texts are tiny) — and even if forced, it saves only ~1.5%
  (14,004 vs 14,212 positions): the padding is *between row widths*, not the prefix.
- Multi-state batching re-runs the same padded grid per state (measured linear).

## Position grids and what a fix buys (mean positions/text)

| execution | grid | vs current |
|---|---:|---:|
| current (17 rows padded to choice width) | 14,212 | 1.0× |
| shared-prefix only (forced) | 14,004 | 1.0× |
| **split by width class** (choice alone + compact noul rows) | **1,572** | **9.0×** |
| split + state-prefix KV reuse | 1,364 | 10.4× |
| kev packed block-causal ideal (no padding) | 1,288 | 11.0× |

## What the reference implementation already has (`~/jev-models/kev`)

tinyjev is a slimmed serving port; kev is the full-speed sibling of the same design
and contains, working and tested:

- `kev/model.py::hidden_batch` — one forward for **a batch of records** under a
  **block-causal mask** (`branch_mask_batch`), each state run once, branches packed.
- `kev/model.py::forward_rows_batch` — row form where **rows of all records are
  batched together** and chunked by `rows_per_pass(...)` to bound memory.
- `probs_batch` + prefix cache — serving path: state encoded once per record,
  branches continue from its KV; “one device sync for the request, not one per
  question”.
- `kev/serve.py` — request queue where “the model thread takes everything queued
  when it becomes free and runs it as one batch” (MAX_BATCH), with an LRU prefix
  cache and OOM retries.
- Optional accelerators: `kev/cuda_graphs.py`, `kev/fused_qwen35.py` (~1/3 less GPU
  time per batch when `fla` is installed).
- Equivalence of the forms is documented and tested: “state + rows[k] as one causal
  row is equivalent to the packed block-causal form” and
  `tests/test_model.py::test_rows_match_packed`.

## Implementation options (ranked by effort/risk)

**A. Minimal batched runner on the existing tinyjev backbone** (recommended first)
- Reuse `family.encode` as-is (rows unchanged; plain form = independent causal rows).
- Collect rows from B records; **group by width class** (choice rows together, noul
  rows together); one padded forward per group via the loaded backbone
  (`agent.backbone.model`, exactly how `_rows_plain` calls it — attention mask
  zeroes pads); trim per row; reassemble each record's `hidden_rows` list in original
  order; call `family.logits` unchanged.
- No package edits, no new kernels. ~200 lines + validation. Expected ~5–7×
  (grid ratio 9×, haircut for small-kernel efficiency).

**B. A + prefix-KV reuse** — state once, branches continue from cache
(+~15% on top of A; more plumbing; optional).

**C. Port kev's packed block-causal path (+CUDA graphs)** — ceiling ~8–11×, but
more porting (and its checkpoint format differs; would need adapting to tinyjev
weights). Diminishing returns vs A/B for this workload (choice row dominates
regardless).

**D. Check whether `kev.serve` can load the TinyJev-0.6B checkpoint directly** —
unknown format compatibility; worth 15 min of inspection before building B/C.

## Verification — slice test (2026-10-04, done; GPU paused for it)

Triple-diff on the first 256 train texts (all with golden cache entries), same
session: fresh single-record run vs fresh batched run vs golden cache.
Tools: `tools/slice_diffdetail.py`; raw: `runs/banking77/slice_diffdetail.json`.

| pair | choice flips | noul max |Δ| | noul p95 | choice-prob max |Δ| |
|---|---:|---:|---:|---:|
| single vs golden (noise floor) | 0/256 | 5.0e-05 | 4.7e-05 | 1.2e-07 |
| batched vs golden | 0/256 | 3.11e-03 | 1.54e-03 | 3.1e-03 |
| batched vs single | 0/256 | 3.08e-03 | 1.54e-03 | 3.1e-03 |

- The current path is run-to-run deterministic (5e-5 = golden's own 4-dp rounding).
- The batched path introduces a real but small fp16 drift: ≤~3e-3 on probabilities,
  from different kernel shapes over the regrouped batches. **0 choice flips in 256**;
  noul deltas are feature-level negligible (the downstream classical models consume
  these as continuous features).
- **Speed: 502.6 → 63.0 ms/text = 7.98×** in the same process/session (excl. encode
  ~4 ms). Realized grid ratio 7.75× — matches the measured speedup (compute-bound,
  again). Full 13,083-text extraction ≈ ~15 min incl. load.

## Adoption — EXECUTED (2026-10-04): option 1

The batched path was adopted for the baseline's final data. Full run: 13,083 texts in
896 s = 68.5 ms/text (**7.3x vs the single-record path's ~503 ms/text**), one
consistent method, then reconcile + evaluate + full cross-check vs the single-path
backup: **4/3,547 near-tie choice flips (0.11%, all train), noul max |Δ| 3.5e-3,
none > 5e-3, deterministic across reruns**. Remaining options 2/3 are moot for this
baseline; the fast extractor stays available for the probe-discovery rounds (any
re-extraction now costs ~15 min instead of ~110).

Residual risk to note: near-tie choices could in principle flip on a rare row given
≤3e-3 perturbations; none observed in 256. A 1k-text slice can be run in ~2 min for
more confidence. The batched path does not change the cache schema, so the bundle's
`evaluate-jev` remains untouched either way.

<!-- superseded note: the first slice run (tools/batched_slice_test.py) reported
"30x vs cache median" — inflated, because first-256 cache latencies include the
4-way-contended shard-seeding window (median 1.87s). Correct reference: same-session
single-record 502.6 ms/text → 7.98x. -->


## Why it matters beyond this run

The bundle's follow-on protocol (probe discovery, teacher rounds) requires
**re-extraction of the training split every round** (probes_v2, v3, ...). At
~0.5 s/text each re-extraction is ~2 h; at ~0.1 s/text it's ~20 min — which changes
how many discovery rounds are practical. Same applies to any production use of the
Jev-family models on this box.
