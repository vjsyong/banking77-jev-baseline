# Readout v1 — frozen pipeline, deployment parity, benchmark, confidence (2026-10-04)

The immediate deliverable of round 2: a reproducible, deployable readout for
**choice-only Jev → 77 log-probabilities → Logistic Regression**, with parity tests,
an end-to-end benchmark against TF-IDF, and an OOF-based confidence study.

## 1. Frozen artifact — `readout/banking77-readout-v1/`

- `readout_lr.joblib` (53 KB), `labels.json` (77 intents, alphabetical = feature
  order), `manifest.json` (full frozen config), `README.md`.
- **Config**: features `log(clip(p, 1e-6, 1.0))` over the 77 Choice probabilities →
  `StandardScaler → LogisticRegression(C=1.0, lbfgs, max_iter=3000)`; C selected by
  3-fold stratified CV on train.
- **Upstream**: `AnkitAI/TinyJev-0.6B` (HF snapshot `c559c2f7ea95…`), tinyjev 0.1.3,
  torch 2.14.1+cu130, transformers 5.18.0; probabilities extracted with the batched
  width-grouped extractor (`tools/batched_extract.py`, chunk 64).
- **Quality**: macro-F1 **0.9008** · acc **0.9006** · top-3 **0.9766** (official test,
  3,080 rows). Reproduce: `readout/export_v1.py` (deterministic; asserts the metrics).

## 2. Deployment parity — `readout/test_parity.py` (5/5 PASS)

- (a) saved artifact reproduces the experimental test metrics;
- (b) **shuffled probability keys** → identical predictions (250 rows);
- (c) **tiny/zero probabilities** clipped safely (incl. all-zero vector) — finite
  outputs, deterministic, no `log(0)`;
- (d) **single vs batched** requests identical (100 rows), dict and array forms agree;
- (e) guard rails: missing intents / wrong width raise `ValueError`.

## 3. Benchmark — complete pipeline, text → prediction (`readout/benchmark.py`)

| stage | Jev choice-only + readout | TF-IDF + LR |
|---|---:|---:|
| single request p50 | **62.5 ms** | **2.29 ms** |
| single request p95 | 68.2 ms | 2.89 ms |
| batched, per text | **30.2 ms** | **0.229 ms** |
| batched throughput | **33.1 texts/s** | **4,374 texts/s** |

Jev breakdown (single): encode 2.6 ms · forward 58.0 ms · head 0.65 ms · readout
1.35 ms. Readout alone: 1.2 ms single-row; 0.60 ms/text in a 512 batch.
Hardware: RTX 3090 fp16 (in-process) vs CPU; warmups excluded.

The readout step is negligible in both pipelines; the Jev forward dominates. TF-IDF
remains dramatically cheaper per inference (the deployment tradeoff is explicit).

## 4. Confidence — accuracy vs coverage (`readout/confidence.py`)

Thresholds selected on **train-fold OOF** predictions (5-fold, seed 77); test split
reported only.

| τ | OOF coverage | OOF accuracy (accepted) | test coverage | test accuracy (accepted) |
|---:|---:|---:|---:|---:|
| 0.335 | 98.9% | 90.0% | 98.8% | 90.8% |
| **0.645** | 86.7% | 95.1% | 88.1% | 95.4% |
| 0.85 | 72.9% | 98.1% | 74.2% | 98.0% |

Recommended default: **τ = 0.645** (~95% accepted-accuracy at ~87–88% coverage);
deferrals would route to a human or the TF-IDF model (not yet wired).

## 5. Notes

- Probe-bank v2 is paused; its acceptance rule (recorded): **Pareto improvement** —
  higher quality at acceptable cost, or comparable quality at lower cost, not a fixed
  bar against 0.9008.
- Raw outputs: `runs/banking77/analysis/{pipeline_benchmark,confidence_coverage}.json`;
  TF-IDF refit artifact: `runs/banking77/tfidf_lr_refit.joblib` (reproduce:
  `readout/fit_tfidf_ref.py`).

## Reproduce

```bash
./venv/bin/python readout/export_v1.py         # rebuild + verify the bundle
./venv/bin/python readout/test_parity.py -v    # 5 parity tests
./venv/bin/python readout/confidence.py        # OOF threshold study
./venv-serve/bin/python readout/benchmark.py   # end-to-end benchmark (GPU)
```
