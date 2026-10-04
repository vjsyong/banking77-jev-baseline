# Consolidated report — BANKING77 Jev engagement (2026-10-04)

Project: `~/banking77-jev-baseline` · commits `b254ba8 → 3839fdf` · nothing running.
Everything below traces to a request from this thread; artifacts are committed in the repo.

## 1. Run the Banking77 Jev baseline bundle (as received)

- Setup: fresh Python 3.12 venv (datasets 3.6.0 / numpy 2.5.3 / sklearn 1.9.1); the
  legacy BANKING77 loader needs `HF_DATASETS_TRUST_REMOTE_CODE=1` in this environment
  (env var only — bundle code unmodified). Bundle unit tests 6/6.
- Jev endpoint: local TinyJev-0.6B (existing install; fresh CUDA serving env because
  the spike venv is CPU-pinned). GPU0 is in an error state on this box; all inference
  ran on GPU1 (RTX 3090, fp16).
- **Final results, official test split (macro-F1 / accuracy / weighted F1):**
  - TF–IDF LogReg: **0.9094 / 0.9091 / 0.9094**
  - TF–IDF LinearSVC: **0.9087 / 0.9084 / 0.9087**
  - TF–IDF ComplementNB: 0.8068 / 0.8143 / 0.8068
  - Jev direct 77-way choice: **0.7574 / 0.7633 / 0.7574** (top-3 0.9078)
  - Jev 16 probes → LogReg: 0.4038 / 0.4295; → LinearSVC: 0.4307 / 0.4614;
    → ExtraTrees: 0.5707 / 0.5773
- Cost descriptors: extraction 13,083 texts in 896 s (68.5 ms/text batched);
  decision latency mean 68.4 / p50 67.9 / p95 86.6 ms; 16 probes; 0 uncached at reconcile.
- Files: `runs/banking77/{metrics.json,jev_manifest.json,jev_features.csv,jev_predictions.csv}`.

## 2. Locate the prior TinyJev setup (home path / opencode)

- Found `~/tinyjev-spike` (spike + reports), `~/mail-triage-jev` (harness), `~/jev-models`
  (kev, NanoJev), HF cache (TinyJev-0.6B snapshot `c559c2f7ea95…`), and the opencode
  session "tinyjev experiment setup walkthrough" which documented the whole setup.
  Reused the model + knowledge; box left as found apart from this project.

## 3. Batching investigation (two requests: "can inputs be batched" / "viable shot, research only")

- Measured: multi-state batching gives **zero amortization** (0.494 vs 0.497 s/text —
  `_run` loops internally); 4 parallel replicas were *worse* (GPU serializes).
- Research finding: the run was **compute-bound on padding** — 14,212 padded positions
  per text vs 1,288 real (~90% waste), entirely because all 17 rows pad to the
  823-token 77-option choice row. Fix space: 9–11× fewer positions.
- Reference implementation (`~/jev-models/kev`) already contains the full machinery
  (block-causal packed batch, row batching, prefix cache, batching server).
- Artifacts: `docs/batching-research.md`, `tools/{batch_probe,token_accounting}.py`.

## 4. Slice equivalence test → switch to the batched path

- 256-text triple-diff (cache vs fresh single vs fresh batched): **0/256 choice flips**;
  noul drift ≤3.1e-3 (fp16 kernel-shape noise; single-vs-single floor 5e-5);
  speed 502.6 → 63.0 ms/text (**7.98×**).
- Productionized `tools/batched_extract.py` (cache-compatible: same keys, System-One
  payload shapes; `evaluate-jev` untouched). Full re-extraction in one consistent pass:
  13,083 texts, 896 s, **7.3×**, 0 errors.
- Full cross-check vs single-path backup: 3,547 common rows, **4 near-tie flips
  (0.11%, all train)**, noul max 3.5e-3 (none >5e-3), deterministic on rerun.
- Single-path cache preserved: `runs/banking77/jev_cache.single_path_backup.sqlite`.

## 5. "Report just the metrics the README requested"

Delivered (see §1). Metrics table above is exactly the README/plan set: macro-F1,
accuracy, weighted F1 per model; latency distribution, total extraction wall time,
uncached requests, probe count as cost descriptors; run sizes 10,003/3,080.

## 6. "What were the probes?"

The 16 hand-written Noul probes (`probes.json`): asks_for_information,
requests_an_action, reports_a_problem, unauthorised_activity, transaction_is_pending,
transaction_is_completed, money_outgoing, money_incoming, physical_or_virtual_card,
cash_or_atm, bank_transfer, account_funding_or_top_up, fee_or_charge,
currency_conversion, cancel_or_reverse, account_access_or_security — full questions in
`probes.json` (each is a boolean probability asked once per message; the direct
77-way choice question is separate).

## 7. Round-1 follow-ups (your four-part scope)

- **Probe audit**: no saturation (0.000 at both tails); **compression** (means 0.49–0.63,
  std 0.08–0.16, 15–43% within ±0.05 of 0.5); **common-mode redundancy**
  (`money_outgoing~money_incoming` +0.81); **clear-case extraction errors**
  ("Nothing goes through on my card" → card probe 0.17); currency_conversion works
  (0.86 for exchange intents). → `runs/banking77/analysis/probe_audit.json`, `tools/audit_probes.py`.
- **Scaled linear**: tuning C >> scaling — unscaled CV LR 0.5222, scaled CV LR **0.5286**
  (C*=10); the former "14pp tree gap" shrinks to **4.2pp** vs ExtraTrees.
  → `runs/banking77/analysis/scaled_linear.json`.
- **Choice readout**: LR on 77 log-probs → **0.9008 macro-F1 / 0.9006 acc / top-3 0.9766**
  (+14.3pp over argmax; within 0.9pp of TF–IDF). Probes add nothing (−0.3pp).
  → `runs/banking77/analysis/choice_readout.json`.
- **Cost split**: choice-only 29.9 ms · probes-only 40.5 ms · combined 70.7 ms (additive).
- → `docs/round1-followups.md`.

## 8. Round-2 scope (your four-point list) — DELIVERED

1. **Frozen, exported pipeline**: `readout/banking77-readout-v1/`
   (`readout/banking77-readout-v1/`: `readout_lr.joblib`, `labels.json`, `manifest.json`, `README.md`). Config:
   `log(clip(p,1e-6,1))` → StandardScaler → LR(C=1.0 CV-selected); upstream revision,
   clip constants, intent order, metrics all frozen in the manifest.
2. **Deployment parity**: `readout/test_parity.py` **5/5** — artifact reproduces the
   experiment; shuffled keys OK; tiny/zero probabilities safe & deterministic;
   single vs batched identical; guard rails raise.
3. **Benchmark (text→prediction)**: Jev choice-only+readout single p50 **62.5 ms**
   (p95 68.2), batched **30.2 ms/text / 33.1 texts/s**; TF–IDF+LR single p50 **2.29 ms**,
   batched **0.229 ms/text / 4,374 texts/s**. Readout step itself 1.2 ms single /
   0.6 ms per text batched. → `runs/banking77/analysis/pipeline_benchmark.json`.
4. **Confidence (train-OOF selected, test reported)**: τ=0.335 → 98.9% cov @ 90.0%
   (test 98.8/90.8); **τ=0.645 (default) → 86.7% @ 95.1% (test 88.1/95.4)**;
   τ=0.85 → 72.9% @ 98.1% (test 74.2/98.0). → `runs/banking77/analysis/confidence_coverage.json`.
- → `docs/readout-v1-report.md`.

## 9. Paused / open

- **Probe-bank v2**: paused. Acceptance rule recorded: **Pareto improvement** (higher
  quality at acceptable cost, or comparable quality at lower cost) — not a fixed bar.
- Deferral route for below-τ classifications (human / TF-IDF fallback) not yet wired.
