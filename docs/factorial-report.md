# Representation × Selection Factorial — Report

**Question (from review):** isolate the explanation for the pilot's frontier-feedback
failure — does the limitation come from the **representation**, the **selection
mechanism**, or their **interaction**? (2×2 factorial; teacher briefing and diagnostics
held constant.)

**Headline:** the limitation was substantially in the **representation**. Categorical
Choice probes (option distributions retained as features) add **+16.6pp** paired test
macro-F1 inside the strict cells (all seeds positive) and +7.8pp inside the pruned cells.
The **selection mechanism as implemented produces no quality gains**: pruning compresses
banks by ~65% and roughly halves cost, but at a steep quality cost (−10.6pp noul,
−19.5pp mixed) — a compensation tool at best, and worse for the better representation.

---

## 1. Setup (constants held, as directed)

| component | setting |
|---|---|
| Cells | {noul-only, mixed + categorical Choice} × {strict-improvement, Pareto archive + pruning} |
| Regime | 1,000 stratified train labels; search 100% in-train; test (3,080) untouched during search |
| Teacher | **gpt-6.1-sol** (lmuai-pro) for **all four conditions**; accuracy diagnostics briefing constant |
| Instruments | TinyJev-0.6B batched; short-Noul + Choice (2–3 options; full distributions) |
| Learner panel / folds | LR / LinearSVC / ExtraTrees; fixed 5-fold stratified (seed 77) |
| Budget | 3 rounds × ≤4 proposals/round × 3 seeds |
| Selection lane | proposal feedback and candidate selection separate; selection = development data; final evaluation independent (test) |

Prune pass (pareto cells only): single sequential sweep with a single-fold LR screen;
removal kept if the screen does not drop below pre-removal screen − 0.01.

## 2. Results (test split macro-F1; wording-verified)

| cell | CV per seed (search) | TEST per seed | mean | bank size | choices | µms/text |
|---|---|---|---|---|---|---|
| noul-strict | 0.462 / 0.475 / 0.479 | 0.533 / 0.505 / 0.550 | **0.5295** | 28 / 28 / 28 | 0 | 31.7 |
| noul-pareto | 0.376 / 0.418 / 0.361 | 0.379 / 0.467 / 0.424 | **0.4233** | 15 / 22 / 15 | 0 | 14.6 |
| mixed-strict | 0.595 / 0.684 / 0.670 | 0.650 / 0.726 / 0.711 | **0.6956** | 26 / 28 / 28 | 8 / 10 / 12 | 40.2 |
| mixed-pareto | 0.435 / 0.475 / 0.403 | 0.491 / 0.557 / 0.455 | **0.5009** | 11 / 8 / 10 | 5 / 5 / 4 | 15.7 |

**Paired per-seed effects (pp):**

- Representation (mixed − noul), strict: **+11.6 / +22.1 / +16.1** → mean **+16.6**
- Representation (mixed − noul), pareto: **+11.2 / +9.1 / +3.0** → mean **+7.8**
- Selection (pareto − strict), noul: **−15.4 / −3.9 / −12.6** → mean **−10.6**
- Selection (pareto − strict), mixed: **−15.9 / −16.9 / −25.6** → mean **−19.5**

**Compression facts:** 85 probes pruned in total across the pareto cells; mixed-pareto
banks end at 8–11 probes (vs 26–28 in mixed-strict) at ~15.7 vs ~40.2 ms/text. The
compression is real; its quality cost is not acceptable at this screening budget.

**Reference frame (same 1k-label allowance):** TF–IDF 0.7504 · Choice readout 0.8468 ·
direct argmax 0.7574 · B0 probes 0.4387. Mixed-strict at 0.6956 closes most of the
remaining gap to TF–IDF (+25.7pp over B0); the readout remains ahead.

**Best single points (cost-shaped):** mixed-strict s0.7 = 0.7264 (28 probes, ~40 ms/text);
mixed-pareto s0.7 = 0.5574 at 8 probes / ~12 ms/text.

## 3. Integrity & verification (why these numbers are trustworthy)

- **Wording-scoped verification:** any probe id with different wording/options across
  runs was re-extracted per wording (`ev::…` slots) and per run where needed (`rv::…`
  run-scoped slots) — every run scored with exactly the definitions its bank used.
- **Definition-aware cache fix:** the probe store now purges and re-extracts whenever a
  stored definition changes (root-cause fix for the crashes and silent stale reuse found
  mid-run; see pilot report limitations for the pre-fix exposure quantification).
- **Recorded CVs vs re-verified CVs (Δ):** ≤0.6pp for 9/12 runs; mixed-pareto's recorded
  CVs were inflated by store drift during search (−6.1pp to −11.5pp; corrected values
  used here; the test numbers were unchanged by the correction).
- Selection lane remained development data only; the test split was consulted for final
  evaluation only.

## 4. Attribution (per the review's rules)

1. **Mixed representation gains → measurement design.** Positive in all six paired
   comparisons (both selections × three seeds; +3.0…+22.1pp). The categorical Choice
   probes — which retain the full option distribution instead of a single scalar — are
   the lever that closes most of the probe-channel gap to TF–IDF at 1k labels.
2. **Pruning produced no quality gains → the selection mechanism, as implemented, is not
   a source of improvement.** It is an effective compressor with a real quality price;
   its screening (single-fold LR, tolerance 0.01) is too noisy to distinguish dead weight
   from thin signal, and it over-prunes faster than quality degrades gracefully.
3. **Interaction confirmed:** pruning hurts the better representation more (−19.5 vs
   −10.6pp), i.e. dense-information banks lose more per blunt removal.
4. **Frontier-feedback benefit to the teacher: still not established** — by design the
   factorial holds feedback constant; what it does establish is where the failure lived:
   representation first, selection second.

## 5. Implications for the next move (options, not conclusions)

- Keep **Choice-format probe families** as the default instrument; next test whether the
  remaining gap to the readout closes with scale (labels) or with readout-style hybrids.
- If compression is wanted, re-work the prune screen: multi-fold/panel screening with
  quality-aware tolerance, or Pareto acceptance that requires the *test-independent*
  score to hold within noise, not 0.01 off a single fold.
- Cost anchor for decisions: probe-bank pipelines run 15–40 ms/text vs readout ~30 and
  TF–IDF 0.23 — the probe route remains the expensive one; its value here is
  interpretability (named, auditable questions), not speed.

## 6. Limitations

- n=3 seeds/cell; single teacher; 1k regime; 3 rounds; one extractor (0.6B); prune
  screen single-fold LR. Differences are directional; paired differences reported.
- The resumed noul runs were selected under pre-fix cache semantics (exposure quantified;
  wording-scoped re-verification Δ ≤0.6pp on their final numbers).
- Same-folds acceptance during search (development lane) — selection optimism bounded by
  the independent test evaluation reported here.

## 7. Artifacts

- `runs/banking77/discovery/factorial/summary_full.json` — all 12 runs, final banks, traces.
- `runs/banking77/discovery/factorial/test_eval_factorial.json` — wording-scoped evaluation,
  paired effects, cell means, per-run CV re-verification.
- `runs/banking77/discovery/factorial/mixed_pareto_reverify.json` — run-scoped re-verification.
- `runs/banking77/discovery/factorial/archive_full.jsonl`, `teacher_full_*.txt` — raw rounds.
- Code: `discovery/pilot/{run_factorial,evaluate_factorial,reverify_mixed_pareto,reconstruct_runs}.py`,
  `discovery/pilot/extract_probes.py` (definition-aware store).
