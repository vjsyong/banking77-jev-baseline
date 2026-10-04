# Pruning Study Report — BANKING77 mixed-probe banks

**Scope (pre-registered):** isolate whether mixed semantic probe banks can be
compressed while preserving predictive value, with teacher discovery frozen.
This is an **exploratory** continuation (the BANKING77 test split has informed
several earlier redesigns). Bounded question, bounded budget — no expansion.

**Go/no-go: NO-GO for compression-preserving deployment at the pre-registered
standard** (≤1pp test-macro-F1 loss at ≥25% latency reduction). The corrected
screen is a large improvement over the original one (damage cut ~2×) but no
frozen operating point meets both criteria; per the plan's rule table the
recommended action is *pause compression and preserve the unpruned
representation*, and the frontier-feedback study (§6) is therefore **not
triggered**. Details, evidence, and what would change the answer below.

---

## 1. Frozen inputs and provenance

- **Banks:** the three frozen mixed-strict banks (all seeds, no selection):
  `mixed-strict-s0.3` (26 probes / 42 cols), `s0.7` (28/48), `s1.0` (28/52).
- **Immutable feature snapshots** (`discovery/pruning_study/snapshots/`): run-scoped
  definition-checked re-extraction on the 1k split; every snapshot records exact
  question definitions, Choice criteria with option order, extractor revision
  (tinyjev 0.1.3 / TinyJev-0.6B c559c2f7…), inference configuration (no co-query:
  one question per forward; chunk 64), dataset-row sha256 list, and the exact
  numeric column order. Snapshot CVs reproduce the wording-verified factorial CVs
  **exactly** (0.5952 / 0.6844 / 0.6696). npz sha256:
  s0.3 `c9c0855e…`, s0.7 `b70d2147…`, s1.0 `ff28bbb9…`.
- **Subset ≡ column-taking (verified):** fresh subset extraction vs slicing the
  complete bank's columns gives **max |Δ| = 0.0** (per-question forwards have no
  cross-question coupling). Cached splicing is therefore exact; no per-subset
  re-extraction needed (`discovery/pruning_study/snapshots/equivalence_report.json`).
- **Fixed throughout:** 1,000 labeled rows; 5-fold stratified folds (seed 77);
  frozen deployed learner = StandardScaler → LogisticRegression(C=1.0), selected
  best-of-panel on the complete unpruned bank using training data (all seeds: LR);
  budget 150 candidate-bank evaluations per seed across all conditions; corrected
  tolerances 0.005 / 0.010; fixed (non-rolling) reference; whole-question removals
  (all Choice columns); deterministic candidate order (bank order); every
  evaluated removal — accepted or rejected — logged in `paths/`.

## 2. Cache-drift attribution (§1 of the plan)

| Cell(s) | Mechanism found | Layers affected | Treatment | Current validity |
|---|---|---|---|---|
| noul-strict ×3, noul-pareto ×3 | Pre-fix cache served rows by id, not definition; same slug + slightly different wording across runs shared one extraction (19/99 pilot ids; mean per-message |Δ| 0.071, worst 0.50, min r 0.62) | reporting only: re-verified CV Δ ≤ 0.6pp; acceptance margins unaffected in outcome | correct the report (done, pilot report limitations) | **valid** (± wording noise) |
| mixed-strict ×3 | none detected (re-verified Δ = 0.0000) | — | none | **valid** |
| mixed-pareto ×3 | (a) prune pass used probe-index as column-index; with multi-column Choice probes the matrix↔bank mapping desynced mid-sweep; (b) recorded CVs consequently inflated | prune decisions + recorded scores + round-2/3 teacher diagnostics (computed on desynced matrices) | decisions/feedback affected → trajectory rerun required; applied: final scores corrected via run-scoped re-verification (test unchanged, CV −6.1…−11.5pp); rule replayed on clean features (§4b); full teacher-trajectory rerun **not** performed (its mechanism question is answered by the replay; all mixed-pareto search-level claims stand marked **exploratory**) | final scores **usable**; search-level comparisons **excluded from mechanism claims** |

Would decisions have changed under corrected features? For mixed-pareto, yes —
the clean replay of the same rule removes 12–16 probes at −5.3…−7.6pp CV (vs
17–19 removals recorded); both scores and behavior differ. Unaffected cells were
not rerun (§1 instruction).

## 3. Pruning conditions and results (§3 of the plan)

Learner frozen; preprocessing refit within each training fold; candidate compared
to the fixed unpruned reference for corrected conditions; (b) replays the
original rule semantics (rolling single-fold screen, tolerance 0.01, floor 8,
whole-question removals with *correct* column mapping) on clean features.

| bank | ref CV5 | (b) original, clean replay | (c) corrected, δ=0.005 | (d) corrected, δ=0.010 |
|---|---|---|---|---|
| s0.3 | 0.5952 | 11p/23c · 0.5194 (**−7.58**) · 70 evals | 17p/33c · 0.5946 (**−0.06**) · 43 evals | 17p/33c · 0.5861 (**−0.91**) · 43 evals |
| s0.7 | 0.6844 | 12p/30c · 0.6319 (**−5.25**) · 55 evals | 16p/34c · 0.6797 (**−0.47**) · 98 evals | 21p/41c · 0.6745 (**−0.98**) · 24 evals · budget-truncated |
| s1.0 | 0.6696 | 12p/32c · 0.6141 (**−5.55**) · 73 evals | 21p/45c · 0.6647 (**−0.49**) · 49 evals | 22p/46c · 0.6609 (**−0.87**) · 50 evals |

Budget used: 139/150, 150/150 (s0.7 hit the cap; (d) truncated mid-path — reported
as-is, no budget expansion), 139/150. Identical evaluations were reused across the
corrected conditions via a shared cache; **all rejected removals are in the
logged paths**.

**Column-level arm (reported separately, as required):** single-option column
drops are not the same as question-level compression (extraction work is
per-question, so dropping a column saves classifier input only). Sampled drops
mostly hurt CV (−0.66…−1.12pp), one neutral (+0.04pp) — no column-level wins
worth reporting; s0.7's arm was budget-skipped.

**Reading:** the original screen's damage was mostly *screen behavior* (a rolling
single-fold allowance accumulates — −5.3…−7.6pp even on clean features), not only
the drift. The corrected screens reach ≈CV-lossless compression (−0.1…−0.5pp)
on the search folds — but see §5: the search folds are an optimistic lane.

## 4. Frozen deployment points (§4 of the plan)

Selected on training data and frozen **before** any outside evaluation:
`discovery/pruning_study/frozen_points.json`, sha256 `d7650493…` (recorded in the evaluation provenance).
No further selection used the test split.

**Test split (3,080) macro-F1, frozen learner fit on the 1k:**

| bank | unpruned | (c) δ=0.005 | (d) δ=0.010 |
|---|---|---|---|
| s0.3 | 0.6495 | 0.6092 (**−4.03**, CI [−5.28, −2.84]) | 0.6083 (**−4.12**, CI [−5.39, −2.96]) |
| s0.7 | 0.7264 | 0.6949 (**−3.15**, CI [−4.38, −1.96]) | 0.7108 (**−1.56**, CI [−2.52, −0.74]) |
| s1.0 | 0.7109 | 0.6936 (**−1.73**, CI [−2.58, −0.90]) | 0.7014 (**−0.95**, CI [−1.75, −0.21]) |
| **paired mean Δ** | — | **−2.97pp** | **−2.21pp** |

All paired differences significant (exact McNemar p ≤ 0.038; bootstrap 95% CIs
over messages as above). Seed variation is reported separately above; folds and
successive pruning steps are not treated as replicates.

**Latency (same hardware/warmup/sample; includes classifier; batched 1,024 texts,
chunk 64; single-request 50 texts; per-question deployed pipeline shape):**

| bank | metric | unpruned | (c) | (d) |
|---|---|---|---|---|
| s0.3 | batched ms/text (Δ) | 63.68 | 41.64 (**−34.6%**) | 42.03 (**−34.0%**) |
| s0.3 | single p50/p95 ms | 1535/1572 | 1018/1040 | 1064/1095 |
| s0.7 | batched ms/text (Δ) | 64.55 | 37.08 (**−42.6%**) | 47.39 (**−26.6%**) |
| s0.7 | single p50/p95 ms | 1744/1769 | 966/1001 | 1170/1199 |
| s1.0 | batched ms/text (Δ) | 66.36 | 49.41 (**−25.5%**) | 52.13 (**−21.4%**) |
| s1.0 | single p50/p95 ms | 1621/1790 | 1223/1258 | 1246/1284 |

Also available per config in `discovery/pruning_study/deployment_eval.json`: probe count, numeric feature
count, total Choice options, throughput (15.1–27.0 texts/s). Single-request
latency is launch-overhead-bound (per-question forwards); batched numbers are the
deployment-relevant ones.

**Target check (pre-registered: ≥25% latency reduction AND ≤1pp quality loss):**

- (c) δ=0.005: latency **3/3 pass** (25.5–42.6%); quality **0/3 pass** (−1.7…−4.0pp).
- (d) δ=0.010: latency **2/3 pass** (s1.0 fails at −21.4%); quality **1/3 pass** (s1.0 at −0.95pp; others −1.6/−4.1pp).
- No operating point meets both criteria. The search-lane acceptance underestimated
  held-out loss by ~1.3–4× — the third independent instance of selection-lane
  optimism in this project (pilot acceptance, factorial mixed-pareto, now pruning).

## 5. The two success questions, separated (§4/§7 of the plan)

1. **Compression feasibility:** *partially.* Structure reduction of 30–43% of
   probes is achievable at 1.7–4.1pp test cost depending on seed/point; the
   pre-registered "quality-preserving" bar (≤1pp) is not met robustly (one seed
   only, at the edge of its CI). The corrected screen is nonetheless ~2× less
   damaging than the original rule (which was never quality-preserving either).
2. **Practical competitiveness:** *not achieved.* At the frozen points the banks
   score 0.61–0.71 test vs TF–IDF 0.7504 and the Choice readout 0.8468 under the
   same 1k-label allowance, at 37–52 ms/text batched vs ~30 ms (readout) and
   0.23 ms (TF–IDF). The probe route's remaining value is interpretability
   (named, auditable questions), not quality or speed.

## 6. Decision (§5 of the plan)

| Plan rule | Applies? |
|---|---|
| Quality-preserving compression succeeds → return to frontier comparison | **No** (quality bar missed; §6 not triggered) |
| Conservative pruning succeeds but saves little latency | No (latency savings are substantial: 25–43%) |
| Clean pruning still causes large quality losses → pause compression, preserve unpruned | **Closest fit — action taken** (losses are real and fail the pre-registered bound, though far smaller than the original rule's) |
| Drift materially changed previous trajectories → correct those first | Handled in §2 (mixed-pareto marked exploratory; mechanism claims now rest on this study's clean evidence) |

**Explicit decision:** pause compression; preserve the unpruned mixed
representation as the deliverable artifact. Do not expand the pruning budget.
If compression is ever revisited, the acceptance lane must be strengthened first
(independent selection folds / held-out split of the search data) — repeating the
same-lane acceptance would repeat the same optimism.

## 7. §6/§7 status

- §6 (teacher-feedback rerun with shared backend): **not triggered** (conditional
  on compression success).
- §7 (readout track): unchanged; the 0.8468 readout at 1k labels remains the
  strongest reference; any hybrid must show incremental benefit over it and
  justify the extra inference cost.

## 8. Artifacts

- `snapshots/` — immutable inputs (+`discovery/pruning_study/snapshots/equivalence_report.json`, row hashes).
- `paths/{bank}_{condition}.jsonl` — full removal paths incl. rejected candidates.
- `discovery/pruning_study/pruning_results.json` — conditions, budgets, column arms.
- `discovery/pruning_study/frozen_points.json` — the two deployment points per seed (sha256 in provenance).
- `discovery/pruning_study/deployment_eval.json` — test evaluation, paired statistics, latency, targets.
- Code: `discovery/pruning_study/{prepare_snapshots,run_pruning,evaluate_and_benchmark}.py`.
