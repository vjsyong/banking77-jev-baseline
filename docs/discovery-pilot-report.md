# Controlled Discovery Pilot — Feasibility Report

**Question tested:** does frontier feedback change what the teacher discovers?
(3 policies × 3 seeds × 3 rounds; all search inside 1,000 training labels; test split
used only for final scoring.)

**One-line answer:** No — in this controlled setting, the three feedback regimes produced
practically the same search: same action mix (add-only), same budget spent (+4 probes/round,
28-probe banks everywhere), and test-time differences that stay inside seed noise. The
variation that does exist is dominated by wording randomness, not by feedback. The probe
channel itself improved (`B0` 0.4387 → best 0.5245 test macro-F1) but remains far below both
references in the same data regime (TF-IDF 0.7504; Choice readout 0.8468).

---

## 1. Setup (all held constant)

| component | setting |
|---|---|
| Regime | 1,000 stratified train labels; 77 intents; search 100% in-train |
| Extractor | TinyJev-0.6B, batched inference, short-Noul probe format |
| Teacher | gpt-5.6-sol; only the **feedback block** differs per policy |
| Learner panel | LR / LinearSVC / ExtraTrees on probe scores, fixed hyperparams |
| Selection | accept candidate bank iff best-panel 5-fold CV macro-F1 strictly improves (same fixed folds) |
| Budget | 3 rounds, ≤4 proposals/round, evaluated per seed |
| Seeds | teacher temperature {0.3, 0.7, 1.0} |

Feedback per policy: **unguided** = task + current bank only; **accuracy** = + OOF
diagnostics (worst-12 per-class F1, top-12 confusions, per-learner CV scores);
**frontier** = accuracy block + measured extraction cost model + quality/cost archive.

### Steps before the pilot

- **Representation control (done):** one-hot LR on hard argmax labels scores 0.7510
  (vs argmax 0.7574) — supervised re-weighting of hard labels adds nothing. LR on the
  77 log-probabilities scores 0.9008. All of the readout gain is the probability vector.
- **Channel repair (done, 128 audited judgments):** short single-concept questions restore
  discrimination (card AUC 0.777→0.918, pin 0.812→0.972); Choice-format helps separation but
  compresses scores. Repairs adopted: short-Noul probes with "Is/Does" framing.
  Flagship repair: "Nothing goes through on my card" 0.17→0.52; age-limit false-high 0.87→0.49.
- **Small-label references (1k regime, test split):** readout **0.8468** > TF-IDF **0.7504** >
  B0 probes **0.4387**; direct argmax 0.7574.

---

## 2. Pilot results

### 2.1 Actions: all arms behaved identically

| arm | adds | revises | deletes | rounds accepted | final bank |
|---|---|---|---|---|---|
| unguided | 36 | 0 | 0 | 9/9 | 28 probes |
| accuracy | 36 | 0 | 0 | 9/9 | 28 probes |
| frontier | 36 | 0 | 0 | 9/9 | 28 probes |

27/27 rounds accepted — the acceptance rule never bound (every candidate improved CV).
No arm ever pruned, revised, or deleted; the frontier arm's cost framing did not change
behaviour. The teacher always filled the 4-proposal budget with additions.

### 2.2 Probes: same territory, different wording

- Exact-question overlap between arms is near zero (Jaccard: unguided–accuracy 0.00,
  accuracy–frontier 0.00, frontier–unguided 0.04; 83 unique new questions total).
- But the *semantic targets* converge strongly — all three arms independently targeted
  card activation/linking, virtual cards, card acceptance, transfer recipient not received,
  duplicate charges, auto top-up, cash-withdrawal fees, exchange rates.
- Both guided arms' proposals trace directly to B0's worst confusions (e.g. `card_acceptance`
  at F1 0.00 → "does a merchant refuse the card?"; `topping_up_by_card→automatic_top_up`
  confusion → auto-top-up probes; `receiving_money`, `virtual_card_not_working` similarly).
  The unguided arm reached the same areas from the teacher's own domain prior.

### 2.3 Quality: CV said yes, test said "not so fast"

| arm | CV (search metric) per seed | TEST macro-F1 per seed | mean | std |
|---|---|---|---|---|
| unguided | 0.4834 / 0.4691 / 0.4860 | 0.4470 / 0.4167 / 0.4336 | **0.4324** | 0.0152 |
| accuracy | 0.4900 / 0.4799 / 0.4621 | 0.4489 / 0.5245 / 0.4870 | **0.4868** | 0.0378 |
| frontier | 0.4829 / 0.4764 / 0.4642 | 0.5101 / 0.4167 / 0.4354 | **0.4541** | 0.0494 |

- CV-to-test gap: CV rose for every arm (+10–13pp over B0's 0.3631) but most of that was
  fold-fitting; test gains over B0 (0.4387) are ≈0 (unguided), +4.8pp (accuracy), +1.6pp
  (frontier). Seed spread is large (std 1.5–4.9pp) and only accuracy's range clears
  unguided's (by <0.2pp at the boundary) — directional evidence, not significance, at n=3.
- All 9 runs selected LR as best model (mean test: LR 0.4578 > ET 0.4039 > SVM 0.3817).
- Even the best final bank (0.5245) stays far below TF-IDF (0.7504) and the readout (0.8468).

### 2.4 Cost (cumulative discovery + pipeline latency)

- Teacher: 27 calls, ≈1,359 s wall (~151 s/arm/seed average incl. retries).
- Extraction (search): ≈231 s for all round additions on the 1k.
- Final evaluation on test: 99 unique probes ≈ 13 min (~2.5–2.7 ms/text/probe).
- **Pipeline latency of a final bank: 77.0 ms/text** (28 probes) vs B0 46.4 ms/text;
  references: readout ≈30 ms/text batched, TF-IDF 0.23 ms/text. The arms' costs are
  identical because every bank ended the same size.

### 2.5 Concept fidelity (mini-audit, 7 discovered probes × 36 audited messages)

| probe (arm) | acc@0.5 | AUC | sep |
|---|---|---|---|
| transaction_duplicated (unguided) | 1.000 | 1.000 | +0.410 |
| card_payment_rejected (frontier) | 0.722 | 0.988 | +0.295 |
| card_payment_declined (unguided) | 0.722 | 0.978 | +0.224 |
| card_activation_request (accuracy) | 0.722 | 0.926 | +0.288 |
| automatic_balance_topup (frontier) | 0.722 | 0.907 | +0.193 |
| transfer_recipient_not_received (accuracy) | 0.778 | 0.812 | +0.148 |
| cash_withdrawal_fee (unguided) | 0.722 | 0.809 | +0.236 |

Mean AUC ≈ 0.92 — materially better than the original bank's concepts on the same
methodology (0.76–0.97). The two arms' versions of "declined payment" measure the same
thing equally well (0.978 vs 0.988) — wording variance is not the problem; the channel
itself saturates. Labels are agent-verified with intent-pool provenance; sheets in
`fidelity_sheet.csv` for human audit.

---

## 3. What the feedback actually changed

1. **Targeting, yes; outcomes, no.** Guided arms aimed at documented confusions; unguided
   aimed from domain priors — both landed on the same conceptual territory, and both
   produced banks of equal size, cost, and (within noise) quality.
2. **No pruning pressure anywhere.** Even with the cost model + archive, the frontier
   teacher never revised or deleted — cost-awareness in the prompt did not convert into
   cost-saving actions under this acceptance rule.
3. **The bottleneck is the channel, not the search.** With reliable, well-targeted probes,
   adding 12 more moves test macro-F1 only +0–5pp and saturates around 0.45–0.52. The
   measurement medium (single yes/no probabilities → linear learner on 1k) has finite
   information per probe family; better search cannot fix that.
4. **Selection optimism confirmed.** Same-fold acceptance with a strict-improvement rule
   accepted 27/27 candidates; the honest arbiter (test) kept only a fraction of the CV lift.

## 4. Decision (per the pre-registered branches)

- ~~Reliable probes + useful frontier improvements~~ — **no frontier advantage.**
- **→ "Reliable probes, but guided search offers no advantage: the search mechanism needs
  revision."** Probes are reasonably reliable (repair study + fidelity ≈0.92 AUC), but
  feedback granularity changed nothing robust; the current loop (4 adds/round, same-fold
  strict-improvement acceptance) is the part that needs redesign.
- ~~Persistent extraction errors~~ — no longer the blocker (repairs hold).
- Note on the TF–IDF branch: the *probe* pipeline is dominated (≤0.52 vs 0.75/0.85), but
  not all semantic pipelines are — the 77-probability readout beats TF-IDF in this regime
  (0.8468 vs 0.7504) and remains the best use of Jev.

### Concrete revisions to test next

1. **Stiffer selection:** acceptance on a held-out lane or bootstrap-uncertainty test
   (not the same folds the proposal was scored on); allow only multi-round-stable gains.
2. **Convert cost-awareness into mechanics:** budget caps (e.g. bank ≤20 probes), acceptance
   penalties for size, or explicit prune passes — prompt-level cost framing demonstrably
   does nothing.
3. **Stop feeding the channel it can't use:** shift effort from probe *count* to probe
   *family* diversity or to extractor capability (larger Jev model), given per-probe
   information saturates.
4. Keep the readout frame of reference in every future round: it remains the strongest
   Jev-based pipeline (and wins outright at 1k labels — a genuinely useful small-data
   result on its own).

## 5. Limitations

- n=3 seeds/arm; 1k regime; 3 rounds; single teacher model; one extractor; one probe format
  (short-Noul). Differences are directional evidence, not significance tests.
- Acceptance evaluated on the same folds used for selection (optimistic CV); test split is
  the arbiter and was untouched during search.
- B0 comparison uses a fixed (C=1) LR protocol; Stage-C references use CV-tuned C.

## 6. Artifacts

- `runs/banking77/discovery/control_onehot.json` — representation control.
- `runs/banking77/discovery/repair_set.csv`, `repair_scores.json`, `repair_audit_sheet.csv`
  — channel-repair study + audit sheet.
- `runs/banking77/discovery/smallregime_refs.json` — 1k reference table.
- `runs/banking77/discovery/pilot/` — `summary_full.json`, `archive_full.jsonl` (27 rounds),
  `teacher_full_*.txt` (all raw teacher responses), `probe_scores.sqlite` (99-probe store),
  `test_eval.json`, `fidelity_report.json`, `fidelity_sheet.csv`.
- Code: `discovery/` (teacher client, control, repair, refs) and `discovery/pilot/`
  (extractor, loop, evaluator, fidelity).
