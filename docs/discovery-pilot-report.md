# Controlled Discovery Pilot — Feasibility Report

**Question tested:** does frontier feedback change what the teacher discovers?
(3 policies × 3 seeds × 3 rounds; all search inside 1,000 training labels; test split
used only for final scoring.)

**One-line answer:** within the tested budget, frontier guidance produced no cost reduction
and no demonstrated advantage over accuracy guidance. Accuracy guidance averaged +5.4pp over
unguided on test (paired per seed: +0.2 / +10.8 / +5.3) — that could matter and is not
established at three seeds; the three-seed design cannot establish the *absence* of an
effect either. The probe banks improved over B0 (0.4387 → best 0.5245 test macro-F1) but
plateau at 0.45–0.52 for the tested banks; that plateau does not establish a capacity
ceiling for semantic extraction. Substantial positive result alongside: the Choice readout
scores 0.8468 vs TF–IDF 0.7504 at the same 1,000-label allowance.

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
No arm ever pruned, revised, or deleted, and the teacher never proposed compression in any
arm — so the gate's inability to accept some kinds of candidate cannot explain these
trajectories (see §3.2 for the gate's actual limitation). The teacher always filled the
4-proposal budget with additions.

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

- **Paired per-seed test differences (pp):** accuracy−unguided **+0.19 / +10.78 / +5.34**
  (mean +5.44, positive in all three seeds); frontier−unguided **+6.31 / 0.00 / +0.18**
  (mean +2.16, carried by one seed); frontier−accuracy **+6.12 / −10.78 / −5.16**
  (mean −3.27, mixed signs). Three seeds cannot establish the absence of an effect;
  read as: accuracy guidance may matter, frontier guidance shows no advantage over
  accuracy and no cost reduction.
- CV-to-test gap: CV rose for every arm (+10–13pp over B0's 0.3631) but most of that was
  fold-fitting; test gains over B0 (0.4387) are ≈0 (unguided), +4.8pp (accuracy),
  +1.6pp (frontier).
- All 9 runs selected LR as best model (mean test: LR 0.4578 > ET 0.4039 > SVM 0.3817).
- Even the best final bank (0.5245) stays below TF-IDF (0.7504) and the readout (0.8468).

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
thing equally well (0.978 vs 0.988) — wording variance is not the problem here. But high
concept AUC alongside weak intent classification indicates that faithful per-concept
measurements can still omit the distinctions the target requires. Labels are agent-verified
with intent-pool provenance; sheets in `fidelity_sheet.csv` for human audit.

---

## 3. What the feedback actually changed

1. **Targeting, yes; decisive outcome differences, not shown.** Guided arms aimed at
   documented confusions; unguided aimed from domain priors — both landed on the same
   conceptual territory and equal bank sizes/costs. Held-out quality differences between
   feedback regimes are suggestive for accuracy guidance (+5.4pp paired vs unguided) but
   not established at this replication level; frontier guidance shows neither an advantage
   over accuracy nor any cost reduction.
2. **No compression anywhere — teacher-side or gate-side.** The teacher never proposed
   compression despite the frontier arm's cost framing; and the strict-improvement gate,
   while unable to accept equal-quality-cheaper banks (a real limitation), was never given
   anything of that kind to accept. The gate's limitation does not explain these
   trajectories.
3. **The plateau belongs to the tested banks — it is not a capacity ceiling.** Three rounds
   of short binary probes reaching ≈0.45–0.52 does not establish the limit of semantic
   extraction. The fidelity/classification mismatch (§2.5) says the open question is
   measurement design: usable distinctions may be absent from the tested probe families
   rather than from the channel as such.
4. **Selection optimism confirmed.** Same-fold acceptance with a strict-improvement rule
   accepted 27/27 candidates; the independent arbiter (test) kept only a fraction of the
   CV lift.

## 4. Decision (per the pre-registered branches)

**Retained result: reliable, domain-informed probes improved development scores, but
frontier prompting did not induce compression or establish superior held-out trade-offs
within the tested budget.**

- ~~Reliable probes + useful frontier improvements~~ — no frontier advantage in this
  configuration.
- ~~Persistent extraction errors~~ — no longer the blocker (repairs hold; fidelity ≈0.92 AUC).
- TF–IDF branch, stated precisely: the *probe-bank* pipeline is dominated by TF–IDF at
  this scale (≤0.52 vs 0.75), but not all semantic pipelines are — the Choice readout beats
  TF–IDF under the same label allowance (0.8468 vs 0.7504). This supports the usefulness of
  the Choice-distribution representation for small-label settings; discovered-probe label
  efficiency remains unsupported.

### Next experiment (from review): 2×2 factorial

Isolate the explanation rather than running more proposal rounds. Factors:

| factor | conditions |
|---|---|
| Representation | Noul-only / mixed Noul + categorical Choice (distributions retained as features) |
| Selection | strict accuracy improvement / nondominated archive + explicit pruning |

- Teacher briefing and diagnostics held **constant** across all four conditions; the teacher
  is upgraded to **gpt-6.1-sol** (via the lmuai-pro provider) and used for every condition.
- Proposal feedback and candidate selection stay separate: the selection lane is development
  data; the independent final evaluation (test split) remains untouched by the search.
- Readout and TF–IDF references are trained with the same label allowance in every comparison.
- Attribution rules: pruning gains → selection mechanism; mixed-representation gains →
  measurement design; neither → no demonstrated benefit from frontier mechanics to the
  teacher, and the probe-search line pauses with this extractor.

### Concrete revisions being carried into the factorial (qualified)

1. **Separate selection lane:** proposal feedback and candidate selection stay distinct; the
   selection lane is development data only; independent final evaluation retained.
2. **Cost-awareness as mechanics:** nondominated archive + explicit pruning (not prompt
   framing); same teacher backend across all conditions.
3. **Channel capability / family diversity:** permit compact categorical Choice probes and
   retain their distributions; measurement format is now a credible experimental factor.
4. **Readout reference frame:** Choice readout and TF–IDF trained with the same label
   allowance in every comparison.

## 5. Limitations

- n=3 seeds/arm; 1k regime; 3 rounds; single teacher model (gpt-5.6-sol); one extractor;
  one probe format (short-Noul). Differences are directional evidence; the design cannot
  establish the absence of an effect — paired differences are reported for that reason.
- Acceptance evaluated on the same folds used for selection (optimistic for search
  dynamics); the test split is the arbiter and was untouched during search. The gate can
  only take strict CV improvements — it cannot accept equal-quality, cheaper banks (to be
  addressed by the selection factor above).
- B0 comparison uses a fixed (C=1) LR protocol; Stage-C references use CV-tuned C.
- **Probe-cache wording collisions (found after the run, quantified).** 19 of the 99 probe
  ids appear with slightly different wordings across runs while sharing one cache slot
  (pre-fix cache semantics: rows were served by id, not by definition). A re-extraction
  sensitivity study (29 variant pairs, 1k split) measures the substitution effect:
  mean per-message |Δ| ≈ **0.071** (median 0.065; p95 of pair-means 0.140), worst
  single-message Δ 0.50, mean r ≈ 0.81 (min 0.62). All 9 runs' final banks are exposed
  to at least one such id, so per-run numbers carry a near-paraphrase substitution on a
  subset of features; arm-level comparisons are the headline and are affected only
  through that channel. The factorial experiment (next section) runs on the fixed,
  definition-aware cache and re-verifies values per wording at evaluation time.
  Artifact: `runs/banking77/discovery/pilot/collision_sensitivity.json`.

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
