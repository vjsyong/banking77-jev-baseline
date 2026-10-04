# Experiment brief: Example-guided semantic feature discovery under serving budgets

**Version:** 1.0  
**Date:** 2026-10-04  
**Status:** Design brief. Complete the freeze checklist before starting independent confirmation.  
**Development task:** BANKING77  
**Confirmation task:** CLINC150, in-scope classification only

## 1. Purpose

Adapt the example-driven feature discovery mechanism in *LLMs as Feature Engineers for Text-and-Tabular Prediction* [1], then test whether giving the teacher measured cost and frontier information improves its proposals under a fixed serving budget.

**Primary research question:** Does frontier-aware, example-guided semantic feature discovery improve test macro-F1 at a fixed end-to-end serving latency?

**Secondary questions:** Does selecting concrete error examples help beyond simply showing labelled examples? Do discovered features complement strong text representations? Does useful quality emerge with less cumulative discovery compute?

This is a new, prospectively specified extension. It does not reopen the completed pruning experiment or change that experiment's NO-GO decision.

## 2. Relationship to the earlier work

The reference paper uses concrete model failures, categorical definitions, a historical candidate pool, and numerical forward selection. It reports complementarity with lexical and embedding representations, and discusses serving overhead [1].

The previous BANKING77 pilot supplied aggregate confusions without raw messages. Its teacher-policy comparison used Noul features; the later, stronger mixed-format experiment held teacher feedback constant. Those studies motivate this experiment but do not test the same example-level feedback mechanism.

This extension changes several components: TinyJev extraction, probability-vector features, multiclass intent classification, small label budgets, explicit latency constraints, and independent test evaluation. Describe it as an adaptation and extension, not an exact replication.

Do not claim to reproduce the paper's numerical results. A direct replication would require a separate protocol matching its tasks, models, candidate budgets, selection, and feedback templates.

## 3. Hypotheses and registered contrasts

| ID | Hypothesis | Contrast | Role |
| --- | --- | --- | --- |
| H1 | Frontier information improves quality at a fixed serving budget | F minus E at the primary latency limit | Primary |
| H2 | Error-selected examples improve discovery beyond access to examples | E minus R | Secondary |
| H3 | Discovered semantic features add predictive information to a strong text baseline | TF-IDF plus bank minus TF-IDF alone | Secondary |
| H4 | Frontier feedback improves discovery efficiency | F versus E quality at matched cumulative resource checkpoints | Secondary |

Unguided U is a historical reference. E minus U combines access to examples, diagnostic information, and error targeting; it does not isolate error targeting.

Representation quality, compression, and concept fidelity are measured separately. A good Choice representation alone does not establish H1 or H2.

## 4. Experimental arms

| Arm | Feedback to teacher | Purpose |
| --- | --- | --- |
| U: Unguided | Task, current bank, definition history; no messages or performance diagnostics | Prior-driven discovery reference |
| R: Random examples | Labelled random contrastive pairs, predictions, current score, and the same aggregate diagnostic format as E | Controls for example access and diagnostic information |
| E: Error-guided | Targeted error pairs, correctly classified control pairs, predictions, current score, aggregate diagnostics | Tests example-level error targeting |
| F: Frontier-guided | E-format feedback plus measured cost descriptors and quality-cost archive | Tests frontier information supplied to the teacher |

R and E use the same number of examples, metadata fields, text-length limits, and scalar/aggregate diagnostics. Their difference is how example pairs are selected. This tightens the earlier proposed random-example control.

E and F use the same feedback construction procedure. Their literal examples can differ after their banks diverge because their models make different errors.

### Constants across arms

- Same teacher model revision and generation settings.
- Same TinyJev model revision, extraction runtime, prompt construction, option schema, and input truncation.
- Same initial semantic bank. Default: empty bank, with TF-IDF as the common starting representation.
- Same downstream learner, tuning procedure, numerical selector, resource ceilings, and latency budgets.
- Same discovery and confirmation rows, CV folds, and benchmark inputs within each seed.
- Same proposal slots and token limits. Record actual token expenditure.
- Same definition memory and current-bank presentation. No arm sees another arm's proposals or results.
- One predefined current bank generates feedback: the bank selected for the primary latency limit.

F receives cost information; all arms' numerical backends enforce cost constraints. That separates teacher feedback from selection mechanics.

## 5. Data and information boundaries

### Development: BANKING77

Use BANKING77 to debug implementation, calibrate resources, and freeze the protocol. Its earlier test results have influenced research choices, so additional BANKING77 results are developmental or exploratory.

Do not reuse its official test set as a fresh confirmation set.

### Confirmation: CLINC150

Use `data_full.json` from the authors' repository [2]. Restrict the task to its 150 in-scope intents. Exclude out-of-scope examples from this experiment.

For each of five preregistered seeds, sample 20 examples per intent from the official training partition:

| Partition | Per intent | Total | Permitted use |
| --- | ---: | ---: | --- |
| Discovery | 15 | 2,250 | Five-fold CV, proposals, extraction, adaptive selection, teacher feedback |
| Confirmation | 5 | 750 | One evaluation of already selected final banks; no search feedback |
| Official test | 30 | 4,500 | Registered final evaluation after banks and procedures are frozen |

The task-label allowance is 3,000 per seed. The official validation partition is unused. Do not silently add its labels to training or tuning.

Use the same sampled rows and folds in all arms of a seed. Different seeds vary the sampled labelled data and teacher randomness. Record the teacher seed when supported; do not imply API generation is perfectly reproducible when it is not.

The teacher may inspect only discovery messages and labels. Confirmation and test texts are not proposal inputs. Cost calibration uses a fixed subset of discovery texts.

Count task labels shown to the teacher within the 3,000-label allowance. Record human concept annotations as a separate annotation cost; they must not be fed back into confirmation-time search.

### Final refit

Freeze one selected bank for each latency limit before reading confirmation outcomes. Evaluate those banks on confirmation once, without revising or selecting between alternatives based on that result. Then refit the same frozen pipelines on all 3,000 permitted labels and evaluate on official test.

A disappointing confirmation score is reported. It is not permission to restart search, change hyperparameters, or substitute a different bank.

## 6. Representation and extraction

Use categorical Choice probes as the default instrument. Each definition contains:

- Stable name and a single concept.
- A short question answerable from the message alone.
- Three to six options, each with an explicit definition.
- An `unclear/not stated` option where the text cannot support a definite category.
- No use of the true task label during extraction.

Prefer concepts that can distinguish multiple intents. Exact target-label option lists belong to the direct-Choice reference, not the semantic bank. Log near-label proxies for later content analysis.

Example structure:

```json
{
  "name": "transaction_state",
  "question": "What state of the transaction does the message describe?",
  "options": [
    {"id": "pending", "definition": "Awaiting completion or settlement."},
    {"id": "declined", "definition": "Explicitly refused or rejected."},
    {"id": "completed", "definition": "Explicitly completed or posted."},
    {"id": "reversed", "definition": "Explicitly reversed or returned."},
    {"id": "unclear", "definition": "No definite state is stated."}
  ]
}
```

The example illustrates schema, not a required CLINC150 feature.

Retain the full option distribution. Use `log(clip(p, 1e-6, 1.0))` with numeric scaling fitted only on the learner's training partition. Freeze this transformation during development. Missing options, nonfinite values, invalid widths, and invalid all-zero distributions raise extraction errors; do not turn them into apparently valid features.

Select and remove whole question blocks, including all option columns. Numeric width is not a question count.

### Development instrument audit

Audit approximately 200 message-question pairs over six to eight predefined development concepts, including ambiguous cases. Collect human categorical judgments and report per-concept confusion matrices, macro-F1, agreement, and probability discrimination where applicable.

Use this audit to settle prompt and option conventions before confirmation. If the channel remains unreliable, pause before the expensive search. Freeze a minimum acceptable audit criterion in the run configuration; do not decide it after seeing confirmation results.

### Cache identity

Key extraction by text hash, canonical complete definition hash, exact option order, upstream model revision, extraction prompt revision, truncation settings, and decoding/runtime settings that affect outputs.

Definitions are immutable. A revision creates a new definition hash and feature block. A name alone is never cache identity. Persist an explicit question-to-column map.

## 7. Downstream learner and baselines

### Main discovery pipeline

`TF-IDF(text) + scaled Choice log-probability blocks -> LogisticRegression`

Use a word-plus-character TF-IDF implementation and a sparse-compatible joint feature pipeline. Semantic scaling is applied to the semantic block; preserve sparse lexical features.

Freeze vectorizer settings and the LR regularization-selection procedure before confirmation. A reasonable default is a small, predefined C grid selected once on the initial discovery baseline, then fixed throughout adaptive search. Apply the same procedure in every arm. Record the resulting C per seed.

Fit vocabulary, IDF, scaling, and classifier within each discovery fold. Do not build preprocessing on all discovery rows before producing OOF predictions.

Do not select the best of LR/SVC/ExtraTrees on every candidate. A changing learner panel would add another source of adaptive selection. Alternative learners are optional post-freeze robustness analyses.

### References under the same label allowance

| Reference | Purpose |
| --- | --- |
| TF-IDF + LR | Main unaugmented baseline |
| Frozen sentence embeddings + LR | Ordinary dense representation control |
| Direct task Choice | Zero-shot reference |
| Supervised task Choice readout | Existing strong Jev approach, retrained for the task and label budget |
| Discovered semantic bank + LR | Semantic-only diagnostic |

Rebuild the Choice readout for CLINC150 with its exact intent order. The BANKING77 artifact itself is not a CLINC150 classifier. Fix intent descriptions before search.

Freeze the embedding model revision during development. Count its full inference cost. All supervised references train on the same permitted labels and use declared tuning budgets.

Semantic-only performance is secondary because banks are discovered to complement TF-IDF. Failure of that secondary model does not establish failure of complementarity.

## 8. Teacher feedback packets

Generate E/F packets from the current primary-budget bank's discovery OOF predictions:

1. Compute class-normalized errors and recurrent confusion pairs.
2. Sample five contrastive pairs from those failures. Include the mistaken message and a message from the confused class that makes the distinction concrete.
3. Sample three correctly classified contrastive pairs as preservation controls.
4. Supply text, true intent, prediction, and relevant class probabilities for each message.
5. Supply current CV macro-F1 and a fixed-size aggregate diagnostic summary.
6. Ask for concepts that separate failures while preserving known distinctions.

Use a predefined class-balanced sampling policy with a cap on repeated classes and a logged, deterministic fallback if eligible pairs are scarce. Freeze text truncation and packet token limits. Do not imply that fixing one pair guarantees a macro-F1 gain.

R receives eight randomly sampled, class-balanced pairs with the same metadata and diagnostic fields, without conditioning pair selection on errors. U receives no diagnostic packet.

F additionally receives:

- Absolute latency limits and the selected bank's measured latency.
- Per-question cost estimates, with option count and numeric width distinguished.
- Archive entries containing bank definitions, quality, serving cost, and cumulative discovery costs.
- Explicit permission to propose replacements or cheaper alternatives.

Use a fixed maximum teacher output budget. Record actual input/output tokens and their monetary cost. Longer F inputs are part of its actual discovery cost, not free information.

## 9. Candidate pool and common selector

All arms use the following backend:

1. Normalize and validate proposals; deduplicate complete canonical definitions.
2. Add valid definitions to the arm's immutable historical pool.
3. Extract genuinely new definitions on discovery rows.
4. Rebuild candidate banks from the full pool under each latency limit.
5. Evaluate banks with the frozen learner on all five discovery folds.
6. Preserve nondominated quality-cost alternatives, including equal-quality cheaper banks.
7. Select one current bank for the primary latency limit using a fixed tie rule.

Default selection is budgeted greedy forward selection with up to two single-question swaps per latency limit per round. Set a maximum of 12 selected questions as an engineering cap. This cap is shared across arms and does not substitute for measured latency.

Retain prior feasible banks so a later round cannot silently discard them. An empty semantic bank, meaning TF-IDF alone, is a valid fallback.

Every unique candidate-bank CV evaluation counts against the common evaluation cap. Reuse exact results when bank, rows, folds, preprocessing, and model configuration match. Record both unique evaluations and cache hits.

Choose among feasible banks by greatest CV macro-F1. For ties within a frozen numerical tolerance, choose lower latency, then fewer questions, then canonical hash order.

Do not append every proposed feature and rely on a later permissive prune pass. Do not use a single-fold screen for removals. The same cost-aware selector applies to U, R, E, and F; only F's teacher sees costs.

## 10. Serving latency protocol

Primary workload: batched inference, batch size 64, on one frozen hardware/runtime configuration. Use the actual intended deployment workload if development establishes a different choice, then freeze that choice before confirmation.

Measure text-to-prediction latency including encoding, upstream forward passes, probability processing, TF-IDF, and downstream prediction. No cached semantic outputs are allowed in the end-to-end benchmark.

For each seed, measure the task Choice-readout reference on a fixed discovery calibration set. Define latency limits:

- `L_low = 0.5 * L_reference`
- `L_primary = 1.0 * L_reference`
- `L_high = 2.0 * L_reference`

All arms share the same absolute limits within that seed. Freeze those values before teacher search. Report them in milliseconds; do not redefine them from final outcomes.

Use a calibrated cost model for screening candidate banks, with full-bank measurements for retained archive points. If a measured bank exceeds a limit, it is infeasible and the selector uses the next feasible alternative. Record measurement work as discovery expenditure.

Benchmark final frozen banks with repeated runs, excluded warmups, synchronized GPU timing, identical input lengths/order, and interleaved arm order. Report throughput, milliseconds per text, variability, and the complete protocol.

Also report single-request p50/p95. Batch latency does not support an interactive latency claim. Do not attribute single-request overhead to launch costs without component profiling.

Check deployment latency on the final refit artifact using the same discovery calibration texts. Choose the fallback, if necessary, by the preregistered feasibility rule before final test evaluation. Test-set latency is descriptive and never triggers retrospective bank replacement.

## 11. Discovery resource limits

| Resource | Default |
| --- | --- |
| Teacher arms | U, R, E, F |
| Confirmation seeds | Five paired seeds |
| Maximum rounds | Five |
| Proposals per round | Up to 12 |
| Proposed definitions | At most 60 per run |
| Selected questions | At most 12 per bank |
| Latency limits | Three; one declared primary |
| Candidate-bank CV evaluations | Common cap, calibrated on development and frozen |
| Extraction GPU time | Common cap, calibrated on development and frozen |
| Teacher token expenditure | Common ceiling, calibrated and frozen |

The core confirmation matrix is 4 arms x 5 seeds = 20 discovery runs. Multiple latency banks come from each run's pool; they are not independent teacher runs.

Stop a run at five rounds or the first applicable resource ceiling. Finish only operations covered by its resource reservation. Specify how unfinished rounds are handled before confirmation.

Log teacher tokens/cost, unique extraction requests, GPU time, extraction wall time, candidate fits, CPU time, latency calibration, and rejected-proposal expenditure. Publish actual costs as well as ceilings.

Cache sharing can save physical work, but charge each arm its logical standalone extraction cost for fair budget accounting. Record both charged and actual shared-cache costs. Treat different compute resources as separate axes unless a conversion to monetary cost is frozen in advance.

For discovery curves, archive banks at 25%, 50%, 75%, and 100% of the extraction-compute ceiling, using the latest completed state at each checkpoint. Show actual teacher and fitting expenditure alongside those curves. Do not equate matched extraction time with matched total cost.

## 12. Evaluation and statistical analysis

### Primary endpoint

Mean paired test macro-F1 difference `F - E` for the final frozen pipelines at `L_primary`.

Proposed minimum meaningful gain: **0.010 absolute macro-F1**, or one percentage point. Register this threshold before confirmation.

Report all five paired seed differences, their mean and spread, and paired bootstrap intervals over test messages with identical resamples for compared arms. Keep seed variability and test-sampling uncertainty distinct. Shared test messages and folds are not independent experimental replications. Five seeds provide limited precision over the distribution of possible discovery runs.

### Proposed success rule

Evidence supporting the frontier extension requires:

- Mean F-minus-E test gain at least 0.010.
- Positive paired difference in at least four of five seeds.
- Actual feasibility under the primary latency limit.
- A reported interval and seed analysis consistent with the claimed level of certainty.

If the point estimate meets the practical threshold but uncertainty remains broad, label the result promising but inconclusive. A nonsignificant result is not proof of zero effect. Do not substitute a favourable secondary budget for the primary endpoint.

### Secondary reporting

- E minus R and E minus U, with their different interpretations.
- TF-IDF-plus-bank gain over TF-IDF alone.
- Quality at all three latency limits.
- Quality versus charged cumulative extraction compute, with other costs reported.
- Accuracy and weighted F1 as supplementary metrics.
- Question counts, option dimensions, concept families, redundancy, and replacement patterns.
- Final-bank concept fidelity and probability calibration/discrimination.
- Final pipeline latency and baseline dominance.

If testing multiple secondary contrasts, declare the multiplicity procedure or treat them as descriptive. Do not use McNemar results as a test of macro-F1; it evaluates paired binary correctness outcomes.

### Concept fidelity

Sample final-bank definitions randomly, rather than auditing only successful examples. Annotators are blind to arm and model output. Include clear, ambiguous, and difficult messages. Report agreement, per-concept extraction quality, and the separate human annotation budget.

SHAP or coefficients describe contributions to predictions. They do not prove correct measurement of the named concepts or establish a complete semantic explanation of a hybrid TF-IDF model.

## 13. Implementation integrity requirements

- Definition-aware cache keys and immutable revisions.
- Verified question-to-column maps, including mixed option counts.
- Fresh extraction versus matrix slicing parity for sampled subsets.
- Training-fold-only fitting of all learned preprocessing.
- Exact alignment of intent order and probability keys.
- Rejection of invalid distributions and extraction failures.
- Shared seed-level data, folds, baseline artifacts, and calibration inputs across arms.
- Reproducible candidate-evaluation accounting and resource reservations.
- Stored teacher prompts/responses, feedback example IDs, definition hashes, pool snapshots, and archive decisions.
- Final artifact hashes and freeze timestamps before confirmation/test access.

If a defect changes search-time features or diagnostics, affected trajectories cannot support teacher-mechanism claims. Rescoring final banks alone does not reconstruct the decisions made with defective inputs.

## 14. Stages and stop rules

### Stage A: BANKING77 development

Implement and check the four packet types, Choice schema, common selector, cost accounting, cache identity, and data boundaries. Run a bounded development check. Calibrate resource caps and audit the measurement channel.

Stop before confirmation if extraction remains unreliable, selection cannot produce feasible banks, accounting cannot enforce comparable ceilings, or integrity checks fail. Report that engineering outcome without claiming a negative test of H1.

### Stage B: Freeze

Complete the checklist below. Create a timestamped protocol and code/configuration hash. Do not inspect CLINC150 confirmation or test outcomes while making these choices.

### Stage C: Independent confirmation

Execute all 20 registered runs. Freeze the selected banks at each latency limit and resource checkpoint. Perform the one-time confirmation evaluation, final refit, registered feasibility checks, and official test evaluation without adaptive revisions.

### Stage D: Interpret

| Outcome | Decision |
| --- | --- |
| E beats R; F meets the primary success rule versus E | Expand the frontier study to another task and additional label budgets |
| E beats R; F does not establish an advantage | Support example-driven discovery, leave frontier extension unsupported |
| E resembles R | Example access may explain the benefit; error targeting remains unestablished |
| No discovered bank adds useful information to strong references | Report that practical limitation and stop expansion under this setup |
| Concepts are unreliable | Pause discovery with this extractor; measurement is an unresolved prerequisite |
| Results vary substantially across seeds | Report inconclusive evidence; any larger study requires a separate prospective plan |

The primary decision stays tied to the registered endpoint. Useful secondary compression or fidelity findings do not replace a failed H1 test.

## 15. Freeze checklist

- [ ] Teacher revision, settings, and token ceiling.
- [ ] TinyJev revision, extraction prompt, option format, and truncation.
- [ ] Development instrument audit and acceptance criterion.
- [ ] CLINC150 dataset hash, exclusions, five seeds, sampled IDs, and folds.
- [ ] Common initial bank and current-bank rule.
- [ ] TF-IDF settings, LR configuration, and one-time tuning procedure.
- [ ] Embedding reference and direct-Choice intent definitions.
- [ ] R/E pair sampling, class caps, fallbacks, truncation, and packet fields.
- [ ] Cost descriptors shown only to F.
- [ ] Shared selector, swap limits, tie tolerance, and feasibility fallback.
- [ ] Actual latency limits and benchmark protocol per seed.
- [ ] Candidate-evaluation, extraction, and teacher resource ceilings.
- [ ] Cache accounting, resource reservation, and interrupted-round rules.
- [ ] Primary contrast, meaningful-effect threshold, intervals, and secondary-analysis policy.
- [ ] Confirmation/test access boundary and final-refit procedure.
- [ ] Code/configuration hash and protocol freeze timestamp.

Items left unresolved are development decisions. Resolve them once before confirmation; do not treat the brief as a completed preregistration until then.

## 16. Deliverables

1. Frozen protocol and machine-readable configuration.
2. Data/split identities, model revisions, and integrity-check results.
3. Teacher prompts, feedback packets, definitions, and candidate histories.
4. Complete quality-cost archives and resource ledger for every run.
5. Frozen selected-bank artifacts and predictions for each arm/seed/budget.
6. Final report with the primary contrast, all seed results, baselines, cost curves, and concept audit.
7. Explicit conclusion about example targeting, frontier feedback, and practical competitiveness as separate claims.

## 17. References

[1] Barlier, M., and Skrlj, B. (2026). *LLMs as Feature Engineers for Text-and-Tabular Prediction*. arXiv:2609.21894v1, submitted 18 September 2026. https://arxiv.org/abs/2609.21894  
Methodology and prompts: https://arxiv.org/html/2609.21894v1

[2] Larson et al. (2019). *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*. Authors' dataset repository: https://github.com/clinc/oos-eval  
Paper: https://arxiv.org/abs/1909.02027

[3] Cawley, G. C., and Talbot, N. L. C. (2010). *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*. JMLR 11:2079-2107. https://jmlr.org/papers/v11/cawley10a.html

The adaptive-selection distinction in Sections 5 and 12 is motivated by [3]. Numerical defaults elsewhere are proposed design choices, not results or prescriptions from the cited papers.
