# Experiment plan: discovering compact semantic features for intent classification

## Idea

Use a semantic model as a feature generator. For each customer message, ask a fixed or iteratively improved set of short questions and record each answer as a numeric Noul score. Train a conventional classifier on those scores to predict the intent. Compare that pipeline with direct intent selection by Jev and with strong, inexpensive text-classification baselines.

The longer-term experiment uses a teacher model to propose additions, splits, or replacements for the question bank. A bounded search evaluates these candidate banks on training-only validation folds and keeps banks that improve the quality/cost trade-off. The official test split is reserved for a final evaluation of a frozen choice.

## Hypotheses

**Primary hypothesis:** a compact bank of semantic questions can provide useful intent information, and teacher-guided refinement can improve downstream macro-F1 over the initial hand-written bank while keeping inference cost bounded.

**Baseline hypothesis:** a linear classifier trained on Jev's semantic scores may outperform Jev's direct 77-way choice because the classifier can combine signals consistently across intents.

**Cost hypothesis:** a smaller semantic-question bank may preserve useful predictive quality while lowering probe count or measured latency relative to the initial 16-question bank. The initial bank can also reveal whether semantic extraction is worth its added work compared with a single direct Choice question. These are empirical comparisons; the baseline does not assume either result.

## Initial BANKING77 run

The included runner establishes three reference groups:

1. **Text-only classical models:** word and character TF–IDF with Logistic Regression, Linear SVM, and Complement Naive Bayes.
2. **Direct Jev:** one Choice question over the 77 intent names.
3. **Semantic-feature models:** Logistic Regression, Linear SVM, and Extra Trees using the 16 Noul scores in `probes.json`.

Use the official `PolyAI/banking77` split (10,003 training examples and 3,080 test examples). Macro-F1 is the primary quality measure; also report accuracy and weighted F1. Report the Jev request latency distribution, total extraction wall time, number of uncached requests, and probe count as cost descriptors. This local runner does not estimate monetary API cost or GPU energy.

## Controls and data handling

- Preserve the supplied train/test split. Never use test examples or test errors to create, select, or rewrite probes.
- Do not include intent labels in the text sent to Jev. The implementation sends only the customer text and question definitions.
- Fit the TF–IDF vectorizers and downstream semantic-feature classifiers using training data only.
- Freeze the probe wording, model identifier, endpoint version, classifier settings, and evaluation split before the final test run. Record any changes in the run notes.
- Keep the text and label columns in local output files private according to the dataset's terms.
- Run the included two-row endpoint smoke test before the full extraction. The smoke-test subset is only a connectivity check; its scores are not benchmark results.

## Follow-on probe discovery

1. **Establish folds.** On the official training split only, generate stratified five-fold out-of-fold predictions for the text-only and fixed-probe systems. Use the same folds for every candidate. If data volume is constrained, start with a smaller stratified cross-validation setup and record it.
2. **Find training errors.** Summarize per-intent recall, confusion pairs, and examples misclassified out of fold. Share only training-fold errors with the teacher model.
3. **Generate candidates.** Ask the teacher for concise, answerable semantic questions likely to distinguish the confused intents. Require each suggestion to specify the signal, expected distinction, and overlap with existing questions. The teacher proposes; it does not see held-out validation or test labels.
4. **Bound the search.** Deduplicate candidates, cap the number of additions per round, and evaluate a fixed number of probe banks per round. Version every bank (for example, `probes_v1.json`, `probes_v2.json`) instead of silently editing prior definitions.
5. **Evaluate on the same folds.** Re-extract candidate features for training data and score the same classifier panel. Compare macro-F1 with probe count, uncached requests, and measured latency. Retain candidates that improve a chosen quality/cost trade-off; do not select on the official test set.
6. **Stop at saturation.** Stop when a predeclared number of search rounds produces no meaningful improvement to the Pareto frontier, or when the fixed search budget is spent. Define “meaningful” and the budget before running the search.
7. **Final confirmation.** Freeze the selected bank, retrain on the complete training split, and evaluate the frozen systems on the official test split once. Report the full baseline table and the cross-validation selection process.

## Decision rule to preregister

Before starting teacher-guided discovery, choose:

- the practical minimum macro-F1 improvement worth keeping;
- the acceptable increase in extraction time or compute;
- the probe-count or request budget per round;
- the number of no-improvement rounds that ends the search.

There is no universal threshold for these values. Set them based on the intended use and available hardware, then leave them fixed during the search.

## Limitations

The first 16 probes are a small, manually written starting point rather than a complete representation of all 77 intents. Jev probabilities may be poorly calibrated, and broad questions can be redundant. The direct Choice baseline and Noul feature pipeline use different output formats, so interpret their latency in the context of the shared request schema. In this runner, one System One request contains the direct Choice and all 16 Noul questions; changing the number of questions changes request work, but the endpoint may not scale linearly with question count.
