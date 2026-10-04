# TinyJev semantic probes: baselines, label budgets, and guided feature discovery

Research code for measuring what a small question-answering model contributes as a **feature extractor** for text intent classification. Each message is presented to TinyJev-0.6B as one or more schema-bound categorical questions ("Choice" and "Noul" formats); the model's answer distributions become numeric features for conventional classifiers. The repository contains three completed studies on BANKING77 and one registered experiment currently running on CLINC150, each with a report in `docs/`.

**Headline:** a logistic-regression readout on frozen TinyJev-0.6B lifts BANKING77 macro-F1 from **0.7574** (direct argmax) to **0.9008** (+14.3pp) without changing any model weights.

## Studies

| # | Study | Task | Status | Report |
|---|-------|------|--------|--------|
| 1 | Baseline: TF-IDF vs. direct Jev choice vs. semantic-probe features | BANKING77 | complete | `docs/consolidated-report.md` |
| 2 | Label-budget readout and a frozen deployable pipeline | BANKING77 | complete | `docs/readout-v1-report.md` |
| 3 | Guided probe discovery: pilot, representation x selection factorial, pruning study | BANKING77 | complete | `docs/discovery-pilot-report.md`, `docs/factorial-report.md`, `docs/pruning-report.md` |
| 4 | Example-guided discovery with a frontier teacher (arms U/R/E/F) | CLINC150 | in progress | `docs/frontier_guided_discovery_experiment_brief.md`, `docs/fe-freeze.md`, `docs/fe-ops-notes.md` |

## Headline results

**Probability readout, frozen TinyJev-0.6B, BANKING77 (official split, 10,003 train / 3,080 test, macro-F1):** train a logistic regression on Jev's full 77-way probability vector (**log probabilities, scaling, logistic regression**; Jev stays frozen) and macro-F1 improves from **0.7574** (direct argmax) to **0.9008**, +14.34 percentage points, without changing the model weights. The readout step itself adds about 1 ms per text on top of the Jev pass; the full text-to-prediction pipeline runs at p50 62.5 ms single / 30.2 ms batched per text.

**Reference baselines (same split, macro-F1):**
- TF-IDF + LogReg: **0.9094** (accuracy 0.9091; LinearSVC 0.9087; ComplementNB 0.8068). The strongest classical baseline still leads the readout slightly, and is far cheaper at inference (0.229 ms/text batched).
- Jev direct 77-way choice: **0.7574** (accuracy 0.7633, top-3 0.9078).
- 16 hand-written Noul probes, best learner (ExtraTrees): 0.5707 (LogReg 0.4038). The small hand-written bank underperforms, as the plan anticipated.

**Label budgets (test macro-F1, two label seeds, chart in `runs/banking77/discovery/label_curves.png`):** the curves answer a practical question, how many labels do you actually have? At the smallest budget (250) the direct argmax is still the strongest single signal; from 500 through about 5,000 labels the readout leads both argmax and TF-IDF (at 1,000: mean 0.8438 vs 0.7350 for TF-IDF); by the full 10,003-label budget TF-IDF catches up and overtakes (0.9094 vs 0.9008). The readout is the better use of a fixed serving budget while labels are scarce; at large budgets TF-IDF wins on both quality and latency.

**Practical takeaway:** if you use a Jev-style model for classification and already have labelled examples, keep the full probability vector and train a small classifier on top of it. The readout costs little relative to the model pass.

**Guided discovery (BANKING77, 1,000-label allowance, test macro-F1):**
- Pilot: frontier feedback (cost model + quality/cost archive) did not reproducibly change the discovered banks (frontier minus unguided +2.16pp mean, carried by a single seed). Banks did improve over the hand-written set (0.4387 to 0.5245 best).
- Factorial (2x2: representation x selection): representation is the lever. Mixed Choice-format banks reached **0.6956** (strict) and 0.5009 (pareto) vs 0.5295 / 0.4233 for Noul-only; paired representation effect +16.6pp (strict). The selection variant changed compression cost only.
- Pruning study: **NO-GO** at the preregistered standard (at most 1pp macro-F1 loss at 25% or better measured-latency reduction). Conservative pruning passed the latency bar (25% to 43% saved) but failed quality (0/3 and 1/3 banks passing).

**FE-discovery (CLINC150) is in progress:** example-guided vs. frontier-guided discovery under a fixed serving budget, four feedback arms, five seeds, one-shot confirmation on held-out splits. The instrument audit and protocol freeze are complete; the campaign is running. No results to report yet.

## Repository layout

```
banking77_baseline.py   bundle runner: classical baselines, Jev extraction, evaluation
probes.json             16 hand-written Noul probes
EXPERIMENT_PLAN.md      original hypothesis, controls, and discovery protocol (BANKING77)
RUN_NOTES.md            run log for the baseline phase: environment, batching switch, caveats
readout/                choice-readout pipeline, exported artifact, parity tests, benchmark
discovery/              discovery program
  label_curves.py       label-budget learning curves
  teacher_client.py     frontier teacher client (OpenAI-compatible endpoint)
  pilot/                controlled pilot, 2x2 factorial, evaluators
  pruning_study/        pruning screens, feature snapshots, deployment evaluation
  fe_discovery/         CLINC150 experiment: schema, extractor, selector, campaign runner
docs/                   one report per study, design briefs, freeze and ops records
tools/                  batching accelerators, probe audits, latency probes
runs/                   committed result artifacts: summaries, prompts, evaluations
tests/                  bundle unit tests (no network or model required)
data/                   derived dataset samples (see Data and provenance)
```

## Reproducing

Two environments are used. The evaluation environment covers data loading, models, and the discovery harness; the serving environment covers TinyJev and torch.

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
python -m venv venv-serve && venv-serve/bin/pip install -r requirements-serve.txt
```

BANKING77 downloads from Hugging Face on first use. The legacy BANKING77 loader requires `HF_DATASETS_TRUST_REMOTE_CODE=1` in this stack (environment variable only; the bundle code is unmodified).

Start a local Jev endpoint (same or another machine; keep unauthenticated ports bound to loopback or reachable only over a private link):

```bash
tinyjev serve TinyJev-0.6B --backend torch --device cuda   # http://127.0.0.1:8077/v1/systemone
```

Baseline run:

```bash
python banking77_baseline.py classical --out runs/banking77
python banking77_baseline.py extract-jev \
  --endpoint http://127.0.0.1:8077/v1/systemone \
  --model TinyJev-0.6B --out runs/banking77 --workers 1   # smoke test first: --max-rows-per-split 2
python banking77_baseline.py evaluate-jev --out runs/banking77
```

`tools/batched_extract.py` is the faster, cache-compatible extraction path used for the reported numbers (see `docs/batching-research.md` for the measurements and the equivalence checks). Discovery and readout scripts live under `discovery/` and `readout/`; run them from the repository root, and use `--help` where a script takes options. Experiments write their artifacts under `runs/`.

**Portability note:** `banking77_baseline.py` and `readout/` are self-contained, but most scripts under `discovery/` and two under `tools/` resolve the repository root from the development path (`/home/xrim/banking77-jev-baseline`). When cloning elsewhere, adjust the `HERE` constant near the top of those scripts (or clone to that path). A portability cleanup is scheduled after the in-flight study completes.

## Data and provenance

- **BANKING77** (PolyAI): CC BY 4.0. Casanueva et al., 2020, *Efficient Intent Detection with Dual Sentence Encoders*. Fetched at runtime from Hugging Face; derived predictions are committed under `runs/banking77/`.
- **CLINC150**: from the `clinc/oos-eval` release; this repository uses in-scope (150-intent) classification only, out-of-scope queries excluded. Derived samples, folds, and hashes are in `data/clinc150/`, prepared by `discovery/fe_discovery/data_clinc.py`. See the source release for dataset terms.
- **TinyJev**: package `tinyjev` (MIT) with the TinyJev-0.6B model (see its model card for terms). All inference in this repository ran on a single RTX 3090 in fp16.
- **Teacher model** (discovery studies): a frontier LLM served over an OpenAI-compatible endpoint. The model revision and all settings are recorded in the frozen protocols and per-run artifacts. API keys are read from local user configuration at runtime and are never stored in this repository.
- Text and labels are retained in derived artifacts; check dataset terms before redistributing.

## Research discipline notes

- Thresholds and acceptance rules are preregistered before runs (see the experiment briefs and `discovery/fe_discovery/frozen_protocol.json`). Criteria that failed as first written are kept in the record, including a corrected audit criterion in the freeze.
- Probe discovery never uses test-split information: candidate banks are selected on training-only folds, and final banks are frozen before any confirmation or test evaluation.
- Negative results are reported as such: the pruning study's NO-GO and the pilot's null on frontier feedback are part of the record, not footnotes.

## License

Code in this repository is released under the MIT License (see `LICENSE`). Dataset-derived artifacts (`data/`, and text/label/prediction files under `runs/`) remain subject to their source dataset terms (BANKING77: CC BY 4.0; CLINC150: see the `clinc/oos-eval` release). Model weights and the `tinyjev` package keep their own licenses.
