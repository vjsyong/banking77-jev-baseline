# BANKING77 Jev and classical ML baseline

This bundle runs a small, reproducible comparison on the official BANKING77 train and test splits:

1. Word and character TF–IDF with Logistic Regression, Linear SVM, and Complement Naive Bayes.
2. Jev directly choosing one of the 77 intents.
3. A classical learner trained on 16 Noul semantic signals extracted by Jev.

See [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) for the research hypothesis, controls, and follow-on probe-discovery protocol.

The package does not contain BANKING77 or model weights. The data loader fetches the public dataset from Hugging Face on the machine where you run the experiment. The Jev client targets a local TinyJev/TinyJev-compatible System One endpoint; it does not need to send data to a cloud API.

## 1. Create the Python environment

Python 3.10 to 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run downloads the BANKING77 dataset. Keep the official train/test split intact.

## 2. Run the classical baselines

```bash
python banking77_baseline.py classical --out runs/banking77
```

This fits the TF–IDF representation on training text only, fits three classical models, and evaluates each on the official test set. It writes metrics, prediction files, and confusion-pair diagnostics under `runs/banking77/`.

## 3. Start a local Jev-compatible endpoint

One option is TinyJev through Ollama. Install Ollama and a TinyJev release that supports `/v1/systemone`, then pull the model:

```bash
ollama pull parable/tinyjev
```

Ollama normally serves on `http://127.0.0.1:11434`. TinyJev documents the System One route at `/v1/systemone`; the command below checks connectivity and response parsing on two short examples before you start the complete run.

```bash
python banking77_baseline.py extract-jev \
  --endpoint http://127.0.0.1:11434/v1/systemone \
  --model parable/tinyjev \
  --out runs/banking77 \
  --max-rows-per-split 2
```

If the endpoint is on another machine, use a private connection or SSH tunnel and leave the unauthenticated local service bound to loopback. For example, from the machine running this bundle:

```bash
ssh -L 11434:127.0.0.1:11434 your-user@your-model-host
```

Then use the same `127.0.0.1:11434` endpoint. Avoid exposing a no-auth inference port to the public internet.

## 4. Extract Jev outputs and evaluate

After the smoke test succeeds, run on all rows:

```bash
python banking77_baseline.py extract-jev \
  --endpoint http://127.0.0.1:11434/v1/systemone \
  --model parable/tinyjev \
  --out runs/banking77 \
  --workers 1
```

Each request asks the same 16 semantic questions and a direct 77-way intent question. Questions are sent together per text. The SQLite cache saves completed results, so an interrupted run can resume. The source labels are never included in Jev requests. `--workers` defaults to one; increase it only after checking that your serving backend handles concurrent requests safely.

Then evaluate the cached Jev outputs:

```bash
python banking77_baseline.py evaluate-jev --out runs/banking77
```

The semantic feature models are trained using the official training labels and only the 16 Noul values as inputs. The direct Jev score is calculated on the official test split. Results include macro-F1, accuracy, and per-class diagnostics. `jev_features.csv` and `jev_predictions.csv` retain the input text and labels for inspection, so treat them as research data and do not publish them without checking dataset terms.

## 5. Read the results

- `metrics.json`: classical and Jev model scores, run sizes, and timing.
- `tfidf_*_predictions.csv`: reference model predictions and per-row labels, one file per classifier.
- `jev_predictions.csv`: direct Jev selections and option probabilities.
- `jev_features.csv`: extracted semantic signals for train and test rows.
- `jev_cache.sqlite`: resumable extraction cache.
- `jev_manifest.json`: probe schema, current extraction request count, cache reuse, and latency summaries.

To run the offline client and cache checks without downloading BANKING77 or starting a Jev server:

```bash
python -m unittest discover -s tests -v
```

Macro-F1 is the primary metric because there are 77 classes. Accuracy is included for continuity with familiar intent benchmarks. The classical hyperparameters are fixed up front; this is a baseline run, not a large model tournament. Do not revise probe wording in response to official test errors. Use training-only cross-validation for the next discovery round, then keep the official test set for final confirmation.

## Suggested next round

First compare the manual 16-signal bank with direct Jev and the TF–IDF systems. Inspect errors from out-of-fold predictions on the training split. Ask a stronger teacher model to propose probe additions, splits, or replacements based only on those training-fold errors. Deduplicate proposals, append a versioned probe bank, and rerun Jev extraction with the persistent cache. Keep the same evaluation folds and model panel. Compare performance against probe count and measured Jev latency. Evaluate a frozen winner on the official test set once.

The bank is deliberately small and hand-written as a starting point. It is not expected to encode all 77 intent distinctions. In particular, 16 broad signals may underperform a label-aligned probe or a strong word/character classifier.

## Dataset citation

Casanueva, I., Temcinas, T., Gerz, D., Henderson, M., and Vulic, I. (2020). *Efficient Intent Detection with Dual Sentence Encoders*. Proceedings of the 2nd Workshop on NLP for Conversational AI. Dataset: PolyAI/BANKING77, CC BY 4.0.

## Troubleshooting

- If Hugging Face dataset loading fails, try the commands from a machine with internet access or set `HF_HOME` to a writable cache directory. The code pins `datasets` below version 4 because the legacy BANKING77 loading script is not compatible with all newer releases.
- If the Jev endpoint returns a 404, check that the URL ends in `/v1/systemone` and that the serving version supports the TinyJev System One route.
- If the response shape differs, run the two-row smoke test first and inspect the HTTP error before launching the full extraction.
- If disk space is limited, choose a small `--out` folder on a volume with space for the SQLite cache and CSVs.
