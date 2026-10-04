# banking77-readout-v1

Frozen, deployable readout for the BANKING77 77-intent task: **TinyJev-0.6B Choice
probabilities -> Logistic Regression**.

See `manifest.json` for the full frozen configuration (intent order, clip constants,
upstream model revision, training/evaluation metadata).

## Files

- `readout_lr.joblib` — fitted sklearn pipeline (`StandardScaler -> LogisticRegression`)
- `labels.json` — the 77 intents in **exact feature order** (alphabetical)
- `manifest.json` — everything frozen with this artifact

## Input contract

- Features: `log(clip(p, 1e-6, 1.0))` over the 77 Choice probabilities, in
  `labels.json` order. Input dicts may have keys in any order; missing intents raise.
  Tiny probabilities are clipped (no `log(0)`); all-zero vectors are handled.
- Accepts `{intent: probability}` (single), list of dicts (batch), `ndarray (77,)` or
  `(n, 77)` in labels order. Single/batch requests give identical results.

## Usage

```python
import sys; sys.path.insert(0, "<repo>/readout")
from readout_pipeline import Banking77Readout

ro = Banking77Readout.load("<repo>/readout/banking77-readout-v1")
out = ro.predict({intent: p for intent, p in model_distribution.items()})
# {"label": "card_arrival", "confidence": 0.97, "probabilities": {...}, "top3": [...]}
```

## Evaluated quality (official test split, 3,080 rows)

- macro-F1 **0.9008** · accuracy **0.9006** · top-3 **0.9766**
  (vs 0.7574 / 0.7633 / 0.9078 for the raw argmax of the same Jev probabilities;
  TF-IDF + LogReg reference: 0.9094 / 0.9091)

## Provenance

- Upstream: `AnkitAI/TinyJev-0.6B` (HF snapshot `c559c2f7ea95…`), TinyJev package
  0.1.3, torch 2.14.1+cu130; probabilities extracted with the batched width-grouped
  extractor (`tools/batched_extract.py`).
- Reproduce the artifact: `./venv/bin/python readout/export_v1.py` (deterministic;
  asserts the recorded metrics).
- Deployment parity tests: `./venv/bin/python readout/test_parity.py` (5/5: artifact
  reproduction, shuffled keys, tiny probabilities, single-vs-batched, guard rails).
