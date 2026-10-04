# Round-1 follow-ups — probe audit, scaled linear models, choice readout, cost split (2026-10-04)

All four from existing outputs (readout + audit + scaled models need no inference;
timing split used the batched infra for a 512-text slice). Raw outputs in
`runs/banking77/analysis/`; scripts in `tools/`.

## 1. Probe audit (`tools/audit_probes.py` → `analysis/probe_audit.json`)

- **No saturation**: fraction <0.05 = 0.000 and >0.95 = 0.000 for all 16 probes;
  observed ranges ≈ [0.03–0.95], q05–q95 ≈ [0.24–0.85].
- **Compression**: every probe's mean sits in 0.49–0.63 with std 0.08–0.16;
  15–43% of values fall within ±0.05 of 0.5. The Noul head speaks in a narrow band.
- **Redundancy / common mode**: strong positive correlations throughout —
  `money_outgoing ~ money_incoming` **+0.81** (a direction distinction the model does
  not see), `requests_an_action ~ cancel_or_reverse` +0.77. The same texts score high
  across many probes (see examples below): a large global response factor.
- **Extraction errors on clear cases** (examples from the audit):
  - `physical_or_virtual_card`: "Nothing goes through on my card." → p=**0.17** (bottom
    band), while "Can my 19 year old daughter open a savings account" → 0.87.
  - `unauthorised_activity`: "I want to change my pin number from an ATM." → p=**0.85**.
  - Where it works: `currency_conversion` separates exchange intents cleanly
    (exchange_rate 0.857 / exchange_via_app 0.863 vs passcode_forgotten 0.282).
- **Intent separability (eta²)**: top `currency_conversion` 0.49, `money_outgoing`
  0.47, `money_incoming` 0.47; bottom `asks_for_information` 0.19, `cash_or_atm` 0.19,
  `unauthorised_activity` 0.21.
- Verdict vs the two unknowns: **not saturation** — the failure mode is compression +
  common-mode + locally unreliable extraction. The documented TinyJev yes-bias does
  not show as yes-saturation here (question-form likely mitigates); the amplitude
  problem remains. (For contrast, the mail-triage spike measured the opposite bias
  there: median p≈0.30, never >0.59 — the collapse is domain-dependent.)

## 2. Scaled linear models with CV (`tools/scaled_linear.py`)

| variant | C* | CV macro-F1 | test macro-F1 | test acc |
|---|---:|---:|---:|---:|
| StandardScaler → LogReg | 10 | 0.5331 | **0.5286** | 0.5338 |
| StandardScaler → LinearSVC | 10 | 0.4450 | 0.4498 | 0.4756 |
| LogReg (unscaled, CV) | 30 | 0.5220 | 0.5222 | 0.5302 |
| LinearSVC (unscaled, CV) | 30 | 0.4442 | 0.4488 | 0.4750 |
| *(original fixed C=1: LR 0.4038 / SVM 0.4307; ExtraTrees ref 0.5707)* |

- **Tuning >> scaling**: CV-chosen C alone adds +11.8pp to LR (0.4038 → 0.5222);
  standardization then adds +0.6pp (0.5286). The earlier "tree beats linear by 14pp"
  was mostly a suboptimal C.
- **Residual ET gap**: 4.2pp (0.5707 vs 0.5286) — genuine nonlinear structure remains,
  but much smaller than it appeared.

## 3. Choice-probability readout (`tools/choice_readout.py` → `analysis/choice_readout.json`)

| system | macro-F1 | accuracy | top-3 |
|---|---:|---:|---:|
| argmax (direct Jev, baseline) | 0.7574 | 0.7633 | 0.9078 |
| **LR on 77 log-probs (C=1)** | **0.9008** | **0.9006** | **0.9766** |
| LR on 77 log-probs + 16 probes | 0.8977 | 0.8977 | 0.9779 |
| ExtraTrees on 77 log-probs | 0.8735 | 0.8737 | 0.9614 |
| *(reference: TF-IDF LogReg 0.9094)* | | | |

- **A linear readout of Jev's own probability vector improves it by +14.3pp macro-F1**
  — from 0.7574 to 0.9008, within 0.9pp of TF-IDF LogReg, top-3 at 97.7%.
- The 16 probes **add nothing** (−0.3pp, within noise).
- Trees lose to the linear readout here (0.8735) — the nonlinearity advantage was
  specific to the weak probe features, not the probability vector.

## 4. Cost split, batched (`tools/timing_split.py`, 512 texts)

| config | forward | readout | total |
|---|---:|---:|---:|
| choice-only | 29.1 | 0.7 | **29.9 ms/text** |
| probes-only | 37.0 | 3.4 | **40.5 ms/text** |
| combined (production) | 66.4 | 4.3 | **70.7 ms/text** |

Additive (29.9 + 40.5 ≈ 70.7): no penalty for keeping both in one extraction, and the
probe pass costs **+40.5 ms/text (2.4× the direct pass)** — a real cost for zero
measured task gain on this dataset with this bank.

## Reading

1. **Unknown #1 (does Jev extract reliably):** for the Noul probes — demonstrably not
   (compression, common-mode, clear-case errors). For the intent head — yes, and richly:
   the readout lifts it near TF-IDF.
2. **Unknown #2 (do the outputs preserve 77-way distinctions):** the probability vector
   does (readout 0.90 / top-3 0.98); the 16 broad probes do not.
3. The "classical learner on Jev outputs" hypothesis is **confirmed in corrected form**:
   the right substrate is the 77-way choice distribution, not the hand-written probes.
4. Next options: (a) probe-bank v2 designed from training-fold confusions
   (entity × operation × subtype) as originally planned — now with a clear bar to beat
   (the probe-free readout); (b) productionize the readout (calibration, thresholding);
   (c) direct-only config at ~30 ms/text if probes are retired.
