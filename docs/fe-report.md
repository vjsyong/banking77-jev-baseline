# FE discovery, CLINC150: registered confirmation report

Date: 2026-10-05 (runs completed 04:36 UTC). Protocol: `docs/frontier_guided_discovery_experiment_brief.md`
(registered design), `discovery/fe_discovery/frozen_protocol.json` (freeze + instrument audit),
`docs/fe-ops-notes.md` (deviations D1-D8).

## What was run

Five seeds x four feedback arms under a fixed charged serving budget with a 1,000-label
discovery allowance:

- **U** unguided, **R** random-contrastive, **E** example-guided, **F** frontier-guided
  (E plus the cost/frontier/replacement layer)

Each arm produced a question bank per latency limit (low / primary / high; limits 82 / 163 / 326
ms-text vs banks at ~20-40 ms, so the limits did not bind). Banks were frozen before any
confirmation access (`frozen_banks.json`, sha256 `e433f406...`, 60/60 banks). One-shot
confirmation on held-out rows (no revision), then a final refit on the full 3,000-label
allowance and evaluation on the official test split (4,500 messages; macro-F1 over the 150
in-scope intents).

## Results

### Registered contrasts (percentage points; paired per seed; 1,000x bootstrap over test messages)

| contrast | mean | per-seed deltas | bootstrap 95% | seeds > 0 |
|---|---:|---|---:|---:|
| **primary F − E** | **+0.02** | [-0.17, -0.39, +0.08, +0.30, +0.27] | [-0.35, +0.36] | 3/5 |
| secondary E − R | -0.14 | [-0.16, +0.37, -0.56, -0.08, -0.25] | [-0.50, +0.22] | 1/5 |
| secondary E − U | -0.22 | [-0.98, -0.07, +0.08, -0.35, +0.20] | [-0.58, +0.13] | 2/5 |
| secondary F − U | -0.21 | [-1.15, -0.46, +0.16, -0.06, +0.47] | [-0.54, +0.14] | 2/5 |

All four contrasts are indistinguishable from zero. No feedback condition separates from any
other at the primary limit.

### Absolute levels (test macro-F1, mean over 5 seeds)

| system | score |
|---|---:|
| U (unguided discovery) | 0.9073 |
| R (random-contrastive) | 0.9064 |
| F (frontier-guided) | 0.9052 |
| E (example-guided) | 0.9050 |
| TF-IDF + LR | 0.8706 |
| bge-small embeddings + LR | **0.9399** |
| task-Choice readout (fit on discovery split; as registered) | 0.8987 |
| task-Choice readout (refit on full 3,000; supplementary, post-registration) | 0.9059 |
| direct zero-shot choice | 0.7020 |

### Compute (H4)

Mean charged discovery compute per arm: U 616 s / R 635 s / E 594 s / F 632 s. No material cost
difference, including for the arm with the economic layer. Quality-vs-budget checkpoints are
near-identical across arms: ~0.875 of final CV at 25% of the charged budget, ~0.898 at 50%,
~0.908 by 75%, then flat.

## Interpretation

1. **Feedback source does not matter in this configuration.** Unguided matches example-guided and
   frontier-guided; the frontier layer's cost/frontier/replacement economics neither improved
   quality nor reduced charged compute.
2. **The discovery apparatus reaches readout parity, not beyond.** Arms at ~0.905-0.907 slightly
   beat TF-IDF (0.871) under the same allowance and hold their own against the single-question
   readout (~0.899-0.906), but a frozen bge-small embedding baseline wins outright at 0.940.
   Compact Jev question banks were not the best use of the 3,000-label budget on this task.
3. **Consistent with BANKING77.** Frontier feedback was unestablished there as well, and
   representation (not feedback) was the lever. Here even the full discovery loop only ties the
   simplest Jev readout.

## Limitations

- 5 seeds. The primary contrast CI spans about ±0.35 pp, so effects smaller than ~0.4 pp cannot
  be excluded, although the means are essentially zero and seed signs are split.
- CLINC150 comprises short, templated utterances; results may not transfer to messier text.
- Model classes differ (a 0.6B frozen QA model vs a ~33M retrieval-trained embedding model); both
  are used here as frozen extractors with a trained linear head, which is the operational
  comparison.

## Artifacts

- `runs/clinc150/fe_discovery/frozen_banks.json` - banks with hashes, frozen before confirmation access
- `runs/clinc150/fe_discovery/stage_eval_partial.json` - per-bank confirmation and test scores (all 60)
- `runs/clinc150/fe_discovery/confirm_report.json` - baselines, registered analysis, H4 checkpoints
- `runs/clinc150/fe_discovery/readout_3k_supplementary.json` - supplementary readout refit
- Per-run prompts, responses, and round logs under `runs/clinc150/fe_discovery/stage_c/seed_*/[UREF]/`
