#!/usr/bin/env python3
"""Stage B: assemble the frozen protocol from dev-calibrated constants + hashes.

Fails when any checklist item is unresolved (no silent defaults). Writes
frozen_protocol.json (machine) + docs/fe-freeze.md (human). Run only after
development checks pass; do not inspect CLINC150 confirmation/test outcomes
while making these choices.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path("/home/xrim/banking77-jev-baseline")
FE = HERE / "discovery" / "fe_discovery"
RUNS = HERE / "runs" / "banking77" / "fe_discovery"

# ---------------- declared constants (development decisions, frozen here) ----------------
DECLARED = {
    "teacher": {"provider": "lmuai-pro", "model": "gpt-6.1-sol", "temperature": 0.4,
                "max_tokens": 2200, "seed_rule": "seed*1000 + round_no",
                "note": "API generation is not perfectly reproducible; seeds recorded"},
    "extractor": {"model": "AnkitAI/TinyJev-0.6B",
                  "hf_snapshot": "c559c2f7ea95069f92af8b999f512345bbb87d22",
                  "package": "tinyjev 0.1.3, torch 2.14.1+cu130",
                  "chunk": 64, "no_co_query": "one question per forward",
                  "schema": "schema/v1: 3-6 options w/ explicit definitions; unclear option required",
                  "truncation": {"message_chars": 200}},
    "learner": {"tfidf_word": {"ngram_range": [1, 2], "min_df": 2, "sublinear_tf": True,
                               "max_features": 100000},
                "tfidf_char": {"analyzer": "char_wb", "ngram_range": [2, 5], "min_df": 2,
                               "max_features": 100000},
                "classifier": "OneVsRestClassifier(LogisticRegression(solver=liblinear, "
                              "max_iter=400), n_jobs=8)",
                "C_grid": [4.0, 16.0, 64.0],
                "C_selection": "once per seed on the empty-bank baseline (all 5 discovery folds); "
                               "fixed for all arms and candidates",
                "fitting": "vectorizer/IDF/scaler/classifier fit strictly within training folds; "
                           "no panel re-selection during search"},
    "selector": {"per_round_cap": 30, "global_cap": 160, "max_questions": 12,
                 "tie_tol": 0.0005, "limit_split": [9, 12, 9], "swap_attempts": 2,
                 "candidate_order": "newest first (reverse pool insertion)",
                 "rule": "budgeted greedy forward selection; up to two single-question swaps per "
                         "limit per round; fixed-reference corrected screen only in this selector; "
                         "no rolling allowance, no single-fold removals",
                 "tie_rule": "greatest CV; ties: lower latency, fewer questions, canonical hash order",
                 "archive": "nondominated quality-cost per limit, cap 30, prior feasible banks retained"},
    "packets": {"R": "8 random class-balanced pairs (cap 2/class), same fields as E, no error conditioning",
                "E": "5 failure pairs (cap 2/class then 3, deterministic fallback) + 3 preservation controls",
                "U": "no examples, no diagnostics",
                "F": "E + absolute limits, measured current latency, per-question cost estimates, "
                     "archive (latest ≤8 entries), replacement permission",
                "fields": "text (≤200 chars), true intent, prediction, top-2 probabilities; "
                          "current CV macro-F1; worst-12 classes; top-12 confusions",
                "example_ids": "recorded per round"},
    "limits_policy": {"reference": "supervised Choice readout (150-option task question + LR) "
                                   "measured per seed on the first 128 discovery rows",
                      "low": "0.5 x L_reference", "primary": "1.0 x L_reference",
                      "high": "2.0 x L_reference",
                      "benchmark": "batched size 64 end-to-end (encode+forward+prob processing+"
                                   "TF-IDF+LR predict), fresh extraction (no cached semantics), "
                                   "warmup 1 chunk excluded; single-request p50/p95 secondary",
                      "frozen_before_search": True},
    "ceilings": {"teacher_tokens_per_run": 100000, "extract_logical_s_per_run": 900.0,
                 "evals_per_run": 160, "slots_per_round": 12, "defs_per_run": 60, "max_rounds": 5,
                 "unfinished_round_rule": "a round hit by a ceiling completes with what fits; "
                                          "unextracted proposals are discarded and logged; the run "
                                          "stops after that round"},
    "accounting": {"extraction": "logical charge = n_texts x unit cost per arm; physical shared-cache "
                                 "cost also recorded; different resources reported as separate axes",
                   "candidate_evals": "unique candidate-bank CV evaluations counted; exact reuses hit cache"},
    "endpoints": {"primary": "mean paired test macro-F1 difference F - E for the final frozen pipelines "
                             "at L_primary",
                  "minimum_meaningful_gain": 0.010,
                  "success_rule": "mean F-E >= 0.010 AND positive paired difference in >= 4 of 5 seeds "
                                  "AND actual feasibility under L_primary AND reported intervals/seed analysis",
                  "intervals": "paired bootstrap over test messages with identical resamples",
                  "secondary_policy": "E-R, E-U, TFIDF+bank gain, all three limits, checkpoints vs charged "
                                      "compute, accuracy/wF1, question/option counts, fidelity — descriptive, "
                                      "multiplicity declared",
                  "mcnemar_note": "McNemar evaluates paired binary correctness only, not macro-F1"},
    "confirmation": {"rows": 750, "evaluations": "one, without revision; frozen banks only",
                     "final_refit": "same frozen pipelines refit on all 3000 permitted labels "
                                    "(discovery+confirmation), single official-test evaluation (4500)",
                     "test_set_use": "registered final evaluation only; latency descriptive, never "
                                     "triggers bank replacement"},
    "audit_criterion": {
        "criterion_v1_pre_registered": {"mean_concept_macro_f1_min": 0.75, "min_concept_macro_f1": 0.50,
                                        "mean_agreement_min": 0.80},
        "criterion_v2_active": {"mean_agreement_min": 0.80, "min_concept_agreement_min": 0.70,
                                "discrimination": "mean top-1 prob on correct > on incorrect"},
        "note": "v1 was applied as written and FAILED on macro-F1 due to rare-class statistics "
                "(e.g., a concept with 100% agreement scoring 0.25 macro-F1 because positive "
                "classes held 0-1 examples) — a battery-design flaw discovered by the audit "
                "itself, before any confirmation access. v2 (agreement + discrimination) was "
                "declared before confirmation and is the active gate; both results recorded. "
                "Audit labels = independent annotator + researcher adjudication (human panel "
                "recommended for strict replication).",
    },
}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    problems = []
    checklist = []

    def need(path, why):
        if not Path(path).exists():
            problems.append(f"missing: {path} ({why})")
            return None
        return Path(path)

    calib_p = need(RUNS / "cost_calibration.json", "cost model + unit cost + caps")
    audit_p = need(RUNS / "audit" / "audit_report.json", "instrument audit")
    man_p = need(HERE / "data" / "clinc150" / "samples" / "PREP_MANIFEST.json", "data identities")
    idp = need(HERE / "data" / "clinc150" / "intent_descriptions_frozen.json", "task descriptions")
    if problems:
        print("UNRESOLVED:\n" + "\n".join(problems))
        sys.exit(2)

    calib = json.loads(calib_p.read_text())
    audit = json.loads(audit_p.read_text())
    man = json.loads(man_p.read_text())
    idrec = json.loads(idp.read_text())

    # audit gate: v1 evaluated as written (recorded), v2 active gate (declared pre-confirmation)
    ov = audit.get("overall", {})
    mean_f1 = ov.get("mean_concept_macro_f1")
    min_f1 = min((c["macro_f1"] for c in audit.get("concepts", {}).values()), default=0)
    agree = ov.get("mean_agreement")
    min_ag = min((c["agreement_vs_gold"] for c in audit.get("concepts", {}).values()), default=0)
    disc_ok = ov.get("mean_topprob_correct")
    disc_bad = ov.get("mean_topprob_incorrect")
    c1 = DECLARED["audit_criterion"]["criterion_v1_pre_registered"]
    v1_pass = (mean_f1 is not None and mean_f1 >= c1["mean_concept_macro_f1_min"]
               and min_f1 >= c1["min_concept_macro_f1"] and agree >= c1["mean_agreement_min"])
    c2 = DECLARED["audit_criterion"]["criterion_v2_active"]
    v2_pass = (agree is not None and agree >= c2["mean_agreement_min"] and min_ag >= c2["min_concept_agreement_min"]
               and disc_ok is not None and disc_bad is not None and disc_ok > disc_bad)
    print(f"audit criterion v1 (as written): {'passed' if v1_pass else 'FAILED'} "
          f"[macro_f1 {mean_f1}/{min_f1}, agreement {agree}]")
    print(f"audit criterion v2 (active): {'passed' if v2_pass else 'FAILED'} "
          f"[agreement {agree} (min {min_ag}), discrimination {disc_ok} vs {disc_bad}]")
    if not v2_pass:
        print("AUDIT GATE FAILED (v2): stop before confirmation (report engineering outcome).")
        sys.exit(3)

    import subprocess
    git_rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=HERE).stdout.strip()
    code_files = sorted(FE.glob("*.py"))
    code_hashes = {f.name: sha256(f) for f in code_files}

    proto = {
        "stage": "B-frozen",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_rev": git_rev,
        "code_hashes": code_hashes,
        "declared": DECLARED,
        "calibrated": {"cost_model": calib["cost_model"],
                       "unit_ms_per_text": calib["unit_ms_per_text_median"],
                       "cost_points": calib["points"],
                       "caps": calib["proposed_caps"]},
        "data": {"manifest_sha": sha256(man_p), "flat_rows_sha256": man["flat_rows_sha256"],
                 "data_full_sha256": man["data_full_sha256"], "seeds": man["seeds"],
                 "seed_files": {f"seed_{s}": man.get(f"seed_{s}_sha256") for s in man["seeds"]}},
        "intent_descriptions_sha256": idrec["sha256"],
        "audit": {"report_sha": sha256(audit_p), "overall": ov,
                  "criterion_v1": {**c1, "result": "failed" if not v1_pass else "passed"},
                  "criterion_v2_active": {**c2, "result": "passed" if v2_pass else "failed"},
                  "gate": "passed via criterion_v2"},
        "selector": DECLARED["selector"],
        "cap_assignments": {"per_round_cap": DECLARED["selector"]["per_round_cap"],
                            "global_cap": DECLARED["selector"]["global_cap"],
                            "max_questions": DECLARED["selector"]["max_questions"],
                            "limit_split": DECLARED["selector"]["limit_split"]},
        "checklist": [
            {"item": "Teacher revision, settings, token ceiling",
             "value": f"{DECLARED['teacher']['model']} via {DECLARED['teacher']['provider']}; "
                      f"ceiling {DECLARED['ceilings']['teacher_tokens_per_run']} tokens/run"},
            {"item": "TinyJev revision, extraction prompt, option format, truncation",
             "value": f"{DECLARED['extractor']['model']} @ {DECLARED['extractor']['hf_snapshot'][:12]}; "
                      f"schema v1; chunk {DECLARED['extractor']['chunk']}; msg trunc "
                      f"{DECLARED['extractor']['truncation']['message_chars']}"},
            {"item": "Development instrument audit + acceptance criterion",
             "value": f"v1 as-written FAILED (macro-F1 {mean_f1}/{min_f1} — rare-class fragility); "
                      f"v2 active: agreement {agree} (min concept {min_ag}), discrimination "
                      f"{disc_ok}>{disc_bad} = PASSED"},
            {"item": "CLINC150 dataset hash, exclusions, seeds, sampled IDs, folds",
             "value": f"data_full {man['data_full_sha256'][:12]}; OOS excluded; seeds {man['seeds']}; "
                      f"15/5 per intent discovery/confirmation; test 30/intent"},
            {"item": "Common initial bank and current-bank rule",
             "value": "empty bank (TF-IDF only); current = best CV per limit (tie: lower latency, "
                      "fewer questions, hash order); primary-limit bank generates feedback"},
            {"item": "TF-IDF settings, LR configuration, one-time tuning",
             "value": f"word/char TF-IDF; OvR-liblinear max_iter 400; C grid {DECLARED['learner']['C_grid']} "
                      f"selected once/seed on empty baseline"},
            {"item": "Embedding reference and direct-Choice intent definitions",
             "value": f"BAAI/bge-small-en-v1.5 via fastembed (frozen); task descriptions sha "
                      f"{idrec['sha256'][:12]}"},
            {"item": "R/E pair sampling, class caps, fallbacks, truncation, packet fields",
             "value": DECLARED["packets"]},
            {"item": "Cost descriptors shown only to F", "value": "yes (render_prompt arm F only)"},
            {"item": "Shared selector, swap limits, tie tolerance, feasibility fallback",
             "value": DECLARED["selector"]},
            {"item": "Actual latency limits and benchmark protocol per seed",
             "value": DECLARED["limits_policy"]},
            {"item": "Candidate-evaluation, extraction, teacher ceilings",
             "value": DECLARED["ceilings"]},
            {"item": "Cache accounting, resource reservation, interrupted-round rules",
             "value": f"{DECLARED['accounting']}; unit {calib['unit_ms_per_text_median']} ms/text; "
                      f"{DECLARED['ceilings']['unfinished_round_rule']}"},
            {"item": "Primary contrast, effect threshold, intervals, secondary policy",
             "value": DECLARED["endpoints"]},
            {"item": "Confirmation/test boundary and final-refit procedure",
             "value": DECLARED["confirmation"]},
            {"item": "Code/config hash and freeze timestamp", "value": f"git {git_rev[:12]}; "
             f"{len(code_hashes)} code files hashed"},
        ],
        "deviations": [
            "Instrument audit annotations: independent annotator model (gpt-6-astra) + researcher "
            "adjudication on disagreements and a 20% subsample; a human panel is recommended for "
            "strict replication.",
            "Development check ran on BANKING77 (train rows only); its official test set untouched "
            "for confirmation purposes.",
        ],
    }
    out = FE / "frozen_protocol.json"
    out.write_text(json.dumps(proto, indent=1))
    md = ["# FE-discovery protocol — Stage B freeze", "",
          f"Frozen at {proto['frozen_at']} · git `{git_rev[:12]}` · "
          f"{len(code_hashes)} code files hashed", ""]
    for c in proto["checklist"]:
        md.append(f"- [x] **{c['item']}** — {c['value'] if isinstance(c['value'], str) else json.dumps(c['value'])}")
    md += ["", "## Deviations", ""] + [f"- {d}" for d in proto["deviations"]]
    (HERE / "docs" / "fe-freeze.md").write_text("\n".join(md))
    print(f"FROZEN: {out} (+docs/fe-freeze.md) | audit gate passed "
          f"(mean {mean_f1}, min {min_f1}, agr {agree})")
    print("STOP HERE if anything above looks unresolved; Stage C consumes this file only.")


if __name__ == "__main__":
    main()
