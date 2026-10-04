#!/usr/bin/env python3
"""Controlled discovery pilot: does frontier feedback change what the teacher discovers?

Three policies x three seeds x three rounds, same budget everywhere:
  instruments : short-Noul probes, TinyJev-0.6B, batched extraction (held constant)
  teacher     : gpt-5.6-sol, temperature per seed (held constant otherwise)
  learner panel: LR / LinearSVC / ExtraTrees on probe scores (constant)
  selection   : accept a candidate bank iff best-panel 5-fold CV macro-F1 strictly
                improves on the same fixed folds (constant)
  budget      : 3 rounds, <=4 proposals/round (constant)
All search runs inside 1,000 training rows; test split untouched until final eval.

Usage:
  python run_pilot.py --policies unguided,accuracy,frontier --seeds 0.3,0.7,1.0 --rounds 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "5")
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "discovery"))
sys.path.insert(0, str(HERE / "discovery" / "pilot"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from extract_probes import ProbeExtractor, ProbeStore  # noqa: E402
from teacher_client import teacher_chat  # noqa: E402

RUNS = HERE / "runs" / "banking77"
PILOT = RUNS / "discovery" / "pilot"
PILOT.mkdir(parents=True, exist_ok=True)
SEED = 77
MAX_PROPOSALS = 4
EXTRACT_COST_MS = 2.5  # measured: ms per message per probe (batched, short-Noul)


# ---------- panel + scoring ----------
def make_panel():
    return {
        "lr": Pipeline([("sc", StandardScaler()),
                        ("m", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"))]),
        "svm": Pipeline([("sc", StandardScaler()), ("m", LinearSVC(C=1.0))]),
        "et": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                   max_features="sqrt", n_jobs=-1, random_state=77),
    }


def bank_cv(X, y, folds):
    scores = {}
    for name, model in make_panel().items():
        vals = []
        for tr, va in folds:
            model.fit(X[tr], y[tr])
            vals.append(f1_score(y[va], model.predict(X[va]), average="macro", zero_division=0))
        scores[name] = float(np.mean(vals))
    best = max(scores.values())
    return scores, best


def diagnostics(X, y, folds, cv_scores):
    best_name = max(cv_scores, key=cv_scores.get)
    model = make_panel()[best_name]
    oof = cross_val_predict(model, X, y, cv=[(tr, va) for tr, va in folds])
    per_class = {}
    for cls in sorted(set(y)):
        mask = y == cls
        per_class[cls] = float(f1_score(y[mask], oof[mask], average="macro", zero_division=0))
    worst = sorted(per_class.items(), key=lambda kv: kv[1])[:12]
    pairs = Counter((t, p) for t, p in zip(y, oof) if t != p).most_common(12)
    lines = ["Per-class F1 (worst 12): " + ", ".join(f"{c} {s:.2f}" for c, s in worst)]
    lines.append("Top confusions: " + "; ".join(f"{t}->{p} x{n}" for (t, p), n in pairs))
    lines.append("Learner CV macro-F1: " + ", ".join(f"{k} {v:.3f}" for k, v in cv_scores.items())
                 + f" | best {max(cv_scores.values()):.3f}")
    return "\n".join(lines), worst, pairs


# ---------- proposals ----------
def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")
    return s[:40] or "probe"


def norm_q(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def parse_proposals(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return [], ""
    obj = json.loads(m.group(0))
    props = obj.get("proposals") or []
    out = []
    for p in props[:MAX_PROPOSALS]:
        action = p.get("action")
        if action not in ("add", "revise", "delete"):
            continue
        q = (p.get("question") or "").strip()
        out.append({"action": action, "name": slug(p.get("name") or p.get("target") or "probe"),
                    "question": q, "target": (p.get("target") or "").strip(),
                    "rationale": (p.get("rationale") or "")[:160]})
    return out, (obj.get("summary") or "")[:240]


def bank_question_norms(bank):
    return {norm_q(p["question"]) for p in bank}


def apply_proposals(bank, proposals, round_no):
    """Returns (candidate_bank, applied_records, notes)."""
    cand = [dict(p) for p in bank]
    applied, notes = [], []
    ids = {p["id"] for p in cand}
    for pr in proposals:
        a, q, name = pr["action"], pr["question"], pr["name"]
        if a == "add":
            if not q:
                notes.append("add skipped (empty question)")
                continue
            if norm_q(q) in bank_question_norms(cand):
                notes.append(f"add skipped (duplicate): {q[:60]}")
                continue
            pid = name if name not in ids else f"{name}_{round_no}"
            cand.append({"id": pid, "question": q})
            ids.add(pid)
            applied.append({**pr, "id": pid})
        elif a == "revise":
            tgt = next((p for p in cand if p["id"] == pr["target"]
                        or norm_q(p["question"]) == norm_q(pr["target"])), None)
            if tgt is None or not q:
                notes.append(f"revise skipped (target not found): {pr['target'][:40]}")
                continue
            cand.remove(tgt)
            old = tgt["id"]
            pid = f"{old}__r{round_no}"
            cand.append({"id": pid, "question": q})
            ids.add(pid)
            applied.append({**pr, "id": pid, "replaced": old})
        else:  # delete
            tgt = next((p for p in cand if p["id"] == pr["target"]
                        or norm_q(p["question"]) == norm_q(pr["target"])), None)
            if tgt is None:
                notes.append(f"delete skipped (target not found): {pr['target'][:40]}")
                continue
            cand.remove(tgt)
            applied.append({**pr, "id": tgt["id"]})
    return cand, applied, notes


# ---------- teacher prompts ----------
BASE = """You are designing compact semantic probes for a bank-intent classifier.
TASK: classify a customer message into one of 77 BANKING77 intents. A small decision
model (TinyJev-0.6B, a pointer model trained for typed decisions - NOT a generative
LLM) answers SHORT semantic probes about each message and returns a probability in
[0,1]. A classical classifier (LogisticRegression / LinearSVC / ExtraTrees) is
trained on the probe scores to predict the intent.

PROBE RULES (fixed): a single-concept yes/no question about THIS message, answerable
from the message text alone; short (<= 20 words); starts with "Is"/"Does"; no overlap
with existing probes; measures one distinction that helps separate intents.

CURRENT PROBE BANK ({n} probes):
{bank}

{feedback}

Return STRICT JSON only: {{"proposals": [{{"action": "add"|"revise"|"delete",
"name": "<new probe id, snake_case>", "question": "<the probe question>",
"target": "<existing probe id, for revise/delete>", "rationale": "<=25 words"}}],
"summary": "<=40 words"}}
At most 4 proposals. Prefer additions (and targeted revisions/deletions) that
separate the confusable intents.{round_line}"""

FEEDBACK = {
    "unguided": ("Round {r} of iterative refinement. No performance diagnostics are "
                 "provided in this arm: improve the bank using your own judgment about "
                 "what bank customer messages need to be distinguished."),
    "accuracy": ("Round {r}. Use these training-fold diagnostics of the current bank's "
                 "classifier to propose probes that separate the confusable intents.\n"
                 "DIAGNOSTICS:\n{diag}"),
    "frontier": ("Round {r}. Use these training-fold diagnostics AND the cost model to "
                 "advance the quality/cost frontier (higher CV macro-F1 per unit of "
                 "extraction cost). Thinning redundant probes (revise/delete) is welcome "
                 "when it preserves quality.\nDIAGNOSTICS:\n{diag}\n"
                 "COST MODEL: extraction ~{ms} ms per message per probe (measured, "
                 "batched, RTX 3090); classifier fit negligible.\nARCHIVE:\n{archive}"),
}


def build_prompt(policy, rnd, bank, diag, archive_lines):
    bank_txt = "\n".join(f"- {p['id']}: {p['question']}" for p in bank)
    fb = FEEDBACK[policy].format(r=rnd, diag=diag, ms=EXTRACT_COST_MS,
                                 archive="\n".join(archive_lines) or "(round 1)")
    return BASE.format(n=len(bank), bank=bank_txt, feedback=fb,
                       round_line=f"\nThis is round {rnd} of 3.")


# ---------- main pilot ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="unguided,accuracy,frontier")
    ap.add_argument("--seeds", default="0.3,0.7,1.0")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    policies = args.policies.split(",")
    seeds = [float(s) for s in args.seeds.split(",")]

    idx = pd.read_csv(RUNS / "discovery" / "smallregime_index.csv")["csv_index"].to_numpy()
    pred = pd.read_csv(RUNS / "jev_predictions.csv").iloc[idx].reset_index(drop=True)
    texts = pred["text"].tolist()
    y = pred["label"].to_numpy()
    print(f"pilot regime: {len(texts)} train rows", flush=True)

    b0 = json.loads((HERE / "probes.json").read_text())
    b0 = [{"id": p["id"], "question": p["question"]} for p in b0]

    store = ProbeStore(PILOT / "probe_scores.sqlite")
    ex = ProbeExtractor(store)
    t0 = time.perf_counter()
    for p in b0:
        ex.extract_probe(p["id"], p["question"], texts)
    print(f"B0 extracted in {time.perf_counter() - t0:.1f}s", flush=True)

    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                 .split(np.zeros(len(y)), y))

    archive_path = PILOT / f"archive_{args.tag}.jsonl"
    summary_path = PILOT / f"summary_{args.tag}.json"

    def cv_of(bank):
        ids = [p["id"] for p in bank]
        X = ex.matrix(ids, texts)
        return bank_cv(X, y, folds)

    t_score = 0.0
    t0 = time.perf_counter()
    _, b0_best = cv_of(b0)
    t_score += time.perf_counter() - t0
    print(f"B0 CV best: {b0_best:.4f}", flush=True)

    runs_summary = []
    with archive_path.open("w") as af:
        for policy in policies:
            for seed in seeds:
                bank = [dict(p) for p in b0]
                cur = b0_best
                archive_lines = [f"round 0: {len(bank)} probes, cv {cur:.3f}, "
                                 f"cum cost {len(bank) * len(texts) * EXTRACT_COST_MS / 1000:.0f}s"]
                cum_cost_ms = len(bank) * len(texts) * EXTRACT_COST_MS
                t_teacher = t_add = 0.0
                run_trace = []
                for rnd in range(1, args.rounds + 1):
                    ids = [p["id"] for p in bank]
                    X = ex.matrix(ids, texts)
                    t0 = time.perf_counter()
                    cv_scores, _ = bank_cv(X, y, folds)
                    diag, _, _ = diagnostics(X, y, folds, cv_scores)
                    t_score += time.perf_counter() - t0

                    prompt = build_prompt(policy, rnd, bank, diag, archive_lines)
                    t0 = time.perf_counter()
                    try:
                        raw = teacher_chat([{"role": "user", "content": prompt}],
                                           temperature=seed, max_tokens=1600)
                    except Exception as exc:  # noqa: BLE001
                        raw = ""
                        print(f"TEACHER FAIL {policy} s{seed} r{rnd}: {exc}", flush=True)
                    t_call = time.perf_counter() - t0
                    t_teacher += t_call
                    proposals, tsummary = parse_proposals(raw)
                    (PILOT / f"teacher_{args.tag}_{policy}_s{seed}_r{rnd}.txt").write_text(
                        raw if isinstance(raw, str) else str(raw))

                    cand, applied, notes = apply_proposals(bank, proposals, rnd)
                    cur_before, size_before = cur, len(bank)
                    # extract new/changed probes on the 1k (cost accrues regardless of acceptance)
                    added_ids = [a["id"] for a in applied if a["action"] in ("add", "revise")]
                    t_run_add = 0.0
                    for a in applied:
                        if a["action"] in ("add", "revise"):
                            q = next(p["question"] for p in cand if p["id"] == a["id"])
                            t_run_add += ex.extract_probe(a["id"], q, texts)
                    t_add += t_run_add
                    cost_add_ms = len(added_ids) * len(texts) * EXTRACT_COST_MS
                    cum_cost_ms += cost_add_ms

                    accepted, cand_best = False, None
                    if cand != bank:
                        Xc = ex.matrix([p["id"] for p in cand], texts)
                        t0 = time.perf_counter()
                        _, cand_best = bank_cv(Xc, y, folds)
                        t_score += time.perf_counter() - t0
                        accepted = cand_best > cur_before + 1e-9
                        if accepted:
                            bank, cur = cand, cand_best
                    archive_lines.append(
                        f"round {rnd}: {len(bank)} probes, cv {cur:.3f}, "
                        f"cum cost {cum_cost_ms / 1000:.0f}s "
                        f"({'accepted' if accepted else 'rejected' if cand_best is not None else 'no change'})")

                    rec = {
                        "policy": policy, "seed": seed, "round": rnd,
                        "teacher_wall_s": round(t_call, 1),
                        "proposals": proposals, "applied": applied, "notes": notes,
                        "teacher_summary": tsummary,
                        "bank_size_before": size_before, "bank_size_after": len(bank),
                        "cv_before": round(cur_before, 4),
                        "cand_cv": None if cand_best is None else round(cand_best, 4),
                        "accepted": accepted,
                        "extract_added_s": round(t_run_add, 2),
                        "cum_cost_ms": round(cum_cost_ms, 1),
                    }
                    af.write(json.dumps(rec) + "\n")
                    af.flush()
                    print(f"{policy} s{seed} r{rnd}: {len(proposals)} proposals, "
                          f"{len(applied)} applied, cand_cv={cand_best if cand_best is None else round(cand_best, 4)}, "
                          f"accepted={accepted}, bank={len(bank)}", flush=True)
                    run_trace.append(rec)
                runs_summary.append({
                    "policy": policy, "seed": seed,
                    "final_bank": bank, "final_cv_best": round(cur, 4),
                    "bank_size": len(bank),
                    "t_teacher": round(t_teacher, 1), "t_extract_1k": round(t_add, 1),
                    "n_rounds": args.rounds, "trace": run_trace,
                })
                print(f"== {policy} seed {seed}: final bank {len(bank)} probes, "
                      f"cv {cur:.4f}", flush=True)

    store.close()
    summary = {
        "tag": args.tag, "policies": policies, "seeds": seeds, "rounds": args.rounds,
        "b0_size": len(b0), "b0_cv_best": round(b0_best, 4),
        "t_score_total_s": round(t_score, 1),
        "runs": runs_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print("written", summary_path, flush=True)


if __name__ == "__main__":
    main()
