#!/usr/bin/env python3
"""2x2 factorial on probe discovery: representation x selection.

Constant across all four conditions: teacher gpt-6.1-sol (via lmuai-pro),
accuracy-level diagnostics briefing, learner panel, fixed folds, 1k-label
regime, <=4 proposals/round, 3 rounds/seed, test split untouched.

  representation : noul  = yes/no probes only
                   mixed = yes/no OR categorical Choice (2-3 options; the full
                           option distribution is retained as features)
  selection      : strict = accept candidate iff best-panel CV strictly improves
                   pareto = nondominated (size, score) archive; also accepts
                            equal-quality with fewer probes; explicit prune pass
                            each round (single-fold LR screen, tolerance 0.01)

Usage:
  python run_factorial.py [--cells all|noul-strict,...] [--seeds 0.3,0.7,1.0]
                          [--rounds 3] [--tag full]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "5")
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "discovery"))
sys.path.insert(0, str(HERE / "discovery" / "pilot"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from extract_probes import ProbeExtractor, ProbeStore  # noqa: E402
from run_pilot import bank_cv, diagnostics, make_panel  # noqa: E402
from teacher_client import teacher_chat  # noqa: E402

# taildash monitoring (best-effort; the fallback keeps the script independent)
for _p in ("/home/xrim/taildash/client", "/home/xrim/progtrack/client"):
    if Path(_p).is_dir():
        sys.path.insert(0, _p)
        break
try:
    from taildash import TaskMonitor as _TaskMonitor
except Exception:  # noqa: BLE001
    _TaskMonitor = None


class _NullMonitor:
    def log(self, *a, **k):
        pass

    def update(self, *a, **k):
        pass

    def complete(self, *a, **k):
        pass

    def fail(self, *a, **k):
        pass

RUNS = HERE / "runs" / "banking77"
OUT = RUNS / "discovery" / "factorial"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 77
MAX_PROPOSALS = 4
EXTRACT_COST_MS = 2.5  # ms per message per probe (batched, measured)
PRUNE_TOL = 0.01

CELLS = {
    "noul-strict": {"repr": "noul", "sel": "strict"},
    "noul-pareto": {"repr": "noul", "sel": "pareto"},
    "mixed-strict": {"repr": "mixed", "sel": "strict"},
    "mixed-pareto": {"repr": "mixed", "sel": "pareto"},
}

BASE = """You are designing compact semantic probes for a bank-intent classifier.
TASK: classify a customer message into one of 77 BANKING77 intents. A small decision
model (TinyJev-0.6B, a pointer model trained for typed decisions - NOT a generative
LLM) answers SHORT semantic probes about each message and returns a probability
distribution over the probe's options. A classical classifier (LogisticRegression /
LinearSVC / ExtraTrees) is trained on the probe outputs to predict the intent.

{format_rules}

CURRENT PROBE BANK ({n} probes):
{bank}

Round {r}. Use these training-fold diagnostics of the current bank's classifier to
propose probes that separate the confusable intents.
DIAGNOSTICS:
{diag}

Return STRICT JSON only: {{"proposals": [{{"action": "add"|"revise"|"delete",
"format": "noul"|"choice", "name": "<new probe id, snake_case>",
"question": "<the probe question>", "options": ["<label>", "<label>"],
"target": "<existing probe id, for revise/delete>", "rationale": "<=25 words"}}],
"summary": "<=40 words"}}
At most 4 proposals. Prefer additions (and targeted revisions/deletions) that
separate the confusable intents."""

FORMAT_RULES = {
    "noul": ("PROBE FORMAT (fixed for this condition): yes/no Noul questions only "
             "(format \"noul\"). Each probe is a single-concept question, <= 20 words, "
             "answerable from the message text alone."),
    "mixed": ("PROBE FORMAT (this condition): you MAY propose either (a) yes/no Noul "
              "questions (format \"noul\") or (b) compact categorical Choice questions "
              "(format \"choice\") with 2-3 short, mutually exclusive option labels in "
              "\"options\": [...]. The full option distribution is retained as features. "
              "Questions must be answerable from the message text alone."),
}


def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")
    return s[:40] or "probe"


def norm_q(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def bank_txt(bank) -> str:
    lines = []
    for p in bank:
        fmt = p.get("format", "noul")
        suffix = f" [choice: {' | '.join(p.get('options') or [])}]" if fmt == "choice" else ""
        lines.append(f"- {p['id']}{suffix}: {p['question']}")
    return "\n".join(lines)


def parse_proposals(text: str, allow_choice: bool):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return [], "", ["parse: no JSON found"]
    try:
        obj = json.loads(m.group(0))
    except Exception as exc:  # noqa: BLE001
        return [], "", [f"parse: {exc}"]
    out, notes = [], []
    for p in (obj.get("proposals") or [])[:MAX_PROPOSALS]:
        action = p.get("action")
        if action not in ("add", "revise", "delete"):
            notes.append(f"skip (bad action {action!r})")
            continue
        fmt = (p.get("format") or "noul").strip().lower()
        if fmt not in ("noul", "choice"):
            notes.append(f"skip (bad format {fmt!r})")
            continue
        if fmt == "choice":
            if not allow_choice:
                notes.append(f"skip (choice not permitted in this condition): {str(p.get('name'))[:30]}")
                continue
            opts = [str(o).strip() for o in (p.get("options") or [])]
            if not (2 <= len(opts) <= 3) or any(not o or len(o) > 60 for o in opts):
                notes.append(f"skip (bad options): {str(p.get('name'))[:30]}")
                continue
        rec = {"action": action, "format": fmt, "name": slug(p.get("name") or p.get("target") or "probe"),
               "question": (p.get("question") or "").strip(),
               "options": [str(o).strip() for o in (p.get("options") or [])] if fmt == "choice" else None,
               "target": (p.get("target") or "").strip(),
               "rationale": (p.get("rationale") or "")[:160]}
        out.append(rec)
    return out, (obj.get("summary") or "")[:240], notes


def apply_proposals(bank, proposals, round_no):
    cand = [dict(p) for p in bank]
    applied, notes = [], []
    ids = {p["id"] for p in cand}
    for pr in proposals:
        a, q, name = pr["action"], pr["question"], pr["name"]
        if a == "add":
            if not q:
                notes.append("add skipped (empty question)")
                continue
            if norm_q(q) in {norm_q(p["question"]) for p in cand}:
                notes.append(f"add skipped (duplicate): {q[:60]}")
                continue
            pid = name if name not in ids else f"{name}_{round_no}"
            rec = {"id": pid, "question": q, "format": pr["format"]}
            if pr["format"] == "choice":
                rec["options"] = pr["options"]
            cand.append(rec)
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
            rec = {"id": pid, "question": q, "format": pr["format"]}
            if pr["format"] == "choice":
                rec["options"] = pr["options"]
            cand.append(rec)
            ids.add(pid)
            applied.append({**pr, "id": pid, "replaced": old})
        else:
            tgt = next((p for p in cand if p["id"] == pr["target"]
                        or norm_q(p["question"]) == norm_q(pr["target"])), None)
            if tgt is None:
                notes.append(f"delete skipped (target not found): {pr['target'][:40]}")
                continue
            cand.remove(tgt)
            applied.append({**pr, "id": tgt["id"]})
    return cand, applied, notes


def lr_screen(X, y, fold) -> float:
    m = Pipeline([("sc", StandardScaler()),
                  ("m", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"))])
    m.fit(X[fold[0]], y[fold[0]])
    return float(f1_score(y[fold[1]], m.predict(X[fold[1]]), average="macro", zero_division=0))


def prune_pass(bank, X, y, folds, cur):
    """Single sequential sweep. A removal is kept if the single-fold LR screen
    does not drop below the pre-removal screen minus PRUNE_TOL. Returns
    (bank, X, cur_after_full_rescore, pruned_ids, n_screen_evals)."""
    fold = folds[0]
    cur_screen = lr_screen(X, y, fold)
    pruned, evals = [], 0
    i = 0
    while i < len(bank):
        if len(bank) <= 8:
            break
        Xc = np.delete(X, i, axis=1)
        s = lr_screen(Xc, y, fold)
        evals += 1
        if s >= cur_screen - PRUNE_TOL:
            pruned.append(bank[i]["id"])
            bank = bank[:i] + bank[i + 1:]
            X = Xc
            cur_screen = s
            continue
        i += 1
    if pruned:
        _, cur_new = bank_cv(X, y, folds)
        cur = cur_new
    return bank, X, cur, pruned, evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="all")
    ap.add_argument("--seeds", default="0.3,0.7,1.0")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    cells = list(CELLS) if args.cells == "all" else args.cells.split(",")
    seeds = [float(s) for s in args.seeds.split(",")]

    mon = (_TaskMonitor(server="http://localhost:8080",
                        title=f"banking77 factorial [{args.tag}] {len(cells)} cells x {len(seeds)} seeds",
                        total=len(cells) * len(seeds), agent_name="banking77-factorial")
           if _TaskMonitor else _NullMonitor())
    mon.log(f"registered: cells={cells} seeds={seeds} rounds={args.rounds} regime=1k rows")

    idx = pd.read_csv(RUNS / "discovery" / "smallregime_index.csv")["csv_index"].to_numpy()
    pred = pd.read_csv(RUNS / "jev_predictions.csv").iloc[idx].reset_index(drop=True)
    texts, y = pred["text"].tolist(), pred["label"].to_numpy()
    print(f"factorial regime: {len(texts)} train rows | cells: {cells} | seeds: {seeds}", flush=True)

    b0 = json.loads((HERE / "probes.json").read_text())
    B0 = [{"id": p["id"], "question": p["question"], "format": "noul"} for p in b0]

    store = ProbeStore(RUNS / "discovery" / "pilot" / "probe_scores.sqlite")
    ex = ProbeExtractor(store)
    t0 = time.perf_counter()
    for p in B0:
        ex.extract_probe(p["id"], p["question"], texts)
    print(f"B0 ready in {time.perf_counter() - t0:.1f}s", flush=True)

    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                 .split(np.zeros(len(y)), y))
    X0, _ = ex.matrix_mixed(B0, texts)
    cv_scores, b0_best = bank_cv(X0, y, folds)
    print(f"B0 CV best: {b0_best:.4f}", flush=True)

    archive_path = OUT / f"archive_{args.tag}.jsonl"
    summary_path = OUT / f"summary_{args.tag}.json"
    done_runs = 0
    runs_summary = []
    with archive_path.open("w") as af:
        for cell in cells:
            cfg_cell = CELLS[cell]
            for seed in seeds:
                bank = [dict(p) for p in B0]
                X = X0.copy()
                cur = b0_best
                t_teacher = t_add = t_score = 0.0
                calls = 0
                nd_archive = [{"size": len(bank), "score": cur, "label": "round0"}]
                run_trace = []
                for rnd in range(1, args.rounds + 1):
                    cur_before = cur
                    diag, _, _ = diagnostics(X, y, folds, cv_scores)
                    prompt = BASE.format(
                        format_rules=FORMAT_RULES[cfg_cell["repr"]],
                        n=len(bank), bank=bank_txt(bank), r=rnd, diag=diag)
                    t0 = time.perf_counter()
                    try:
                        raw = teacher_chat([{"role": "user", "content": prompt}],
                                           temperature=seed, max_tokens=1600)
                        calls += 1
                    except Exception as exc:  # noqa: BLE001
                        raw = ""
                        print(f"TEACHER FAIL {cell} s{seed} r{rnd}: {exc}", flush=True)
                    t_call = time.perf_counter() - t0
                    t_teacher += t_call
                    (OUT / f"teacher_{args.tag}_{cell}_s{seed}_r{rnd}.txt").write_text(
                        raw if isinstance(raw, str) else str(raw))

                    proposals, tsum, pnotes = parse_proposals(raw, cfg_cell["repr"] == "mixed")
                    cand, applied, notes = apply_proposals(bank, proposals, rnd)
                    size_before = len(bank)
                    t_run_add = 0.0
                    for a in applied:
                        if a["action"] in ("add", "revise"):
                            defn = next(p for p in cand if p["id"] == a["id"])
                            if defn["format"] == "choice":
                                t_run_add += ex.extract_choice(defn["id"], defn["question"],
                                                               defn["options"], texts)
                            else:
                                t_run_add += ex.extract_probe(defn["id"], defn["question"], texts)
                    t_add += t_run_add

                    accepted, cand_best = False, None
                    if cand != bank:
                        Xc, _ = ex.matrix_mixed(cand, texts)
                        t0 = time.perf_counter()
                        cv_scores_c, cand_best = bank_cv(Xc, y, folds)
                        t_score += time.perf_counter() - t0
                        if cfg_cell["sel"] == "strict":
                            accepted = cand_best > cur + 1e-9
                        else:
                            accepted = (cand_best > cur + 1e-9) or (
                                cand_best >= cur - 1e-9 and len(cand) < size_before)
                        nd_archive.append({"size": len(cand), "score": round(cand_best, 4),
                                           "label": f"r{rnd}cand"})
                        if accepted:
                            bank, X, cur, cv_scores = cand, Xc, cand_best, cv_scores_c

                    pruned, prune_evals = [], 0
                    if cfg_cell["sel"] == "pareto":
                        t0 = time.perf_counter()
                        bank, X, cur, pruned, prune_evals = prune_pass(bank, X, y, folds, cur)
                        t_score += time.perf_counter() - t0
                        if pruned:
                            _, cv_scores = bank_cv(X, y, folds)
                        nd_archive.append({"size": len(bank), "score": round(cur, 4),
                                           "label": f"r{rnd}postprune"})

                    rec = {
                        "cell": cell, "repr": cfg_cell["repr"], "sel": cfg_cell["sel"],
                        "seed": seed, "round": rnd, "teacher_wall_s": round(t_call, 1),
                        "proposals": proposals, "applied": applied,
                        "notes": pnotes + notes, "teacher_summary": tsum,
                        "bank_size_before": size_before, "bank_size_after": len(bank),
                        "cv_before": round(cur_before, 4),
                        "cand_cv": None if cand_best is None else round(cand_best, 4),
                        "accepted": accepted,
                        "pruned": pruned, "prune_screen_evals": prune_evals,
                        "extract_added_s": round(t_run_add, 2),
                    }
                    af.write(json.dumps(rec) + "\n")
                    af.flush()
                    print(f"{cell} s{seed} r{rnd}: {len(proposals)} proposals, {len(applied)} applied, "
                          f"cand_cv={rec['cand_cv']}, accepted={accepted}, pruned={len(pruned)}, "
                          f"bank={len(bank)}, cv={cur:.4f}", flush=True)
                    mon.log(f"{cell} s{seed} r{rnd}: applied {len(applied)}, "
                            f"cand_cv={rec['cand_cv']}, accepted={accepted}, "
                            f"pruned={len(pruned)}, bank={len(bank)}, cv={cur:.4f}")
                    run_trace.append(rec)

                # nondominated set for the archive
                nd = []
                for a in nd_archive:
                    dominated = any(b["size"] <= a["size"] and b["score"] >= a["score"] - 1e-12
                                    and (b["size"], b["score"]) != (a["size"], a["score"])
                                    for b in nd_archive)
                    if not dominated:
                        nd.append(a)
                runs_summary.append({
                    "cell": cell, "repr": cfg_cell["repr"], "sel": cfg_cell["sel"], "seed": seed,
                    "final_bank": bank, "final_cv_best": round(cur, 4), "bank_size": len(bank),
                    "n_choices": sum(1 for p in bank if p.get("format") == "choice"),
                    "t_teacher": round(t_teacher, 1), "t_extract_1k": round(t_add, 1),
                    "teacher_calls": calls, "archive": nd, "trace": run_trace,
                })
                print(f"== {cell} seed {seed}: bank {len(bank)} (choices "
                      f"{sum(1 for p in bank if p.get('format') == 'choice')}), cv {cur:.4f}, "
                      f"pruned total {sum(len(r['pruned']) for r in run_trace)}", flush=True)
                done_runs += 1
                mon.update(done_runs, message=f"{cell} seed {seed}: bank {len(bank)}, cv {cur:.4f}")

    store.close()
    summary = {
        "tag": args.tag, "cells": cells, "seeds": seeds, "rounds": args.rounds,
        "b0_cv_best": round(b0_best, 4), "runs": runs_summary,
        "constants": {"teacher": "gpt-6.1-sol (lmuai-pro)", "briefing": "accuracy diagnostics",
                      "panel": "LR/LinearSVC/ExtraTrees", "folds": "5-fold StratifiedKFold seed 77",
                      "budget": f"{args.rounds} rounds x <={MAX_PROPOSALS} proposals, seeds {seeds}"},
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print("written", summary_path, flush=True)
    mon.complete(f"{done_runs} runs done")


if __name__ == "__main__":
    main()
