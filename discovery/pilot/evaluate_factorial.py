#!/usr/bin/env python3
"""Final test-split evaluation for the factorial + paired effect analysis.

Wording-scoped verification: probes whose (id, wording/options) varies across
runs are extracted under per-wording evaluation slots (`ev::<id>::wK`) so every
run is scored with the exact definitions its bank used. Probes with a single
wording keep their shared slot (cache hit, no extra cost).

Extracts all needed slots on the untouched test split (3,080) and the 1k train
split, scores each run (best-CV panel model on the 1k), recomputes CV
wording-scoped (reported alongside the run-time recorded value), and computes
the paired effects: representation (mixed - noul) and selection (pareto - strict).
Registered with taildash before running.
"""
import collections
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "5")
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "discovery"))
sys.path.insert(0, str(HERE / "discovery" / "pilot"))

for _p in ("/home/xrim/taildash/client", "/home/xrim/progtrack/client"):
    if Path(_p).is_dir():
        sys.path.insert(0, _p)
        break
try:
    from taildash import TaskMonitor as _TaskMonitor
except Exception:  # noqa: BLE001
    _TaskMonitor = None

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from extract_probes import ProbeExtractor, ProbeStore, text_key  # noqa: E402
from run_pilot import bank_cv, make_panel  # noqa: E402

RUNS = HERE / "runs" / "banking77"
FACT = RUNS / "discovery" / "factorial"

_ACTIVE_MONITOR = None


class _NullMonitor:
    def log(self, *a, **k):
        pass

    def update(self, *a, **k):
        pass

    def complete(self, *a, **k):
        pass

    def fail(self, *a, **k):
        pass


def def_key(p) -> str:
    if p.get("format") == "choice":
        return p["question"] + " || " + " | ".join(p.get("options") or [])
    return p["question"]


def main():
    global _ACTIVE_MONITOR
    summary = json.loads((FACT / "summary_full.json").read_text())
    runs = summary["runs"]

    # ---- wording-scoped slot resolution ----
    by_id = collections.defaultdict(list)
    for run in runs:
        for p in run["final_bank"]:
            by_id[p["id"]].append(p)
    slot_map = {}
    colliding = {}
    for pid, plist in by_id.items():
        keys = sorted({def_key(p) for p in plist})
        if len(keys) <= 1:
            continue
        colliding[pid] = keys
        for k, key in enumerate(keys):
            slot_map[(pid, key)] = f"ev::{pid}::w{k}"

    def slot_of(p):
        return slot_map.get((p["id"], def_key(p)), p["id"])

    plan = {}
    for run in runs:
        for p in run["final_bank"]:
            plan[slot_of(p)] = p

    mon = (_TaskMonitor(server="http://localhost:8080",
                        title=f"banking77 factorial eval ({len(runs)} banks, wording-scoped)",
                        total=len(runs) + 1, agent_name="banking77-factorial-eval")
           if _TaskMonitor else _NullMonitor())
    _ACTIVE_MONITOR = mon
    mon.log(f"eval: {len(runs)} runs | {len(by_id)} ids | {len(colliding)} colliding | "
            f"{len(plan)} slots to verify")

    idx = pd.read_csv(RUNS / "discovery" / "smallregime_index.csv")["csv_index"].to_numpy()
    pred = pd.read_csv(RUNS / "jev_predictions.csv")
    train = pred.iloc[idx].reset_index(drop=True)
    test = pred[pred["split"] == "test"].reset_index(drop=True)
    texts_1k, y = train["text"].tolist(), train["label"].to_numpy()
    texts_te, yte = test["text"].tolist(), test["label"].to_numpy()
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
                 .split(np.zeros(len(y)), y))

    store = ProbeStore(RUNS / "discovery" / "pilot" / "probe_scores.sqlite")
    ex = ProbeExtractor(store)

    slot_ms = {}
    t0 = time.perf_counter()
    for i, (slot, p) in enumerate(plan.items(), 1):
        t_s = time.perf_counter()
        if p.get("format") == "choice":
            ex.extract_choice(slot, p["question"], p["options"], texts_1k)
            ex.extract_choice(slot, p["question"], p["options"], texts_te)
        else:
            ex.extract_probe(slot, p["question"], texts_1k)
            ex.extract_probe(slot, p["question"], texts_te)
        slot_ms[slot] = round(1000 * (time.perf_counter() - t_s) / len(texts_te), 3)
        if i % 10 == 0 or i == len(plan):
            print(f"  slot {i}/{len(plan)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"extraction done in {time.perf_counter()-t0:.0f}s", flush=True)

    def matrix(pairs, texts):
        data = []
        for slot, p in pairs:
            if p.get("format") == "choice":
                for o in p.get("options") or []:
                    data.append([store.choice_score_of(slot, text_key(t), o) for t in texts])
            else:
                data.append([store.score_of(slot, text_key(t)) for t in texts])
        return np.array(data, dtype=np.float64).T

    results = []
    for i, run in enumerate(runs, 1):
        bank = run["final_bank"]
        pairs = [(slot_of(p), p) for p in bank]
        X1k = matrix(pairs, texts_1k)
        Xte = matrix(pairs, texts_te)
        cv_scores, cv_best = bank_cv(X1k, y, folds)
        best_name = max(cv_scores, key=cv_scores.get)
        per_model = {}
        for name, m in make_panel().items():
            m.fit(X1k, y)
            p = m.predict(Xte)
            per_model[name] = round(float(f1_score(yte, p, average="macro", zero_division=0)), 4)
        model = make_panel()[best_name]
        model.fit(X1k, y)
        preds = model.predict(Xte)
        n_coll = sum(1 for p in bank if (p["id"], def_key(p)) in slot_map)
        r = {"cell": run["cell"], "repr": run["repr"], "sel": run["sel"], "seed": run["seed"],
             "bank_size": run["bank_size"], "n_choices": run["n_choices"],
             "cv_recorded": run["final_cv_best"],
             "cv_recomputed_wording_scoped": round(float(cv_best), 4),
             "cv_delta": round(float(cv_best) - run["final_cv_best"], 4),
             "best_model": best_name,
             "test_macro_f1": round(float(f1_score(yte, preds, average="macro", zero_division=0)), 4),
             "test_accuracy": round(float(accuracy_score(yte, preds)), 4),
             "test_per_model": per_model,
             "n_colliding_probes_in_bank": n_coll,
             "extract_ms_per_text": round(sum(slot_ms[slot_of(p)] for p in bank), 3)}
        results.append(r)
        print(f"{r['cell']:13s} s{r['seed']}: bank {r['bank_size']:2d}, test {r['test_macro_f1']:.4f} "
              f"(cv {r['cv_recomputed_wording_scoped']:.4f} vs recorded {r['cv_recorded']:.4f}, "
              f"Δ {r['cv_delta']:+.4f})", flush=True)
        mon.update(i, message=f"{r['cell']} s{r['seed']}: test {r['test_macro_f1']:.4f}")

    def get(repr_, sel, seed):
        return next(r for r in results if r["repr"] == repr_ and r["sel"] == sel and r["seed"] == seed)

    seeds = sorted({r["seed"] for r in results})
    effects = {"repr_mixed_minus_noul": {}, "sel_pareto_minus_strict": {}}
    for sel in ("strict", "pareto"):
        diffs = [get("mixed", sel, s)["test_macro_f1"] - get("noul", sel, s)["test_macro_f1"]
                 for s in seeds]
        effects["repr_mixed_minus_noul"][sel] = [round(d * 100, 2) for d in diffs]
        print(f"repr effect (mixed-noul, {sel}): {[round(d*100,2) for d in diffs]} pp")
    for repr_ in ("noul", "mixed"):
        diffs = [get(repr_, "pareto", s)["test_macro_f1"] - get(repr_, "strict", s)["test_macro_f1"]
                 for s in seeds]
        effects["sel_pareto_minus_strict"][repr_] = [round(d * 100, 2) for d in diffs]
        print(f"sel effect (pareto-strict, {repr_}): {[round(d*100,2) for d in diffs]} pp")

    cell_means = {}
    for cell in sorted({r["cell"] for r in results}):
        sub = [r for r in results if r["cell"] == cell]
        cell_means[cell] = {
            "mean_test_macro_f1": round(float(np.mean([r["test_macro_f1"] for r in sub])), 4),
            "mean_bank_size": round(float(np.mean([r["bank_size"] for r in sub])), 2),
            "mean_choices": round(float(np.mean([r["n_choices"] for r in sub])), 2),
            "mean_extract_ms": round(float(np.mean([r["extract_ms_per_text"] for r in sub])), 2),
        }
    out = {"runs": results, "paired_effects_pp": effects, "cell_means": cell_means,
           "wording_scope": {"n_ids": len(by_id), "n_colliding_ids": len(colliding),
                             "colliding_ids": sorted(colliding)[:60],
                             "n_slots_extracted": len(plan)},
           "refs": json.loads((RUNS / "discovery" / "smallregime_refs.json").read_text())}
    (FACT / "test_eval_factorial.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(cell_means, indent=2))
    print("written", FACT / "test_eval_factorial.json")
    mon.complete(f"eval done: {len(results)} banks, {len(plan)} slots")
    store.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        if _ACTIVE_MONITOR is not None:
            _ACTIVE_MONITOR.fail(f"{type(exc).__name__}: {exc}")
        raise
