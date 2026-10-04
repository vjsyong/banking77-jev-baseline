#!/usr/bin/env python3
"""Common cost-aware selector shared by all arms (brief §9).

- Immutable per-arm pool; candidate banks rebuilt from the full pool.
- Evaluation: frozen learner on all five discovery folds; every unique
  candidate-bank evaluation counts against the common cap (hits are reused).
- Nondominated quality-cost archive per latency limit; prior feasible banks
  retained; equal-quality cheaper banks preserved.
- Current bank per limit: greatest CV macro-F1; ties (within tol) -> lower
  latency, then fewer questions, then canonical hash order.
- Default selection: budgeted greedy forward selection, up to two
  single-question swaps per limit per round; engineering cap 12 questions.
- Screening uses the frozen cost model; retained archive points receive full
  measurements via measure_fn.
"""
from collections import OrderedDict
import fcntl
import threading
import time

import numpy as np

from learner import eval_bank


class BudgetExhausted(Exception):
    pass


def bank_hash_order(defs):
    from schema import canonical_hash
    return tuple(sorted(canonical_hash(d) for d in defs))


class _EvalGate:
    """Cross-process serialization for CV evaluations (CPU storms).

    flock on a shared file; RLock keeps in-process nesting safe. When
    eval_lock_path is None the gate is a no-op.
    """

    def __init__(self, path):
        self.path = path
        self._tlock = threading.RLock()
        self._depth = 0
        self._fd = None

    def __enter__(self):
        self._tlock.acquire()
        try:
            if self._depth == 0 and self.path is not None:
                self._fd = open(self.path, "a+")
                fcntl.flock(self._fd, fcntl.LOCK_EX)
            self._depth += 1
        except BaseException:
            self._tlock.release()
            raise
        return self

    def __exit__(self, *exc):
        try:
            self._depth -= 1
            if self._depth == 0 and self._fd is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                self._fd.close()
                self._fd = None
        finally:
            self._tlock.release()
        return False


class Selector:
    def __init__(self, tfidf, y, c_value, cost_model, limits, store,
                 per_round_cap=30, global_cap=200, max_questions=12,
                 tie_tol=0.0005, limit_split=(9, 12, 9), swap_attempts=2,
                 eval_lock_path=None):
        self.tfidf, self.y, self.C = tfidf, y, c_value
        self.cost_model, self.limits, self.store = cost_model, limits, store
        self.per_round_cap = per_round_cap
        self.global_cap = global_cap
        self.max_questions = max_questions
        self.tie_tol = tie_tol
        self.limit_split = limit_split  # (low, primary, high) eval shares
        self.swap_attempts = swap_attempts
        self.pool = []                  # list of defs (with "slot")
        self._pool_hashes = set()
        self._sem_cache = {}            # def_hash -> dense (n x w) block
        self._eval_cache = {}           # bank_key -> {"cv", "folds", "oof_pred", ...}
        self.unique_evals = 0
        self.cache_hits = 0
        self.round_evals = 0
        self.eval_log = []
        self.archive = {"low": [], "primary": [], "high": []}
        self.current = {"low": None, "primary": None, "high": None}
        self.round_no = 0
        self.measure_calls = 0
        self._gate = _EvalGate(eval_lock_path)

    # ---------- pool / features ----------
    def add_defs(self, defs):
        from schema import canonical_hash
        added = []
        for d in defs:
            h = canonical_hash(d)
            if h in self._pool_hashes:
                continue
            self._pool_hashes.add(h)
            self.pool.append(dict(d, idh=h))
            added.append(d)
        return added

    def def_block(self, d):
        from fe_extract import bank_matrix
        h = d["idh"] if "idh" in d else None
        if h is None:
            from schema import canonical_hash
            h = canonical_hash(d)
        if h not in self._sem_cache:
            from extract_probes import text_key
            rowkeys = [text_key(t) for t in self._texts]
            self._sem_cache[h] = bank_matrix(self.store, [dict(d, slot=d["slot"])], rowkeys)
        return self._sem_cache[h]

    def bank_matrix(self, defs):
        if not defs:
            return np.zeros((len(self._texts), 0))
        return np.hstack([self.def_block(d) for d in defs])

    def attach_texts(self, texts):
        self._texts = texts

    # ---------- evaluation with budget accounting ----------
    def bank_key(self, defs):
        return tuple(sorted(d["idh"] for d in defs))

    def eval(self, defs):
        key = self.bank_key(defs)
        if key in self._eval_cache:
            self.cache_hits += 1
            return self._eval_cache[key]
        if self.round_evals >= self.per_round_cap or self.unique_evals >= self.global_cap:
            raise BudgetExhausted()
        with self._gate:  # serialize CV fits across pool workers
            t0 = time.perf_counter()
            sem = self.bank_matrix(defs)
            r = eval_bank(self.tfidf, sem, self.y, self.C, want_oof=True)
            rec = {"cv": r["cv_macro_f1"], "folds": r["fold_scores"],
                   "oof_pred": r["oof_pred"], "oof_proba": r["oof_proba"],
                   "classes": r["classes"], "eval_s": round(time.perf_counter() - t0, 1)}
        self._eval_cache[key] = rec
        self.unique_evals += 1
        self.round_evals += 1
        return rec

    def cost_of(self, defs, limit_name=None):
        est = self.cost_model.predict_bank_ms(defs)
        return est

    def feasible(self, defs, limit_name):
        return self.cost_of(defs) <= self.limits[limit_name]["ms_per_text"]

    # ---------- comparison / archive ----------
    def _better(self, a, b):
        """a better than b? (cv desc, cost asc, questions asc, hash order)."""
        tol = self.tie_tol
        if a["cv"] > b["cv"] + tol:
            return True
        if abs(a["cv"] - b["cv"]) <= tol:
            if a["cost"] < b["cost"] - 1e-9:
                return True
            if abs(a["cost"] - b["cost"]) <= 1e-9:
                if a["nq"] < b["nq"]:
                    return True
                if a["nq"] == b["nq"]:
                    return a["hash_order"] < b["hash_order"]
        return False

    def _entry(self, defs, rec, measured_ms=None):
        return {"defs": [d["idh"] for d in defs], "names": [d["name"] for d in defs],
                "cv": rec["cv"], "cost": measured_ms if measured_ms is not None else self.cost_of(defs),
                "est_ms": self.cost_of(defs), "measured_ms": measured_ms,
                "nq": len(defs), "hash_order": bank_hash_order(defs),
                "measured": measured_ms is not None, "round": self.round_no}

    def _prune_archive(self, entries, cap=30):
        kept = []
        for e in entries:
            dominated = False
            for f in entries:
                if f is e:
                    continue
                if f["cv"] >= e["cv"] - self.tie_tol and f["cost"] <= e["cost"]:
                    if f["cv"] > e["cv"] + self.tie_tol or f["cost"] < e["cost"]:
                        dominated = True
                        break
            if not dominated:
                kept.append(e)
        # dedupe identical defs (keep measured/newest)
        seen, out = {}, []
        for e in sorted(kept, key=lambda x: (not x["measured"], -x["round"])):
            k = tuple(e["defs"])
            if k in seen:
                if not out[seen[k]]["measured"] and e["measured"]:
                    out[seen[k]] = e
                continue
            seen[k] = len(out)
            out.append(e)
        if len(out) > cap:
            out = sorted(out, key=lambda x: x["cost"])
            idxs = np.unique(np.linspace(0, len(out) - 1, cap).round().astype(int))
            out = [out[i] for i in idxs]
        return out

    # ---------- selection ----------
    def select_round(self, measure_fn=None):
        self.round_no += 1
        self.round_evals = 0
        log = {"round": self.round_no, "limits": {}}
        if self._eval_cache.get(self.bank_key([])) is None:
            self.eval([])  # empty-bank baseline (TF-IDF alone)
        for li, ln in enumerate(("low", "primary", "high")):
            share = self.limit_split[li]
            spent = 0
            cur = self.current[ln]
            bank = [next(d for d in self.pool if d["idh"] == h) for h in (cur["defs"] if cur else [])]
            cur_rec = self._eval_cache[self.bank_key(bank)]
            actions = []
            try:
                # forward phase (one improving add per scan window; repeat while improving)
                while len(bank) < self.max_questions and spent < share:
                    cands = [d for d in reversed(self.pool) if d["idh"] not in {x["idh"] for x in bank}]
                    window = cands[: max(1, share - spent)]
                    best, best_cv = None, cur_rec["cv"]
                    for d in window:
                        cand_bank = bank + [d]
                        if not self.feasible(cand_bank, ln):
                            actions.append({"action": "screen-skip", "add": d["name"]})
                            continue
                        r = self.eval(cand_bank)
                        spent += 1
                        actions.append({"action": "add-eval", "add": d["name"],
                                        "cv": round(r["cv"], 4), "eval_s": r.get("eval_s")})
                        if r["cv"] > best_cv + self.tie_tol:
                            best, best_cv = d, r["cv"]
                    if best is None:
                        break
                    bank = bank + [best]
                    cur_rec = self._eval_cache[self.bank_key(bank)]
                    actions.append({"action": "add-commit", "add": best["name"], "cv": round(cur_rec["cv"], 4)})
                # swaps (up to swap_attempts)
                for k in range(self.swap_attempts):
                    if spent >= share:
                        break
                    if not bank:
                        break
                    q = bank[-1]  # last committed question
                    tried = {a.get("swap-out") for a in actions if a["action"] == "swap-eval"}
                    if q["name"] in tried:
                        break
                    cands = [d for d in reversed(self.pool)
                             if d["idh"] not in {x["idh"] for x in bank}
                             and not any(a["action"] == "swap-eval" and a.get("swap-in") == d["name"] for a in actions)]
                    if not cands:
                        break
                    u = cands[0]
                    cand_bank = [x for x in bank if x["idh"] != q["idh"]] + [u]
                    if not self.feasible(cand_bank, ln):
                        continue
                    r = self.eval(cand_bank)
                    spent += 1
                    improved = r["cv"] > cur_rec["cv"] + self.tie_tol
                    actions.append({"action": "swap-eval", "swap-out": q["name"], "swap-in": u["name"],
                                    "cv": round(r["cv"], 4), "accepted": bool(improved)})
                    if improved:
                        bank, cur_rec = cand_bank, r
            except BudgetExhausted:
                actions.append({"action": "budget-stop"})
            # finalize limit
            entry = self._entry(bank, cur_rec)
            if cur is None or self._better(entry, cur):
                self.current[ln] = entry
            # archive: new entry + retained prior feasible banks (cost = measured if available else est)
            entries = self.archive[ln] + [entry]
            entries = [e for e in entries if e["cost"] <= self.limits[ln]["ms_per_text"]]
            self.archive[ln] = self._prune_archive(entries)
            # measure new archive points lacking measurement
            if measure_fn is not None:
                need = [e for e in self.archive[ln] if not e["measured"]]
                for e in need[:3]:
                    defs = [next(d for d in self.pool if d["idh"] == h) for h in e["defs"]]
                    ms = measure_fn(defs)
                    self.measure_calls += 1
                    e["measured_ms"] = ms
                    e["measured"] = True
                    e["cost"] = ms
                if need[:3]:
                    self.archive[ln] = self._prune_archive(self.archive[ln])
            log["limits"][ln] = {"actions": actions, "final": {k: entry[k] for k in
                                ("cv", "nq", "est_ms", "cost")},
                                 "share_used": spent}
        log["unique_evals"], log["cache_hits"] = self.unique_evals, self.cache_hits
        return log

    def primary_bank_defs(self):
        cur = self.current["primary"]
        if cur is None or not cur["defs"]:
            return []
        return [next(d for d in self.pool if d["idh"] == h) for h in cur["defs"]]

    def primary_oof(self):
        cur = self.current["primary"]
        if cur is None:
            return None
        return self._eval_cache[tuple(sorted(cur["defs"]))]
