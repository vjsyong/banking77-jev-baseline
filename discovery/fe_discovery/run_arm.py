#!/usr/bin/env python3
"""One discovery run (arm x seed): teacher -> proposals -> extraction -> selector.

Arm-agnostic to dataset (BANKING77 dev / CLINC150 confirmation). Everything is
logged: teacher prompts/responses + token usage, feedback example ids, proposal
rejections, extraction ledger (physical wall + logical charge), selector evals,
archives per limit, ceilings, stop reason. Unfinished-round rule (frozen): a
round hit by a ceiling completes with what fits; unextracted proposals are
discarded and logged; the run stops after that round.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path("/home/xrim/banking77-jev-baseline")
sys.path.insert(0, str(HERE / "discovery"))
sys.path.insert(0, str(HERE / "discovery" / "pilot"))
sys.path.insert(0, str(HERE / "discovery" / "fe_discovery"))

import numpy as np  # noqa: E402

for _p in ("/home/xrim/taildash/client", "/home/xrim/progtrack/client"):
    if Path(_p).is_dir():
        sys.path.insert(0, _p)
        break
try:
    from taildash import TaskMonitor as _TaskMonitor
except Exception:  # noqa: BLE001
    _TaskMonitor = None

from schema import canonical_hash, validate_proposal, DefinitionError, near_label_flags  # noqa: E402
import packets as pk  # noqa: E402
from teacher_client import teacher_chat_full  # noqa: E402

# frozen teacher config
TEACHER = {"provider": "lmuai-pro", "model": "gpt-6.1-sol", "temperature": 0.4,
           "max_tokens": 2200}
SLOTS_PER_ROUND = 12
DEFS_PER_RUN = 60
MAX_ROUNDS = 5


def parse_proposals(text: str):
    """Extract a JSON array of proposals from a teacher response."""
    s = text.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
    except Exception:  # noqa: BLE001
        pass
    m = re.search(r"\[", s)
    if not m:
        return None
    depth, end = 0, None
    for i in range(m.start(), len(s)):
        if s[i] == "[":
            depth += 1
        elif s[i] == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    try:
        obj = json.loads(s[m.start():end])
        return obj if isinstance(obj, list) else None
    except Exception:  # noqa: BLE001
        return None


class Ledger:
    """Extraction accounting: physical wall + logical charge (frozen unit cost)."""

    def __init__(self, unit_ms_per_text):
        self.unit_ms = unit_ms_per_text
        self.physical_s = 0.0
        self.logical_s = 0.0
        self.requests = 0
        self.hits = 0
        self.entries = []

    def record(self, slot, name, n_texts, wall_s, is_new):
        logical = n_texts * self.unit_ms / 1000.0
        self.logical_s += logical
        self.physical_s += wall_s if is_new else 0.0
        self.requests += 1
        self.hits += 0 if is_new else 1
        self.entries.append({"slot": slot, "name": name, "n_texts": n_texts,
                             "wall_s": round(wall_s, 2), "logical_s": round(logical, 2),
                             "new": bool(is_new)})

    def summary(self):
        return {"physical_extract_s": round(self.physical_s, 1),
                "logical_charge_s": round(self.logical_s, 1),
                "extract_requests": self.requests, "cache_hits": self.hits}


class Runner:
    def __init__(self, arm, seed, dataset, extractor, store, tfidf, selector,
                 out_dir, task_name, intent_labels, ceilings, unit_ms_per_text,
                 label_of_row, texts_cal, dev_notes=""):
        assert arm in ("U", "R", "E", "F")
        self.arm, self.seed, self.ds = arm, seed, dataset
        self.ex, self.store, self.tfidf = extractor, store, tfidf
        self.sel = selector
        self.out = Path(out_dir)
        (self.out / "teacher_prompts").mkdir(parents=True, exist_ok=True)
        (self.out / "teacher_responses").mkdir(parents=True, exist_ok=True)
        self.task_name = task_name
        self.intent_labels = intent_labels
        self.ceilings = ceilings  # {"teacher_tokens": N, "evals": N, "extract_logical_s": N}
        self.ledger = Ledger(unit_ms_per_text)
        self.teacher_calls = 0
        self.teacher_tokens = {"in": 0, "out": 0}
        self.memory_names = []
        self.rounds_log = []
        self.stop_reason = "max_rounds"
        self.dev_notes = dev_notes
        self.texts_cal = texts_cal
        self.label_of_row = label_of_row
        self.mon = (_TaskMonitor(server=os.environ.get("TAILDASH_URL", "http://localhost:8080"),
                                 title=f"FE discovery: {arm} seed {seed} [{task_name}]",
                                 total=MAX_ROUNDS + 1, agent_name="fe-discovery")
                    if _TaskMonitor else None)

    def _mon(self):
        class _N:
            def log(self, *a, **k): pass
            def update(self, *a, **k): pass
            def complete(self, *a, **k): pass
            def fail(self, *a, **k): pass
        return self.mon or _N()

    # -------- helpers --------
    def _measure_fn(self, defs):
        """Full-bank measurement for retained archive points (batched protocol).

        Fitting the deployment pipeline uses cached (store) features on discovery
        rows; the latency run itself re-extracts fresh on calibration texts
        (no cached semantic outputs in the benchmark).
        """
        from costs import build_pipeline, measure_batched
        from fe_extract import bank_matrix
        from extract_probes import text_key
        rk = [text_key(t) for t in self.ds["texts_d"]]
        block = bank_matrix(self.store, defs, rk)
        pipe = build_pipeline(self.ds["texts_d"], self.ds["y_d"], block, c_value=self.sel.C)
        out = measure_batched(self.ex, pipe, defs, self.texts_cal["texts"])
        return out["ms_per_text"]

    def _extract_new(self, defs, round_no):
        """Extract defs on discovery rows; respect logical ceiling; ledger."""
        from fe_extract import bank_matrix  # noqa: F401
        from schema import canonical_hash
        kept, skipped = [], []
        for d in defs:
            if self.ledger.logical_s >= self.ceilings["extract_logical_s"]:
                skipped.append({"name": d["name"], "reason": "extract_logical_ceiling"})
                continue
            slot = d["slot"]
            wall, n_new = self.ex.extract_choice_fe(slot, d["question"], d["options"],
                                                    self.ds["texts_d"])
            is_new = n_new > 0
            self.ledger.record(slot, d["name"], len(self.ds["texts_d"]), wall, is_new)
            kept.append(d)
        return kept, skipped

    def _teacher_round(self, round_no, packet_attrs):
        """Build packet+prompt, call teacher, parse+validate proposals."""
        bank_defs = self.sel.primary_bank_defs()
        oof = self.sel.primary_oof()
        y_d = self.ds["y_d"]
        classes = np.array(sorted(set(y_d)))
        if oof is None:
            # no current bank yet: empty bank OOF == TF-IDF-only model
            oof = self.sel.eval([])
        packet, meta = pk.build_packet(
            self.arm, self.seed, round_no, self.ds["texts_d"], y_d,
            oof["oof_pred"], oof["oof_proba"], classes, oof["cv"],
            limits=packet_attrs.get("limits"), cost_info=packet_attrs.get("cost_info"),
            archive=packet_attrs.get("archive"))
        prompt = pk.render_prompt(self.arm, bank_defs, self.memory_names, packet,
                                  pk.task_blurb(), slots=SLOTS_PER_ROUND)
        (self.out / "teacher_prompts" / f"r{round_no}.txt").write_text(prompt)
        resp = teacher_chat_full(
            [{"role": "user", "content": prompt}],
            model=TEACHER["model"], provider=TEACHER["provider"],
            temperature=TEACHER["temperature"], seed=self.seed * 1000 + round_no,
            max_tokens=TEACHER["max_tokens"])
        self.teacher_calls += 1
        u = resp.get("usage") or {}
        self.teacher_tokens["in"] += int(u.get("prompt_tokens") or 0)
        self.teacher_tokens["out"] += int(u.get("completion_tokens") or 0)
        (self.out / "teacher_responses" / f"r{round_no}.txt").write_text(resp["text"])
        raw = parse_proposals(resp["text"])
        parsed, rejected = [], []
        if raw is None:
            rejected.append({"name": None, "reason": "no_json_array_in_response"})
        else:
            seen_new = set()
            for item in raw[: SLOTS_PER_ROUND * 2]:
                try:
                    defn, notes = validate_proposal(item)
                except DefinitionError as e:
                    rejected.append({"name": item.get("name") if isinstance(item, dict) else None,
                                     "reason": str(e)})
                    continue
                h = canonical_hash(defn)
                if h in seen_new or h in self.sel._pool_hashes:
                    rejected.append({"name": defn["name"], "reason": "duplicate"})
                    continue
                seen_new.add(h)
                flags = near_label_flags(defn, self.intent_labels)
                parsed.append(dict(defn, slot=f"fe::{h[:20]}", notes=notes,
                                   near_label=flags))
        parsed = parsed[:SLOTS_PER_ROUND]
        meta["n_valid"], meta["n_rejected"] = len(parsed), len(rejected)
        meta["rejected"] = rejected[:30]
        meta["usage"] = u
        meta["elapsed_s"] = round(resp.get("elapsed_s", 0), 1)
        return parsed, rejected, meta

    # -------- main loop --------
    def run(self):
        t0 = time.perf_counter()
        mon = self._mon()
        mon.log(f"start {self.arm} s{self.seed} on {self.task_name} "
                f"(rounds<={MAX_ROUNDS}, slots<={SLOTS_PER_ROUND})")
        try:
            for round_no in range(1, MAX_ROUNDS + 1):
                if self.teacher_tokens["in"] + self.teacher_tokens["out"] >= self.ceilings["teacher_tokens"]:
                    self.stop_reason = "teacher_token_ceiling"
                    break
                if self.sel.unique_evals >= self.ceilings["evals"]:
                    self.stop_reason = "eval_ceiling"
                    break
                if len(self.sel.pool) >= DEFS_PER_RUN:
                    self.stop_reason = "defs_ceiling"
                    break
                # packet attributes for F
                packet_attrs = {}
                if self.arm == "F":
                    cur = self.sel.current["primary"]
                    limits_ms = {k: v["ms_per_text"] for k, v in self.sel.limits.items()}
                    per_q = [{"name": d["name"], "n_options": len(d["options"]),
                              "est_ms": round(self.sel.cost_model.predict_bank_ms([d]), 2)}
                             for d in self.sel.primary_bank_defs()]
                    arch = []
                    for ln in ("low", "primary", "high"):
                        for e in self.sel.archive[ln][:4]:
                            arch.append(f"[{ln}] {e['names']} cv={e['cv']:.4f} "
                                        f"cost={'%.2f' % e['cost']}ms nq={e['nq']}")
                    packet_attrs = {"limits": limits_ms, "per_question": per_q,
                                    "archive": arch,
                                    "cost_info": {"current_latency_ms": (cur["cost"] if cur else None),
                                                  "n_questions": (cur["nq"] if cur else 0),
                                                  "n_columns": sum(len(d["options"]) for d in self.sel.primary_bank_defs()),
                                                  "per_question": per_q}}
                proposals, rejected, tmeta = self._teacher_round(round_no, packet_attrs)
                for d in proposals:
                    self.memory_names.append(d["name"])
                kept, skipped = self._extract_new(proposals, round_no)
                added = self.sel.add_defs(kept)
                sel_log = self.sel.select_round(measure_fn=self._measure_fn)
                cur = self.sel.current["primary"]
                entry = {"round": round_no, "teacher": tmeta,
                         "proposed": len(proposals), "extracted": len(kept),
                         "unextracted": skipped, "pool_size": len(self.sel.pool),
                         "selector": sel_log,
                         "current_primary": {"names": cur["names"], "cv": cur["cv"],
                                             "cost": cur["cost"], "nq": cur["nq"]} if cur else None,
                         "ledger": self.ledger.summary(),
                         "teacher_tokens": dict(self.teacher_tokens)}
                self.rounds_log.append(entry)
                (self.out / "rounds.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in self.rounds_log))
                print(f"[{self.arm} s{self.seed} r{round_no}] prop {len(proposals)} ext {len(kept)} "
                      f"pool {len(self.sel.pool)} | primary cv "
                      f"{cur['cv'] if cur else float('nan'):.4f} nq {cur['nq'] if cur else 0} | "
                      f"evals {self.sel.unique_evals} | tok {self.teacher_tokens['in']+self.teacher_tokens['out']}",
                      flush=True)
                mon.update(round_no,
                           message=(f"r{round_no}: pool {len(self.sel.pool)}, primary cv "
                                    f"{(cur['cv'] if cur else float('nan')):.4f}, "
                                    f"nq {(cur['nq'] if cur else 0)}, "
                                    f"evals {self.sel.unique_evals}"))
        except BaseException as exc:  # noqa: BLE001
            mon.fail(f"{type(exc).__name__}: {exc}")
            raise
        final = {
            "arm": self.arm, "seed": self.seed, "task": self.task_name,
            "stop_reason": self.stop_reason,
            "rounds": len(self.rounds_log),
            "teacher_calls": self.teacher_calls,
            "teacher_tokens": dict(self.teacher_tokens),
            "teacher_config": TEACHER,
            "ledger": self.ledger.summary(),
            "unique_evals": self.sel.unique_evals, "cache_hits": self.sel.cache_hits,
            "measure_calls": self.sel.measure_calls,
            "pool": [{"name": d["name"], "idh": d["idh"], "question": d["question"],
                      "options": d["options"], "slot": d["slot"],
                      "notes": d.get("notes"), "near_label": d.get("near_label")}
                     for d in self.sel.pool],
            "current": {ln: self.sel.current[ln] for ln in ("low", "primary", "high")},
            "archive": {ln: [{k: e[k] for k in ("defs", "names", "cv", "cost", "est_ms",
                                                "measured_ms", "nq", "round")}
                             for e in self.sel.archive[ln]]
                        for ln in ("low", "primary", "high")},
            "wall_s": round(time.perf_counter() - t0, 1),
            "dev_notes": self.dev_notes,
            "integrity": {"teacher_revision": "gpt-6.1-sol via lmuai-pro (settings frozen)",
                          "slots_per_round": SLOTS_PER_ROUND, "defs_cap": DEFS_PER_RUN,
                          "max_rounds": MAX_ROUNDS},
        }
        (self.out / "rounds.jsonl").write_text(
            "\n".join(json.dumps(r) for r in self.rounds_log))
        (self.out / "final.json").write_text(json.dumps(final, indent=1))
        cur = self.sel.current["primary"]
        mon.complete(f"stop={self.stop_reason}; rounds={len(self.rounds_log)}; "
                     f"primary cv {(cur['cv'] if cur else float('nan')):.4f} "
                     f"nq {(cur['nq'] if cur else 0)}; evals {self.sel.unique_evals}")
        return final
