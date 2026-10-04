#!/usr/bin/env python3
"""Extraction engine for the FE-discovery experiment (CLINC150).

Content-addressed slots (fe::<canonical-hash-head>), full option distributions
retained, option definitions wired through the pointer family's criteria
({id: definition} -> rendered "id: description"), invalid outputs raise.
Extraction ledger records wall time per slot so arms can be charged their
logical standalone cost even when caches are shared.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
HERE = Path("/home/xrim/banking77-jev-baseline")
sys.path.insert(0, str(HERE / "discovery" / "pilot"))
from extract_probes import ProbeStore, text_key  # noqa: E402
from tinyjev import load as load_agent  # noqa: E402
from tinyjev.families import softmax  # noqa: E402

MAX_OPTIONS = 255  # channel limit (pointer family); probe schema validation caps at 6


class ExtractionError(RuntimeError):
    pass


class FEExtractor:
    def __init__(self, store: ProbeStore, device: str = "cuda"):
        self.store = store
        agent = load_agent("TinyJev-0.6B", backend="torch", device=device)
        self.fam = agent.family
        self.model = agent.backbone.model
        self.dev = agent.backbone.device

    def extract_choice_fe(self, slot: str, question: str, options: list[dict],
                          texts: list[str], chunk: int = 64, verbose: bool = False):
        """Extract one categorical question (3-6 options with definitions).

        Returns (wall_seconds, n_extracted_new). Validates outputs; a definition
        change purges the slot (content-addressed ids make this near-impossible).
        """
        assert len(options) <= MAX_OPTIONS, "too many options"
        criteria = {o["id"]: o["definition"] for o in options}
        want = question + " || " + json.dumps(criteria, ensure_ascii=True,
                                              separators=(",", ":"), sort_keys=True)
        stored = self.store.def_question(slot)
        if stored is not None and stored != want:
            self.store.purge(slot)
        self.store.define(slot, want)
        have = {}
        for tsha, opt in self.store.db.execute(
                "SELECT text_sha, opt FROM choice_scores WHERE probe_id=?", (slot,)):
            have.setdefault(tsha, set()).add(opt)
        want_set = set(criteria)
        missing = [t for t in texts if have.get(text_key(t)) != want_set]
        if not missing:
            return 0.0, 0
        t0 = time.perf_counter()
        q = {"id": "q", "type": "choice", "instructions": question, "criteria": criteria}
        for s in range(0, len(missing), chunk):
            part = missing[s:s + chunk]
            encs = [self.fam.encode({"id": f"x{i}", "state": t, "questions": [dict(q)]})
                    for i, t in enumerate(part)]
            rows = [e.prefix + e.rows[0] for e in encs]
            width = max(len(r) for r in rows)
            ids = torch.full((len(rows), width), self.fam.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(rows), width), dtype=torch.long)
            for i, r in enumerate(rows):
                ids[i, :len(r)] = torch.tensor(r)
                att[i, :len(r)] = 1
            with torch.inference_mode():
                hs = self.model(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev),
                                use_cache=False).last_hidden_state.float().cpu().numpy()
            for j, e in enumerate(encs):
                h = hs[j, :len(rows[j])]
                z = self.fam.logits([h], e)
                probs = softmax(z["q"])
                if not np.isfinite(probs).all() or float(probs.sum()) <= 0 or len(probs) != len(criteria):
                    raise ExtractionError(f"{slot}: invalid distribution "
                                          f"(len {len(probs)} vs {len(criteria)}, finite={np.isfinite(probs).all()})")
                keys = e.questions[0]["keys"]
                for oid in criteria:
                    self.store.put_choice(slot, text_key(part[j]), oid,
                                          float(probs[keys.index(oid)]))
            if verbose:
                print(f"  {slot}: {min(s + chunk, len(missing))}/{len(missing)}", flush=True)
        return time.perf_counter() - t0, len(missing)


def bank_matrix(store, defs: list[dict], row_keys: list[str]) -> np.ndarray:
    """Dense log(clip(p)) block over a bank's columns; raises if incomplete."""
    cols = []
    for d in defs:
        for o in d["options"]:
            col = np.empty(len(row_keys), dtype=np.float64)
            for i, k in enumerate(row_keys):
                v = store.choice_score_of(d["slot"], k, o["id"])
                if v is None or not np.isfinite(v):
                    raise ExtractionError(f"missing/invalid score {d['slot']}/{o['id']}/{k}")
                col[i] = np.log(min(max(v, 1e-6), 1.0))
            cols.append(col)
    return np.array(cols, dtype=np.float64).T
