#!/usr/bin/env python3
"""Generic probe extractor for the discovery pilot (batched, cached).

Extracts arbitrary short-Noul probes on a set of texts with TinyJev-0.6B and
caches every (probe_id, text_sha) score in a sqlite store, so rounds and reruns
never re-extract. Probe definitions are stored for provenance; a probe whose
question changes gets a NEW id (`<id>__rN`) to keep past scores valid.

CLI (smoke): python extract_probes.py --probes probes.json --texts texts.json
  probes.json: [{"id": "...", "question": "..."}]
  texts.json:  [{"key": "t123", "text": "..."}]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "5")
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(5)

from tinyjev import load as load_agent  # noqa: E402
from tinyjev.families import softmax  # noqa: E402


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class ProbeStore:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS scores (
            probe_id TEXT NOT NULL, text_sha TEXT NOT NULL, score REAL NOT NULL,
            PRIMARY KEY (probe_id, text_sha))""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS probe_defs (
            probe_id TEXT PRIMARY KEY, question TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS choice_scores (
            probe_id TEXT NOT NULL, text_sha TEXT NOT NULL, opt TEXT NOT NULL,
            score REAL NOT NULL, PRIMARY KEY (probe_id, text_sha, opt))""")
        self.db.commit()

    def define(self, probe_id: str, question: str):
        self.db.execute("INSERT OR REPLACE INTO probe_defs VALUES (?, ?, CURRENT_TIMESTAMP)",
                        (probe_id, question))
        self.db.commit()

    def have(self, probe_id: str, tsha: str) -> bool:
        return self.db.execute("SELECT 1 FROM scores WHERE probe_id=? AND text_sha=?",
                               (probe_id, tsha)).fetchone() is not None

    def put(self, probe_id: str, tsha: str, score: float):
        self.db.execute("INSERT OR REPLACE INTO scores VALUES (?, ?, ?)",
                        (probe_id, tsha, float(score)))
        self.db.commit()

    def score_of(self, probe_id: str, tsha: str) -> float | None:
        row = self.db.execute("SELECT score FROM scores WHERE probe_id=? AND text_sha=?",
                              (probe_id, tsha)).fetchone()
        return None if row is None else float(row[0])

    def have_choice(self, probe_id: str, tsha: str, options: list[str]) -> bool:
        n = self.db.execute("SELECT COUNT(*) FROM choice_scores WHERE probe_id=? AND text_sha=?",
                            (probe_id, tsha)).fetchone()[0]
        return n == len(options)

    def put_choice(self, probe_id: str, tsha: str, opt: str, score: float):
        self.db.execute("INSERT OR REPLACE INTO choice_scores VALUES (?, ?, ?, ?)",
                        (probe_id, tsha, opt, float(score)))
        self.db.commit()

    def choice_score_of(self, probe_id: str, tsha: str, opt: str) -> float | None:
        row = self.db.execute("SELECT score FROM choice_scores WHERE probe_id=? AND text_sha=? AND opt=?",
                              (probe_id, tsha, opt)).fetchone()
        return None if row is None else float(row[0])

    def def_question(self, probe_id: str) -> str | None:
        row = self.db.execute("SELECT question FROM probe_defs WHERE probe_id=?",
                              (probe_id,)).fetchone()
        return None if row is None else row[0]

    def purge(self, probe_id: str):
        """Drop all rows + definition for a probe id (used when its definition
        changed and stale rows must not be served from cache)."""
        self.db.execute("DELETE FROM scores WHERE probe_id=?", (probe_id,))
        self.db.execute("DELETE FROM choice_scores WHERE probe_id=?", (probe_id,))
        self.db.execute("DELETE FROM probe_defs WHERE probe_id=?", (probe_id,))
        self.db.commit()

    def close(self):
        self.db.close()


class ProbeExtractor:
    def __init__(self, store: ProbeStore, device: str = "cuda"):
        self.store = store
        agent = load_agent("TinyJev-0.6B", backend="torch", device=device)
        self.fam = agent.family
        self.model = agent.backbone.model
        self.dev = agent.backbone.device

    def extract_probe(self, probe_id: str, question: str, texts: list[str],
                      chunk: int = 64, verbose: bool = False) -> float:
        """Extract one Noul probe over texts. Returns wall seconds spent.

        Definition-aware: if the stored definition for this id differs (same slug,
        different question), stale rows are purged and everything is re-extracted.
        """
        stored = self.store.def_question(probe_id)
        if stored is not None and stored != question:
            self.store.purge(probe_id)
        self.store.define(probe_id, question)
        have = {row[0] for row in self.store.db.execute(
            "SELECT text_sha FROM scores WHERE probe_id=?", (probe_id,))}
        missing = [t for t in texts if text_key(t) not in have]
        if not missing:
            return 0.0
        t0 = time.perf_counter()
        q = {"id": "q", "type": "boolean", "instructions": question}
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
                self.store.put(probe_id, text_key(part[j]), float(probs[1]))
            if verbose:
                print(f"  {probe_id}: {min(s + chunk, len(missing))}/{len(missing)}", flush=True)
        return time.perf_counter() - t0

    def extract_choice(self, probe_id: str, question: str, options: list[str],
                       texts: list[str], chunk: int = 64, verbose: bool = False) -> float:
        """Extract one categorical Choice probe (2..3 short option labels).

        Option labels are the criteria keys, so the returned distribution is keyed
        by them; scores are stored per (probe, text, option). Returns wall seconds.
        """
        assert 2 <= len(options) <= 3, "choice probes need 2..3 options"
        want = question + " || options: " + " | ".join(options)
        stored = self.store.def_question(probe_id)
        if stored is not None and stored != want:
            self.store.purge(probe_id)
        self.store.define(probe_id, want)
        rows: dict = {}
        for tsha, opt in self.store.db.execute(
                "SELECT text_sha, opt FROM choice_scores WHERE probe_id=?", (probe_id,)):
            rows.setdefault(tsha, set()).add(opt)
        want_set = set(options)
        missing = [t for t in texts if rows.get(text_key(t)) != want_set]
        if not missing:
            return 0.0
        t0 = time.perf_counter()
        q = {"id": "q", "type": "choice", "instructions": question,
             "criteria": {o: None for o in options}}
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
                keys = e.questions[0]["keys"]
                for o in options:
                    self.store.put_choice(probe_id, text_key(part[j]), o,
                                          float(probs[keys.index(o)]))
            if verbose:
                print(f"  {probe_id}: {min(s + chunk, len(missing))}/{len(missing)}", flush=True)
        return time.perf_counter() - t0

    def matrix_mixed(self, probe_defs: list[dict], texts: list[str]):
        """Feature matrix for a mixed bank (Noul scalars + Choice option distributions).

        probe_defs: [{"id":.., "format": "noul"|"choice", "options": [...]}, ...]
        (options present for choice probes). Returns (X, colnames); raises if the
        store is missing any required score.
        """
        cols, data = [], []
        for p in probe_defs:
            if p.get("format") == "choice":
                for o in p.get("options") or []:
                    cols.append(f"{p['id']}::{o}")
                    data.append([self.store.choice_score_of(p["id"], text_key(t), o) for t in texts])
            else:
                cols.append(p["id"])
                data.append([self.store.score_of(p["id"], text_key(t)) for t in texts])
        X = np.array(data, dtype=np.float64).T
        if np.isnan(X).any():
            raise RuntimeError("missing probe scores in store (run extraction first)")
        return X, cols

    def matrix(self, probe_ids: list[str], texts: list[str]) -> np.ndarray:
        """(n_texts, n_probes) score matrix from cache; raises if incomplete."""
        out = np.empty((len(texts), len(probe_ids)), dtype=np.float64)
        for j, pid in enumerate(probe_ids):
            for i, t in enumerate(texts):
                v = self.store.score_of(pid, text_key(t))
                if v is None:
                    raise RuntimeError(f"missing score: {pid} / {text_key(t)} (run extract first)")
                out[i, j] = v
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--store", default=str(HERE / "runs" / "banking77" / "discovery" / "pilot" / "probe_scores.sqlite"))
    args = ap.parse_args()
    probes = json.loads(Path(args.probes).read_text())
    texts = [r["text"] for r in json.loads(Path(args.texts).read_text())]
    store = ProbeStore(args.store)
    ex = ProbeExtractor(store)
    for p in probes:
        dt = ex.extract_probe(p["id"], p["question"], texts)
        print(f"{p['id']}: {dt:.2f}s")
    store.close()


if __name__ == "__main__":
    main()
