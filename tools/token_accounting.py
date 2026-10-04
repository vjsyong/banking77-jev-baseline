#!/usr/bin/env python3
"""CPU-only token accounting for the BANKING77 extraction payload (no GPU, no weights).

For a sample of texts, computes the padded-position grid ("positions" ~ FLOPs) of:
  cur          tinyjev today: one plain-path call per record, all 17 rows padded
               to the widest row (the 77-option choice row). 17 x (P + choice).
  shared       same rows but state encoded once (the >=96-token shared path):
               P + 17 x max(row).
  split2       split by width class: choice row alone + compact noul rows
               (plain; prefix repeated per row): (P+choice) + 16 x (P+noul).
  split2_shared same split with prefix once: P + choice + 16 x noul.
  packed_ideal block-causal packed form (kev's), no padding: P + sum(rows).

Prints per-stat mean/median/p90 and the big ratios.
"""
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402
from banking77_baseline import question_set  # noqa: E402
from tinyjev.families import make as make_family  # noqa: E402

SNAP = Path("/home/xrim/.cache/huggingface/hub/models--AnkitAI--TinyJev-0.6B/"
            "snapshots/c559c2f7ea95069f92af8b999f512345bbb87d22")


def stats(name, xs):
    xs = sorted(xs)
    n = len(xs)
    p90 = xs[min(n - 1, int(0.9 * n))]
    print(f"{name:14s} mean {statistics.mean(xs):8.1f} | p50 {statistics.median(xs):8.1f} | "
          f"p90 {p90:8.1f} | max {max(xs):8.1f}")


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    manifest = json.loads((SNAP / "tinyjev.json").read_text())
    fam = make_family(manifest["family"], SNAP, manifest)

    d = load_dataset("PolyAI/banking77")
    labels = d["train"].features["label"].names
    qmap = question_set(labels, "both")
    questions = [dict(id=k, **v) for k, v in qmap.items()]
    for q in questions:
        if q["type"] == "noul":
            q["type"] = "boolean"

    P, choice, noul, cur, shared, split2, split2_shared, packed = [], [], [], [], [], [], [], []
    for i in range(N):
        text = d["train"][i]["text"]
        enc = fam.encode({"id": f"t{i}", "state": text, "questions": questions})
        pl = len(enc.prefix)
        Ls = [len(r) for r in enc.rows]
        ch, nu = Ls[0], max(Ls[1:])
        P.append(pl)
        choice.append(ch)
        noul.append(nu)
        cur.append(len(Ls) * (pl + max(Ls)))
        shared.append(pl + len(Ls) * max(Ls))
        split2.append((pl + ch) + (len(Ls) - 1) * (pl + nu))
        split2_shared.append(pl + ch + (len(Ls) - 1) * nu)
        packed.append(pl + sum(Ls))

    print(f"sample: {N} train texts, {len(questions)} questions "
          f"(1 choice x {len(labels)} options + {len(questions) - 1} noul)")
    print(f"prefix >= 96 tokens (would take tinyjev's shared path): "
          f"{sum(1 for p in P if p >= 96)}/{N}\n")
    stats("prefix P", P)
    stats("choice row", choice)
    stats("noul row max", noul)
    print()
    stats("cur grid", cur)
    stats("shared grid", shared)
    stats("split2 grid", split2)
    stats("split2_shared", split2_shared)
    stats("packed ideal", packed)
    print()
    print(f"ratio cur/split2        : {statistics.mean(cur) / statistics.mean(split2):.2f}x fewer positions")
    print(f"ratio cur/split2_shared : {statistics.mean(cur) / statistics.mean(split2_shared):.2f}x")
    print(f"ratio cur/packed_ideal  : {statistics.mean(cur) / statistics.mean(packed):.2f}x")


if __name__ == "__main__":
    main()
