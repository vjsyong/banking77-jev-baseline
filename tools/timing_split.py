#!/usr/bin/env python3
"""Separate direct-only vs probe-only vs combined latency (batched infra, 512 texts).

Measures the cost split of the question types in one chunked batched pass per
configuration: choice-only (77-option row), probes-only (16 noul rows), combined
(production layout). Forward and readout times reported per text.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "5")
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
torch.set_num_threads(5)

from datasets import load_dataset  # noqa: E402
from banking77_baseline import question_set  # noqa: E402
from tinyjev import load as load_agent  # noqa: E402
from tinyjev.families import softmax  # noqa: E402

N = 512
CHUNK = 64


def main():
    d = load_dataset("PolyAI/banking77")
    labels = d["train"].features["label"].names
    qmap = question_set(labels, "both")
    questions = [dict(id=k, **v) for k, v in qmap.items()]
    for q in questions:
        if q["type"] == "noul":
            q["type"] = "boolean"
    texts = [d["train"][2000 + i]["text"] for i in range(N)]

    agent = load_agent("TinyJev-0.6B", backend="torch", device="cuda")
    fam, model, device = agent.family, agent.backbone.model, agent.backbone.device
    encs = [fam.encode({"id": f"t{i}", "state": t, "questions": questions}) for i, t in enumerate(texts)]

    def run(config):
        fwd = rd = 0.0
        for s in range(0, N, CHUNK):
            idxs = list(range(s, min(s + CHUNK, N)))
            rows = []
            for ri in idxs:
                enc = encs[ri]
                for q in enc.questions:
                    if config == "choice" and q["type"] != "choice":
                        continue
                    if config == "probes" and q["type"] == "choice":
                        continue
                    rows.append((ri, q["id"], q["type"], enc.prefix + enc.rows[q["row"]]))
            t0 = time.perf_counter()
            # single group for choice/probes; two groups for combined
            groups = [rows] if config != "combined" else [
                [r for r in rows if r[2] == "choice"], [r for r in rows if r[2] != "choice"]]
            hidden = {}
            for group in groups:
                if not group:
                    continue
                width = max(len(r[3]) for r in group)
                ids = torch.full((len(group), width), fam.pad_token_id, dtype=torch.long)
                att = torch.zeros((len(group), width), dtype=torch.long)
                for i, (_, _, _, toks) in enumerate(group):
                    ids[i, :len(toks)] = torch.tensor(toks)
                    att[i, :len(toks)] = 1
                with torch.inference_mode():
                    h = model(input_ids=ids.to(device), attention_mask=att.to(device),
                              use_cache=False).last_hidden_state.float().cpu().numpy()
                for i, (ri, qid, _, toks) in enumerate(group):
                    hidden[(ri, qid)] = h[i, :len(toks)]
            t1 = time.perf_counter()
            from tinyjev.families import Encoded
            for ri in idxs:
                enc = encs[ri]
                sel = [q for q in enc.questions if (ri, q["id"]) in hidden]
                if not sel:
                    continue
                rows = [enc.rows[q["row"]] for q in sel]
                newq = []
                for j, q in enumerate(sel):
                    qq = dict(q)
                    qq["row"] = j
                    newq.append(qq)
                fenc = Encoded(prefix=enc.prefix, rows=rows, questions=newq)
                hidden_rows = [hidden[(ri, q["id"])] for q in sel]
                z = fam.logits(hidden_rows, fenc)
                for q in sel:
                    fam.answer(q, softmax(z[q["id"]]))
            t2 = time.perf_counter()
            fwd += t1 - t0
            rd += t2 - t1
        return 1000 * fwd / N, 1000 * rd / N

    # warmup
    run("combined")
    for config in ("choice", "probes", "combined"):
        f, r = run(config)
        print(f"{config:9s}: forward {f:6.1f} ms/text | readout {r:5.1f} ms/text | total {f + r:6.1f} ms/text")


if __name__ == "__main__":
    main()
