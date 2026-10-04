#!/usr/bin/env python3
"""End-to-end benchmark: text -> prediction, frozen readout pipeline vs TF-IDF + LR.

A. Jev choice-only + readout, single requests (N=300): p50/p95/mean + component breakdown
B. Jev choice-only + readout, batched (chunk=64, N=512): throughput, ms/text
C. Readout alone: single-row predict vs batch predict
D. TF-IDF + LR (refit artifact): single requests (N=300) and batched (N=512)

Jev runs on the RTX 3090 (fp16, in-process); TF-IDF on CPU. Warmups excluded.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "5")
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(5)

from datasets import load_dataset  # noqa: E402
from banking77_baseline import question_set  # noqa: E402
from tinyjev import load as load_agent  # noqa: E402
from tinyjev.families import softmax  # noqa: E402
from readout_pipeline import Banking77Readout  # noqa: E402

RUNS = REPO / "runs" / "banking77"
OUT = RUNS / "analysis" / "pipeline_benchmark.json"


def pct(xs, p):
    return float(np.percentile(xs, p))


def main():
    d = load_dataset("PolyAI/banking77")
    labels = d["train"].features["label"].names
    qmap = question_set(labels, "both")
    q_choice = {k: v for k, v in qmap.items() if k == "direct_intent"}
    questions = [dict(id=k, **v) for k, v in q_choice.items()]

    ro = Banking77Readout.load(HERE / "banking77-readout-v1")
    results = {"hardware": {"jev": "RTX 3090 fp16 (torch, in-process)", "tfidf": "CPU",
                            "note": "warmups excluded; box otherwise idle"}}

    agent = load_agent("TinyJev-0.6B", backend="torch", device="cuda")
    fam, model, dev = agent.family, agent.backbone.model, agent.backbone.device

    def jev_probs_one(text):
        """single-request path: encode -> 1-row forward -> head -> probs dict"""
        t0 = time.perf_counter()
        enc = fam.encode({"id": "x", "state": text, "questions": questions})
        t1 = time.perf_counter()
        row = enc.prefix + enc.rows[0]
        ids = torch.tensor([row], device=dev)
        att = torch.ones_like(ids)
        with torch.inference_mode():
            h = model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state.float().cpu().numpy()[0]
        t2 = time.perf_counter()
        z = fam.logits([h], enc)
        probs = softmax(z["direct_intent"])
        pdict = {k: float(v) for k, v in zip(enc.questions[0]["keys"], probs)}
        t3 = time.perf_counter()
        return pdict, (t1 - t0, t2 - t1, t3 - t2)

    # warmups
    for i in range(10):
        jev_probs_one(d["train"][9000 + i]["text"])
        ro.predict(np.zeros(77))

    # ---- A: single requests, N=300 (full chain incl. readout)
    a_texts = [d["train"][5000 + i]["text"] for i in range(300)]
    a_lat, a_enc, a_fwd, a_head_ro = [], [], [], []
    a_ro = []
    for t in a_texts:
        pdict, (te_, tf_, th_) = jev_probs_one(t)
        t0 = time.perf_counter()
        ro.predict(pdict)
        tro = time.perf_counter() - t0
        a_lat.append(te_ + tf_ + th_ + tro)
        a_enc.append(te_); a_fwd.append(tf_); a_head_ro.append(th_); a_ro.append(tro)
    results["jev_single_300"] = {
        "total_ms": {"p50": 1e3 * pct(a_lat, 50), "p95": 1e3 * pct(a_lat, 95),
                     "mean": 1e3 * float(np.mean(a_lat))},
        "encode_ms": {"p50": 1e3 * pct(a_enc, 50), "mean": 1e3 * float(np.mean(a_enc))},
        "forward_ms": {"p50": 1e3 * pct(a_fwd, 50), "mean": 1e3 * float(np.mean(a_fwd))},
        "head_ms": {"p50": 1e3 * pct(a_head_ro, 50), "mean": 1e3 * float(np.mean(a_head_ro))},
        "readout_ms": {"p50": 1e3 * pct(a_ro, 50), "mean": 1e3 * float(np.mean(a_ro))},
    }
    print("A jev single:", json.dumps(results["jev_single_300"]["total_ms"]))

    # ---- B: batched, N=512, chunk=64
    b_texts = [d["train"][6000 + i]["text"] for i in range(512)]
    encs = [fam.encode({"id": f"b{i}", "state": t, "questions": questions}) for i, t in enumerate(b_texts)]
    chunk = 64
    per_text, chunk_times = [], []
    t_all = time.perf_counter()
    for s in range(0, len(encs), chunk):
        idxs = list(range(s, min(s + chunk, len(encs))))
        t0 = time.perf_counter()
        rows = [encs[i].prefix + encs[i].rows[0] for i in idxs]
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), fam.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = torch.tensor(r)
            att[i, :len(r)] = 1
        with torch.inference_mode():
            hs = model(input_ids=ids.to(dev), attention_mask=att.to(dev),
                       use_cache=False).last_hidden_state.float().cpu().numpy()
        pdicts = []
        for j, i in enumerate(idxs):
            h = hs[j, :len(rows[j])]
            z = fam.logits([h], encs[i])
            probs = softmax(z["direct_intent"])
            pdicts.append({k: float(v) for k, v in zip(encs[i].questions[0]["keys"], probs)})
        ro.predict(pdicts)
        dt = time.perf_counter() - t0
        chunk_times.append(1e3 * dt / len(idxs))
        per_text.append(dt)
    b_total = time.perf_counter() - t_all
    results["jev_batched_512"] = {
        "ms_per_text_p50": pct(chunk_times, 50), "ms_per_text_mean": float(np.mean(chunk_times)),
        "texts_per_s": len(encs) / b_total,
        "note": "includes encode, forward, head, readout"}
    print("B jev batched:", json.dumps(results["jev_batched_512"]))

    # ---- C: readout alone
    c_rows = [pdict for pdict in
              [ {k: 0.01 for k in labels} for _ in range(300)]]
    t0 = time.perf_counter()
    for r in c_rows:
        ro.predict(r)
    c_single = (time.perf_counter() - t0) / len(c_rows)
    c_mat = np.full((512, 77), 0.01)
    t0 = time.perf_counter()
    ro.predict(c_mat)
    c_batch = time.perf_counter() - t0
    results["readout_alone"] = {"single_ms": 1e3 * c_single, "batch512_ms": 1e3 * c_batch,
                                "batch512_ms_per_text": 1e3 * c_batch / 512}
    print("C readout:", json.dumps(results["readout_alone"]))

    # ---- D: TF-IDF + LR
    tf = joblib.load(RUNS / "tfidf_lr_refit.joblib")
    vect, clf = tf["vect"], tf["clf"]
    d_texts = [d["train"][7000 + i]["text"] for i in range(300)]
    # warmup
    for t in d_texts[:10]:
        clf.predict(vect.transform([t]))
    d_lat = []
    for t in d_texts:
        t0 = time.perf_counter()
        clf.predict(vect.transform([t]))
        d_lat.append(time.perf_counter() - t0)
    results["tfidf_single_300"] = {
        "total_ms": {"p50": 1e3 * pct(d_lat, 50), "p95": 1e3 * pct(d_lat, 95),
                     "mean": 1e3 * float(np.mean(d_lat))}}
    d2_texts = [d["train"][7000 + i]["text"] for i in range(512)]
    t0 = time.perf_counter()
    X = vect.transform(d2_texts)
    t1 = time.perf_counter()
    clf.predict(X)
    t2 = time.perf_counter()
    results["tfidf_batched_512"] = {
        "transform_ms_per_text": 1e3 * (t1 - t0) / 512,
        "predict_ms_per_text": 1e3 * (t2 - t1) / 512,
        "total_ms_per_text": 1e3 * (t2 - t0) / 512,
        "texts_per_s": 512 / (t2 - t0)}
    print("D tfidf:", json.dumps(results["tfidf_single_300"]["total_ms"]),
          json.dumps(results["tfidf_batched_512"]))

    OUT.write_text(json.dumps(results, indent=2))
    print("written", OUT)


if __name__ == "__main__":
    main()
