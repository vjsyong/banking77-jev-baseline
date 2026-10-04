#!/usr/bin/env python3
"""Stage C post-processing (brief §12): freeze selected banks -> confirmation eval
-> final refit on 3,000 labels -> official test -> registered analysis.

Order is enforced:
  1. freeze_banks(): read every run's final.json, freeze the selected bank per
     latency limit with hashes + timestamp BEFORE any confirmation access;
  2. confirmation evaluation (once, no revision): fit frozen learner on the
     2,250 discovery rows, evaluate the frozen banks on the 750 confirmation rows;
  3. final refit on all 3,000 permitted labels -> official test (4,500);
  4. baselines under the same allowance: TF-IDF+LR, embeddings+LR (bge-small),
     direct task Choice (zero-shot), supervised task Choice readout;
  5. analysis: primary F-E at L_primary (paired per seed + bootstrap over test
     messages), secondaries (E-R, E-U, TF-IDF+bank vs TF-IDF), checkpoints (H4).

Usage: python confirm.py stage1   # freeze + confirmation + refit (GPU heavy)
       python confirm.py stage2   # analysis once stage1 artifacts exist
"""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path("/home/xrim/banking77-jev-baseline")
sys.path.insert(0, str(HERE / "discovery"))
sys.path.insert(0, str(HERE / "discovery" / "pilot"))
sys.path.insert(0, str(HERE / "discovery" / "fe_discovery"))

import numpy as np  # noqa: E402

STAGE = HERE / "runs" / "clinc150" / "fe_discovery" / "stage_c"
OUT = HERE / "runs" / "clinc150" / "fe_discovery"
DATA = HERE / "data" / "clinc150" / "samples"
SEEDS = [11, 23, 37, 53, 71]
ARMS = ["U", "R", "E", "F"]
LIMITS = ["low", "primary", "high"]


def load_protocol():
    return json.loads((HERE / "discovery" / "fe_discovery" / "frozen_protocol.json").read_text())


def load_split(seed):
    flat = json.loads((DATA / "flat_rows.json").read_text())
    row = {r["row_id"]: r for r in flat}
    rec = json.loads((DATA / f"seed_{seed}.json").read_text())
    disc = [row[r] for r in rec["discovery_row_ids"]]
    conf = [row[r] for r in rec["confirmation_row_ids"]]
    return disc, conf, row


def test_rows():
    flat = json.loads((DATA / "flat_rows.json").read_text())
    return [r for r in flat if r["split"] == "test"]


# ---------------- 1. freeze banks ----------------
def freeze_banks():
    frozen = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "note": "selected banks per latency limit, frozen before any confirmation access",
              "seeds": {}}
    for seed in SEEDS:
        sdir = STAGE / f"seed_{seed}"
        if not (sdir / "seed_setup.json").exists():
            print(f"seed {seed}: not started yet")
            continue
        setup = json.loads((sdir / "seed_setup.json").read_text())
        frozen["seeds"][seed] = {"C": setup["C"], "limits": setup["limits"],
                                 "reference_ms_per_text": setup["reference_ms_per_text"],
                                 "arms": {}}
        for arm in ARMS:
            fp = sdir / arm / "final.json"
            if not fp.exists():
                print(f"  {arm} s{seed}: final.json missing")
                continue
            fin = json.loads(fp.read_text())
            pool = {d["idh"]: d for d in fin["pool"]}
            arms_rec = {"stop_reason": fin["stop_reason"], "rounds": fin["rounds"],
                        "teacher_tokens": fin["teacher_tokens"],
                        "ledger": fin["ledger"], "unique_evals": fin["unique_evals"],
                        "limits": {}}
            for ln in LIMITS:
                cur = fin["current"][ln]
                if cur is None:
                    continue
                defs = [pool[h] for h in cur["defs"]]
                arms_rec["limits"][ln] = {"cv": cur["cv"], "cost": cur["cost"],
                                          "names": cur["names"],
                                          "defs": [{"name": d["name"], "question": d["question"],
                                                    "options": d["options"], "slot": d["slot"],
                                                    "idh": d["idh"]} for d in defs],
                                          "bank_hash": hashlib.sha256(
                                              json.dumps([d["idh"] for d in defs]).encode()).hexdigest()[:16]}
            frozen["seeds"][seed]["arms"][arm] = arms_rec
    raw = json.dumps(frozen, indent=1, sort_keys=True)
    frozen["sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    (OUT / "frozen_banks.json").write_text(json.dumps(frozen, indent=1))
    print("frozen_banks.json written; sha", frozen["sha256"][:16])
    n = sum(len(a["limits"]) for s in frozen["seeds"].values() for a in s["arms"].values())
    print(f"frozen banks total: {n} (expected {len(SEEDS)*len(ARMS)*len(LIMITS)})")
    return frozen


# ---------------- 2-3. evaluation machinery ----------------
def bank_matrix_rs(store, defs, row_keys):
    """log(clip) block for a bank over given row keys (store-cached)."""
    cols = []
    for d in defs:
        for o in d["options"]:
            col = np.empty(len(row_keys))
            for i, k in enumerate(row_keys):
                v = store.choice_score_of(d["slot"], k, o["id"])
                col[i] = np.log(min(max(v, 1e-6), 1.0))
            cols.append(col)
    return np.array(cols, dtype=np.float64).T


def fit_pipeline(texts, y, sem, C):
    """Deployment fit (all rows): TF-IDF + scaled semantic + LR; returns predict fn."""
    import scipy.sparse as sp
    from learner import TFIDF_CHAR, TFIDF_WORD, make_lr, _hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    from sklearn.preprocessing import StandardScaler
    vec = FeatureUnion([("word", TfidfVectorizer(**TFIDF_WORD)),
                        ("char", TfidfVectorizer(**TFIDF_CHAR))]).fit(texts)
    scaler = StandardScaler().fit(sem) if sem.shape[1] else None
    X = _hstack(vec.transform(texts).tocsr(), scaler.transform(sem) if scaler is not None else None)
    clf = make_lr(C).fit(X, y)

    def predict(new_texts, new_sem):
        Xn = vec.transform(new_texts).tocsr()
        if scaler is not None and new_sem.shape[1]:
            Xn = sp.hstack([Xn, sp.csr_matrix(scaler.transform(new_sem))], format="csr")
        return clf.predict(Xn)
    return predict


def stage1():
    tokens = sys.argv[2] if len(sys.argv) > 2 else "all"
    frozen = freeze_banks()
    from extract_probes import ProbeStore, text_key
    from fe_extract import FEExtractor
    from sklearn.metrics import accuracy_score, f1_score

    store = ProbeStore(str(HERE / "runs" / "clinc150" / "fe_discovery" / "probe_scores.sqlite"))
    ex = FEExtractor(store)
    test = test_rows()
    test_keys = [text_key(r["text"]) for r in test]
    y_test = np.array([r["label"] for r in test])

    results = {}
    for seed in SEEDS:
        if seed not in frozen["seeds"]:
            continue
        disc, conf, _ = load_split(seed)
        texts_d = [r["text"] for r in disc]
        y_d = np.array([r["label"] for r in disc])
        texts_c = [r["text"] for r in conf]
        y_c = np.array([r["label"] for r in conf])
        C = frozen["seeds"][seed]["C"]
        print(f"\n== seed {seed} (C={C}) ==", flush=True)

        for arm in ARMS:
            rec_arm = frozen["seeds"][seed]["arms"].get(arm)
            if not rec_arm:
                continue
            for ln in LIMITS:
                rec = rec_arm["limits"].get(ln)
                if not rec:
                    continue
                defs = rec["defs"]
                # ensure extraction on confirmation + test rows
                for d in defs:
                    wall, _ = ex.extract_choice_fe(d["slot"], d["question"], d["options"],
                                                   texts_c + [r["text"] for r in test])
                sem_c = bank_matrix_rs(store, defs, [text_key(t) for t in texts_c])
                sem_te = bank_matrix_rs(store, defs, test_keys)
                sem_d = bank_matrix_rs(store, defs, [text_key(t) for t in texts_d])
                # confirmation: fit on discovery, evaluate once
                predc = fit_pipeline(texts_d, y_d, sem_d, C)
                f1c = float(f1_score(y_c, predc(texts_c, sem_c), average="macro", zero_division=0))
                # final refit on 3000 -> test
                pred = fit_pipeline(texts_d + texts_c, np.concatenate([y_d, y_c]),
                                    np.vstack([sem_d, sem_c]), C)
                pt = pred([r["text"] for r in test], sem_te)
                f1t = float(f1_score(y_test, pt, average="macro", zero_division=0))
                acc_t = float(accuracy_score(y_test, pt))
                results.setdefault(seed, {}).setdefault(arm, {})[ln] = {
                    "confirmation_macro_f1": round(f1c, 4),
                    "test_macro_f1": round(f1t, 4), "test_accuracy": round(acc_t, 4),
                    "n_questions": len(defs),
                    "test_preds": pt.tolist(),
                }
                print(f"  {arm} {ln}: conf {f1c:.4f} | test {f1t:.4f} (nq {len(defs)})", flush=True)
        (OUT / "stage_eval_partial.json").write_text(json.dumps(results, indent=1))
    store.close()
    print("stage1 done -> stage_eval_partial.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    if cmd == "freeze":
        freeze_banks()
    elif cmd == "stage1":
        stage1()
    else:
        print("unknown")
