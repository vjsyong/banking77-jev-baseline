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


def stage1(mon=None):
    tokens = sys.argv[2] if len(sys.argv) > 2 else "all"
    frozen = freeze_banks()
    if mon is not None:
        mon.log(f"banks frozen sha {frozen['sha256'][:12]}")
    from extract_probes import ProbeStore, text_key
    from fe_extract import FEExtractor
    from sklearn.metrics import accuracy_score, f1_score

    store = ProbeStore(str(HERE / "runs" / "clinc150" / "fe_discovery" / "probe_scores.sqlite"))
    ex = FEExtractor(store)
    test = test_rows()
    test_keys = [text_key(r["text"]) for r in test]
    y_test = np.array([r["label"] for r in test])

    results = {}
    done = 0
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
                done += 1
                if mon is not None:
                    mon.update(done, message=f"s{seed} {arm} {ln}: test {f1t:.4f} (nq {len(defs)})")
        (OUT / "stage_eval_partial.json").write_text(json.dumps(results, indent=1))
    store.close()
    print("stage1 done -> stage_eval_partial.json")


def baselines_and_stage2(mon=None, step_base=0):
    """Baselines under the same allowance + registered analysis (brief §12)."""
    import json as _json
    from extract_probes import ProbeStore, text_key
    from fe_extract import FEExtractor, bank_matrix  # noqa: F401
    from sklearn.metrics import accuracy_score, f1_score
    from learner import make_lr
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    import scipy.sparse as sp

    proto = load_protocol()
    frozen = _json.loads((OUT / "frozen_banks.json").read_text())
    partial = _json.loads((OUT / "stage_eval_partial.json").read_text())
    test = test_rows()
    y_test = np.array([r["label"] for r in test])
    ttexts = [r["text"] for r in test]

    store = ProbeStore(str(HERE / "runs" / "clinc150" / "fe_discovery" / "probe_scores.sqlite"))
    ex = FEExtractor(store)
    intents, idrec = _json.loads((HERE / "data" / "clinc150" / "intent_descriptions_frozen.json").read_text()), None
    intents = intents["intents"]

    # task question (identical across seeds)
    defn = {"name": "task_intent", "question": "Which intent does this message express?",
            "options": [{"id": f"i{k:03d}", "definition": f"{n}: {d}"} for k, (n, d) in enumerate(intents)]}
    import hashlib as _h
    from schema import canonical_hash
    defn["slot"] = f"fe::task::{canonical_hash({k: v for k, v in defn.items() if k != 'slot'})[:12]}"

    baselines = {}
    done = step_base
    for seed in SEEDS:
        disc, conf, _ = load_split(seed)
        td = [r["text"] for r in disc]
        tc = [r["text"] for r in conf]
        yd = np.array([r["label"] for r in disc])
        yc = np.array([r["label"] for r in conf])
        C = frozen["seeds"][seed]["C"]
        # task extraction on test + this seed's conf (discovery done at setup)
        for texts_set in (ttexts, tc):
            ex.extract_choice_fe(defn["slot"], defn["question"], defn["options"], texts_set)

        def task_probs(texts):
            rows = []
            for t in texts:
                rows.append([min(max(store.choice_score_of(defn["slot"], text_key(t), o["id"]), 1e-6), 1.0)
                             for o in defn["options"]])
            return np.log(np.array(rows))

        P_te, P_c, P_d = task_probs(ttexts), task_probs(tc), task_probs(td)
        # direct choice zero-shot: argmax (intents entries are [name, definition] pairs)
        pred_zs = [intents[i][0] for i in P_te.argmax(1)]
        # supervised readout: LR on log-probs
        ro = make_lr(C).fit(P_d, yd)
        ro_te = ro.predict(P_te)
        ro_c = ro.predict(P_c)
        # tfidf only
        vec = FeatureUnion([("word", TfidfVectorizer(**__import__("learner").TFIDF_WORD)),
                            ("char", TfidfVectorizer(**__import__("learner").TFIDF_CHAR))]).fit(td + tc)
        tf = make_lr(C).fit(vec.transform(td + tc), np.concatenate([yd, yc]))
        tf_te = tf.predict(vec.transform(ttexts))
        tf_c = tf.predict(vec.transform(tc))
        # embeddings
        from embeddings import encode as emb_encode, fit_full as emb_fit
        e_d, _ = emb_encode(td)
        e_c, _ = emb_encode(tc)
        e_te, _ = emb_encode(ttexts)
        sc, clf = emb_fit(np.vstack([e_d, e_c]), np.concatenate([yd, yc]), C=max(2.0, C / 8))
        emb_te = clf.predict(sc.transform(e_te))
        baselines[seed] = {
            "C": C,
            "zeroshot_direct_choice": {"test_macro_f1": round(float(f1_score(y_test, pred_zs, average="macro", zero_division=0)), 4)},
            "supervised_readout": {"conf_macro_f1": round(float(f1_score(yc, ro_c, average="macro", zero_division=0)), 4),
                                    "test_macro_f1": round(float(f1_score(y_test, ro_te, average="macro", zero_division=0)), 4)},
            "tfidf_lr": {"conf_macro_f1": round(float(f1_score(yc, tf_c, average="macro", zero_division=0)), 4),
                          "test_macro_f1": round(float(f1_score(y_test, tf_te, average="macro", zero_division=0)), 4)},
            "embeddings_lr": {"test_macro_f1": round(float(f1_score(y_test, emb_te, average="macro", zero_division=0)), 4)},
        }
        print(f"seed {seed} baselines: tfidf {baselines[seed]['tfidf_lr']['test_macro_f1']} | "
              f"emb {baselines[seed]['embeddings_lr']['test_macro_f1']} | readout {baselines[seed]['supervised_readout']['test_macro_f1']} | "
              f"zs {baselines[seed]['zeroshot_direct_choice']['test_macro_f1']}", flush=True)
        done += 1
        if mon is not None:
            mon.update(done, message=f"s{seed} baselines: tfidf {baselines[seed]['tfidf_lr']['test_macro_f1']}")
    store.close()
    if mon is not None:
        mon.log("baselines done; running registered paired analysis (1000x bootstrap)")

    # ---- registered analysis ----
    rng = np.random.default_rng(7)
    n = len(y_test)
    idx = np.arange(n)
    BOOT = 1000
    resamples = [rng.choice(idx, size=n, replace=True) for _ in range(BOOT)]

    def arm_preds(seed, arm, ln="primary"):
        return np.array(partial[str(seed)][arm][ln]["test_preds"])

    def f1(preds, sub=None):
        if sub is None:
            return float(f1_score(y_test, preds, average="macro", zero_division=0))
        return float(f1_score(y_test[sub], preds[sub], average="macro", zero_division=0))

    def paired(name_a, arm_a, arm_b):
        per_seed = []
        for seed in SEEDS:
            if str(seed) not in partial or arm_a not in partial[str(seed)] or arm_b not in partial[str(seed)]:
                continue
            a, b = arm_preds(seed, arm_a), arm_preds(seed, arm_b)
            per_seed.append({"seed": seed, "delta_pp": round(100 * (f1(a) - f1(b)), 2),
                             "f1_a": round(f1(a), 4), "f1_b": round(f1(b), 4)})
        if not per_seed:
            return None
        deltas = [d["delta_pp"] for d in per_seed]
        boot = []
        seeds_ok = [seed for seed in SEEDS if str(seed) in partial and arm_a in partial[str(seed)] and arm_b in partial[str(seed)]]
        for sub in resamples:
            ds = []
            for seed in seeds_ok:
                a, b = arm_preds(seed, arm_a), arm_preds(seed, arm_b)
                ds.append(f1(a, sub) - f1(b, sub))
            boot.append(100 * float(np.mean(ds)))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"contrast": name_a, "per_seed": per_seed,
                "mean_delta_pp": round(float(np.mean(deltas)), 2),
                "seed_spread_pp": [round(min(deltas), 2), round(max(deltas), 2)],
                "bootstrap95_mean_pp": [round(float(lo), 2), round(float(hi), 2)],
                f"positive_seeds": f"{sum(1 for d in deltas if d > 0)}/{len(deltas)}"}

    analysis = {
        "primary_F_minus_E_at_L_primary": paired("F-E", "F", "E"),
        "secondary_E_minus_R": paired("E-R", "E", "R"),
        "secondary_E_minus_U": paired("E-U", "E", "U"),
        "secondary_F_minus_U": paired("F-U", "F", "U"),
    }
    # checkpoints (H4): quality vs charged extraction compute per arm
    checkpoints = {}
    for seed in SEEDS:
        for arm in ARMS:
            rp = STAGE / f"seed_{seed}" / arm / "rounds.jsonl"
            if not rp.exists():
                continue
            rounds = [_json.loads(l) for l in rp.read_text().splitlines() if l.strip()]
            cps = []
            for frac in (0.25, 0.5, 0.75, 1.0):
                limit_q = 900.0 * frac
                bank = None
                for r in rounds:
                    if r["ledger"]["logical_charge_s"] <= limit_q:
                        bank = r["current_primary"]
                cps.append({"frac": frac, "cv": (bank or {}).get("cv"), "nq": (bank or {}).get("nq"),
                            "charged_s": rounds[-1]["ledger"]["logical_charge_s"]})
            checkpoints.setdefault(str(arm), {})[str(seed)] = cps

    out = {"baselines": baselines, "analysis": analysis, "checkpoints": checkpoints,
           "frozen_banks_sha256": frozen.get("sha256")}
    (OUT / "confirm_report.json").write_text(_json.dumps(out, indent=1))
    print(_json.dumps({k: (v if k != "baselines" else "saved") for k, v in analysis.items()}, indent=1))
    print("written confirm_report.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    if cmd == "freeze":
        freeze_banks()
        raise SystemExit(0)
    try:
        sys.path.insert(0, "/home/xrim/taildash/client")
        from taildash import TaskMonitor
        mon = TaskMonitor(server="http://localhost:8080",
                          title=f"FE discovery: confirm ({cmd}) [clinc150]",
                          total={"stage1": 60, "stage2": 5, "full": 65}.get(cmd, 1),
                          agent_name="fe-discovery")
    except Exception:  # noqa: BLE001
        class _N:
            def log(self, *a, **k): pass
            def update(self, *a, **k): pass
            def complete(self, *a, **k): pass
            def fail(self, *a, **k): pass
        mon = _N()
    try:
        if cmd == "stage1":
            stage1(mon)
        elif cmd == "stage2":
            baselines_and_stage2(mon)
        elif cmd == "full":
            stage1(mon)
            baselines_and_stage2(mon, step_base=60)
        else:
            print("unknown")
            raise SystemExit(1)
        mon.complete("done")
    except BaseException as exc:  # noqa: BLE001
        mon.fail(f"{type(exc).__name__}: {exc}")
        raise
