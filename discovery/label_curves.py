#!/usr/bin/env python3
"""Label-budget learning curves: TF-IDF+LR vs Choice readout vs direct argmax.

Answers "is it worth it if TF-IDF is already 0.9?" with the decision-relevant
axis: quality as a function of LABELS. All features already exist (no inference):
the readout consumes stored 77-probability vectors; TF-IDF refits on each
subsample (faithful to the 0.9094 full-label baseline protocol).

Budgets: 250 / 500 / 1000 / 2500 / 5000 / 10003 (full). Seeds: 77, 78.
Test split: full 3,080 (untouched by the subsampling).
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "5")
HERE = Path("/home/xrim/banking77-jev-baseline")
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
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import FeatureUnion  # noqa: E402

RUNS = HERE / "runs" / "banking77"
OUT = RUNS / "discovery" / "label_curves.json"
CLIP = (1e-6, 1.0)
BUDGETS = [250, 500, 1000, 2500, 5000, 10003]
SEEDS = [77, 78]
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


def main():
    global _ACTIVE_MONITOR
    pred = pd.read_csv(RUNS / "jev_predictions.csv")
    train = pred[pred["split"] == "train"].reset_index(drop=True)
    test = pred[pred["split"] == "test"].reset_index(drop=True)
    y_tr = train["label"].to_numpy()
    y_te = test["label"].to_numpy()
    labels77 = sorted(json.loads(pred.iloc[0]["probabilities_json"]).keys())

    def logp(df):
        P = np.array([[json.loads(s)[l] for l in labels77] for s in df["probabilities_json"]],
                     dtype=float)
        return np.log(np.clip(P, *CLIP))

    X_logp_tr_full = logp(train)
    X_logp_te = logp(test)
    argmax_pred = np.array(labels77)[np.exp(X_logp_te).argmax(1)]
    direct_f1 = float(f1_score(y_te, argmax_pred, average="macro"))

    n_jobs = 2 * len(BUDGETS)
    mon = (_TaskMonitor(server="http://localhost:8080",
                        title="label-budget learning curves (TF-IDF vs readout)",
                        total=n_jobs, agent_name="banking77-label-curves")
           if _TaskMonitor else _NullMonitor())
    _ACTIVE_MONITOR = mon
    mon.log(f"curves: budgets={BUDGETS} seeds={SEEDS}; direct argmax={direct_f1:.4f}")

    out = {"direct_argmax": round(direct_f1, 4), "budgets": {}, "seeds": SEEDS}
    done = 0
    for budget in BUDGETS:
        budgets_here = [budget] if budget >= train.shape[0] else SEEDS
        for seed in budgets_here:
            key = f"{budget}"
            t0 = time.perf_counter()
            if budget >= train.shape[0]:
                sub_idx = np.arange(len(train))
            else:
                sub_idx, _ = train_test_split(np.arange(len(train)), train_size=budget,
                                              stratify=y_tr, random_state=seed)
            texts_sub = train.iloc[sub_idx]["text"].tolist()
            texts_te = test["text"].tolist()

            # TF-IDF + LR (bundle config)
            vect = FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                         strip_accents="unicode", max_features=300_000)),
                ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                                         sublinear_tf=True, max_features=300_000)),
            ], n_jobs=1)
            Xt = vect.fit_transform(texts_sub)
            Xe = vect.transform(texts_te)
            clf = LogisticRegression(C=4.0, max_iter=1000, solver="lbfgs")
            clf.fit(Xt, y_tr[sub_idx])
            tfidf_f1 = float(f1_score(y_te, clf.predict(Xe), average="macro"))

            # Choice readout (frozen protocol: log-probs -> StandardScaler -> LR C=1)
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            ro = Pipeline([("sc", StandardScaler()),
                           ("m", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs"))])
            ro.fit(X_logp_tr_full[sub_idx], y_tr[sub_idx])
            readout_f1 = float(f1_score(y_te, ro.predict(X_logp_te), average="macro"))

            dt = time.perf_counter() - t0
            rec = out["budgets"].setdefault(key, {"n": int(len(sub_idx)), "runs": []})
            rec["runs"].append({"seed": seed, "tfidf": round(tfidf_f1, 4),
                                "readout": round(readout_f1, 4), "s": round(dt, 1)})
            print(f"budget {budget:6d} seed {seed}: TF-IDF {tfidf_f1:.4f} | readout {readout_f1:.4f} "
                  f"({dt:.0f}s)", flush=True)
            done += 1
            mon.update(done, message=f"budget {budget} seed {seed}: tfidf {tfidf_f1:.3f} / readout {readout_f1:.3f}")

    for b, rec in out["budgets"].items():
        rec["tfidf_mean"] = round(float(np.mean([r["tfidf"] for r in rec["runs"]])), 4)
        rec["readout_mean"] = round(float(np.mean([r["readout"] for r in rec["runs"]])), 4)
        rec["delta_pp"] = round(100 * (rec["readout_mean"] - rec["tfidf_mean"]), 2)
    OUT.write_text(json.dumps(out, indent=2))
    print("\nSUMMARY (test macro-F1):")
    for b, rec in out["budgets"].items():
        print(f"  n={rec['n']:6d}: tfidf {rec['tfidf_mean']:.4f} | readout {rec['readout_mean']:.4f} "
              f"| Δ {rec['delta_pp']:+.2f}pp")
    print("written", OUT)
    mon.complete("curves done")
    # plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [rec["n"] for rec in out["budgets"].values()]
        tfs = [rec["tfidf_mean"] for rec in out["budgets"].values()]
        ros = [rec["readout_mean"] for rec in out["budgets"].values()]
        plt.figure(figsize=(8, 5), dpi=160)
        plt.plot(xs, tfs, "o-", label="TF-IDF + LR")
        plt.plot(xs, ros, "o-", label="Choice readout")
        plt.axhline(direct_f1, color="gray", ls="--", lw=1, label=f"direct argmax ({direct_f1:.3f})")
        plt.xscale("log")
        plt.xlabel("training labels (log scale)")
        plt.ylabel("test macro-F1")
        plt.title("Label-budget curves — BANKING77 (test split)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(RUNS / "discovery" / "label_curves.png")
        print("plot written:", RUNS / "discovery" / "label_curves.png")
    except Exception as exc:  # noqa: BLE001
        print("plot skipped:", exc)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        if _ACTIVE_MONITOR is not None:
            _ACTIVE_MONITOR.fail(f"{type(exc).__name__}: {exc}")
        raise
