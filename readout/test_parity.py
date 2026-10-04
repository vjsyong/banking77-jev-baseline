#!/usr/bin/env python3
"""Deployment parity tests for banking77-readout-v1.

(a) saved artifact reproduces the experimental test metrics
(b) shuffled probability keys give identical predictions
(c) tiny/zero probabilities handled: clipping, no NaN/inf, deterministic
(d) single vs batched requests identical; dict and array forms agree
(e) guard rails: missing intents / wrong width raise

Run: ./venv/bin/python readout/test_parity.py -v
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from readout_pipeline import Banking77Readout  # noqa: E402

RUNS = HERE.parent / "runs" / "banking77"


class Parity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ro = Banking77Readout.load(HERE / "banking77-readout-v1")
        pred = pd.read_csv(RUNS / "jev_predictions.csv")
        probs = [json.loads(s) for s in pred["probabilities_json"]]
        cls.labels = sorted(probs[0].keys())
        cls.P = np.array([[p[l] for l in cls.labels] for p in probs], dtype=float)
        cls.y = pred["label"].to_numpy()
        cls.te = (pred["split"] == "test").to_numpy()

    def _proba_matrix(self, out):
        return np.array([[o["probabilities"][l] for l in self.ro.labels] for o in out])

    def test_a_saved_artifact_reproduces_experiment(self):
        out = self.ro.predict(self.P[self.te])
        preds = [o["label"] for o in out]
        m = self.ro.manifest["evaluation"]
        self.assertAlmostEqual(f1_score(self.y[self.te], preds, average="macro"),
                               m["macro_f1"], delta=1e-4)
        self.assertAlmostEqual(accuracy_score(self.y[self.te], preds),
                               m["accuracy"], delta=1e-4)
        self.assertAlmostEqual(
            top_k_accuracy_score(self.y[self.te], self._proba_matrix(out), k=3,
                                 labels=self.ro.labels),
            m["top3"], delta=1e-4)

    def test_b_shuffled_probability_keys(self):
        rng = np.random.default_rng(7)
        idx = rng.choice(len(self.P), 250, replace=False)
        for i in idx:
            ordered = {l: float(p) for l, p in zip(self.ro.labels, self.P[i])}
            keys = list(ordered)
            rng.shuffle(keys)
            shuffled = {k: ordered[k] for k in keys}
            a = self.ro.predict(ordered)
            b = self.ro.predict(shuffled)
            self.assertEqual(a["label"], b["label"])
            self.assertAlmostEqual(a["confidence"], b["confidence"], places=12)
            np.testing.assert_allclose(
                [a["probabilities"][l] for l in self.ro.labels],
                [b["probabilities"][l] for l in self.ro.labels], atol=1e-12)

    def test_c_tiny_probabilities(self):
        rng = np.random.default_rng(11)
        idx = rng.choice(len(self.P), 200, replace=False)
        arrs = []
        for i in idx:
            v = self.P[i].copy()
            small = np.argsort(v)[:5]
            v[small[:3]] = 0.0
            v[small[3:]] = 1e-12
            arrs.append(v)
        out = self.ro.predict(np.array(arrs))
        for o in out:
            vals = list(o["probabilities"].values())
            self.assertTrue(np.isfinite(vals).all())
            self.assertTrue(np.isfinite(o["confidence"]))
        # all-zeros pathological vector
        z = self.ro.predict(np.zeros(77))
        self.assertTrue(np.isfinite(z["confidence"]))
        self.assertTrue(np.isfinite(list(z["probabilities"].values())).all())
        # features stay finite under clipping
        X = self.ro.features(np.zeros(77))
        self.assertTrue(np.isfinite(X).all())
        # determinism
        again = self.ro.predict(np.array(arrs))
        np.testing.assert_allclose(self._proba_matrix(out), self._proba_matrix(again), atol=1e-12)

    def test_d_single_vs_batched_and_forms(self):
        rng = np.random.default_rng(13)
        idx = rng.choice(len(self.P), 100, replace=False)
        batched = self.ro.predict(self.P[idx])
        for j, i in enumerate(idx):
            single_arr = self.ro.predict(self.P[i])                      # (77,) array
            single_dict = self.ro.predict({l: float(p) for l, p in
                                           zip(self.ro.labels, self.P[i])})  # dict
            self.assertEqual(single_arr["label"], batched[j]["label"])
            self.assertAlmostEqual(single_arr["confidence"], batched[j]["confidence"],
                                   places=12)
            self.assertEqual(single_dict["label"], single_arr["label"])
            np.testing.assert_allclose(
                [single_arr["probabilities"][l] for l in self.ro.labels],
                [batched[j]["probabilities"][l] for l in self.ro.labels], atol=1e-12)

    def test_e_guard_rails(self):
        with self.assertRaises(ValueError):
            self.ro.predict({l: 0.1 for l in self.ro.labels[:-1]})       # missing intent
        with self.assertRaises(ValueError):
            self.ro.predict(np.ones(70))                                  # wrong width


if __name__ == "__main__":
    unittest.main(verbosity=2)
