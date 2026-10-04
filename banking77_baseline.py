#!/usr/bin/env python3
"""Classical and TinyJev-compatible baselines for BANKING77.

All user text is sent only to the configured System One endpoint. The dataset
label is never included in a Jev request. Completed requests are resumable via
SQLite, keyed by text, question schema and model name.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


HERE = Path(__file__).resolve().parent
PROBE_FILE = HERE / "probes.json"
DATASET_ID = "PolyAI/banking77"
DIRECT_ID = "direct_intent"
USER_AGENT = "banking77-jev-baseline/1.0"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def load_probes() -> list[dict[str, str]]:
    return json.loads(PROBE_FILE.read_text(encoding="utf-8"))


def load_dataset_rows(max_rows_per_split: int | None = None):
    if max_rows_per_split is not None and max_rows_per_split < 1:
        raise ValueError("--max-rows-per-split must be a positive integer")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt first; the 'datasets' package is missing.") from exc
    data = load_dataset(DATASET_ID)
    train = data["train"]
    test = data["test"]
    label_feature = train.features.get("label")
    label_names = getattr(label_feature, "names", None)
    if not label_names:
        raise RuntimeError("BANKING77 label names were not present in the dataset metadata.")

    result = {}
    for split_name, split in (("train", train), ("test", test)):
        count = min(len(split), max_rows_per_split) if max_rows_per_split else len(split)
        rows = []
        for index in range(count):
            item = split[index]
            label_index = int(item["label"])
            rows.append({"split": split_name, "row": index,
                         "text": str(item["text"]), "label": label_names[label_index]})
        result[split_name] = rows
    return result, list(label_names), {"train": len(train), "test": len(test)}


def question_set(labels: list[str], mode: str):
    questions: dict[str, Any] = {}
    if mode in ("both", "direct"):
        questions[DIRECT_ID] = {
            "type": "choice",
            "instructions": "Which one of these intents best matches the customer's main online-banking request or problem? Choose the single best intent.",
            "criteria": {label: label.replace("_", " ") for label in labels},
        }
    if mode in ("both", "features"):
        for probe in load_probes():
            questions["probe__" + probe["id"]] = {
                "type": "noul", "instructions": probe["question"]
            }
    return questions


def cache_key(text: str, model: str, questions: dict[str, Any]) -> str:
    return sha(canonical({"text_sha256": sha(text), "model": model, "questions": questions}))


def response_answers(response: dict[str, Any]) -> dict[str, Any]:
    """Accept the documented System One result envelope and common flat form."""
    if not isinstance(response, dict):
        raise ValueError("System One response must be a JSON object")
    states = response.get("states")
    if isinstance(states, list) and states and isinstance(states[0], dict):
        answers = states[0].get("answers")
        if isinstance(answers, dict):
            return answers
    answers = response.get("answers")
    if isinstance(answers, dict):
        return answers
    raise ValueError("Could not find an 'answers' map in the System One response")


def json_request(endpoint: str, model: str, text: str, questions: dict[str, Any],
                 token: str | None, timeout: float, retries: int):
    body = json.dumps({"model": model, "state": text, "questions": questions},
                      ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json",
               "User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    last_error = None
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            parsed = json.loads(raw.decode("utf-8"))
            answers = response_answers(parsed)
            missing = [key for key in questions if key not in answers]
            if missing:
                raise ValueError("System One response is missing answers: " + ", ".join(missing[:4]))
            elapsed = time.perf_counter() - started
            return parsed, elapsed
        except urllib.error.HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", errors="replace")
            last_error = RuntimeError(f"System One HTTP {exc.code}: {detail}")
            if exc.code not in (408, 425, 429, 500, 502, 503, 504) or attempt >= retries:
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = RuntimeError(f"System One request failed: {exc}")
            if attempt >= retries:
                break
        time.sleep(min(2 ** attempt, 12))
    raise last_error or RuntimeError("System One request failed")


class ResultCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS jev_cache (
            cache_key TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            elapsed_s REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        self.db.commit()

    def get(self, key: str):
        row = self.db.execute("SELECT payload, elapsed_s FROM jev_cache WHERE cache_key=?", (key,)).fetchone()
        if row is None:
            return None
        return json.loads(row[0]), float(row[1])

    def put(self, key: str, payload: dict[str, Any], elapsed: float):
        self.db.execute("INSERT OR REPLACE INTO jev_cache(cache_key,payload,elapsed_s) VALUES(?,?,?)",
                        (key, canonical(payload), float(elapsed)))
        self.db.commit()

    def close(self):
        self.db.close()


def extract_records(rows_by_split, labels, out: Path, endpoint: str, model: str,
                    mode: str, token_env: str | None, workers: int, timeout: float,
                    retries: int):
    command_started = time.perf_counter()
    questions = question_set(labels, mode)
    token = os.environ.get(token_env) if token_env else None
    if token_env and not token:
        raise RuntimeError(f"Environment variable {token_env!r} is empty or unset.")
    cache = ResultCache(out / "jev_cache.sqlite")
    unique: dict[str, str] = {}
    for rows in rows_by_split.values():
        for row in rows:
            key = cache_key(row["text"], model, questions)
            if cache.get(key) is None:
                unique[key] = row["text"]

    print(f"Unique uncached texts: {len(unique):,}; cached requests reused: "
          f"{sum(len(v) for v in rows_by_split.values()) - len(unique):,}")
    if unique:
        def task(item):
            key, text = item
            raw, elapsed = json_request(endpoint, model, text, questions, token, timeout, retries)
            return key, raw, elapsed
        try:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = [pool.submit(task, item) for item in unique.items()]
                for completed, future in enumerate(as_completed(futures), start=1):
                    key, raw, elapsed = future.result()
                    cache.put(key, raw, elapsed)
                    if completed % 100 == 0 or completed == len(futures):
                        print(f"Completed {completed:,}/{len(futures):,} uncached requests")
        except Exception:
            cache.close()
            raise

    records = {}
    for split, rows in rows_by_split.items():
        split_records = []
        for row in rows:
            key = cache_key(row["text"], model, questions)
            hit = cache.get(key)
            if hit is None:
                raise RuntimeError("Cache entry missing after extraction; rerun the command.")
            raw, elapsed = hit
            answers = response_answers(raw)
            split_records.append({**row, "cache_key": key, "answers": answers,
                                  "latency_s": elapsed})
        records[split] = split_records
    cache.close()

    out.mkdir(parents=True, exist_ok=True)
    latency_by_key = {}
    for split_records in records.values():
        for item in split_records:
            latency_by_key.setdefault(item["cache_key"], item["latency_s"])
    all_latencies_ms = [1000 * value for value in latency_by_key.values()]
    request_latencies_ms = [1000 * latency_by_key[key] for key in unique if key in latency_by_key]
    metadata = {"dataset": DATASET_ID, "model": model,
                "endpoint_host": urlsplit(endpoint).hostname,
                "question_mode": mode, "questions": list(questions),
                "question_schema_sha256": sha(canonical(questions)),
                "rows": {split: len(rows) for split, rows in rows_by_split.items()},
                "uncached_requests_this_command": len(unique),
                "rows_reused_or_deduplicated": sum(len(v) for v in rows_by_split.values()) - len(unique),
                "extraction_command_wall_seconds": time.perf_counter() - command_started,
                "response_latency_ms_unique_texts": {
                    "mean": float(np.mean(all_latencies_ms)) if all_latencies_ms else None,
                    "p50": float(np.percentile(all_latencies_ms, 50)) if all_latencies_ms else None,
                    "p95": float(np.percentile(all_latencies_ms, 95)) if all_latencies_ms else None,
                },
                "response_latency_ms_new_requests": {
                    "mean": float(np.mean(request_latencies_ms)) if request_latencies_ms else None,
                    "p50": float(np.percentile(request_latencies_ms, 50)) if request_latencies_ms else None,
                    "p95": float(np.percentile(request_latencies_ms, 95)) if request_latencies_ms else None,
                },
                "official_rows": labels}
    # Store host only, never a token or URL path that might contain credentials.
    metadata.pop("official_rows", None)
    (out / "jev_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_extraction_csvs(records, out, mode)
    return records, questions


def answer_noul(answer: dict[str, Any]) -> float:
    if "noul" not in answer:
        raise ValueError("Expected a Noul answer with a 'noul' probability")
    value = answer["noul"]
    if isinstance(value, dict):
        value = value.get("probability", value.get("value"))
    value = float(value)
    if not np.isfinite(value) or value < 0 or value > 1:
        raise ValueError(f"Noul score must be between 0 and 1, got {value}")
    return value


def answer_choice(answer: dict[str, Any]) -> str:
    choice = answer.get("choice")
    if choice is None:
        raise ValueError("Expected a Choice answer with a 'choice' field")
    return str(choice)


def answer_probabilities(answer: dict[str, Any]) -> dict[str, float]:
    values = answer.get("probabilities")
    if not isinstance(values, dict):
        return {}
    return {str(k): float(v) for k, v in values.items()}


def feature_ids():
    return [p["id"] for p in load_probes()]


def write_extraction_csvs(records, out: Path, mode: str):
    feature_names = feature_ids() if mode in ("both", "features") else []
    feature_path = out / "jev_features.csv"
    if feature_names:
        with feature_path.open("w", newline="", encoding="utf-8") as file:
            fields = ["split", "row", "label", "text", "text_sha256"] + feature_names
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for split_records in records.values():
                for item in split_records:
                    row = {"split": item["split"], "row": item["row"], "label": item["label"],
                           "text": item["text"], "text_sha256": sha(item["text"])}
                    for feature in feature_names:
                        row[feature] = answer_noul(item["answers"]["probe__" + feature])
                    writer.writerow(row)

    if mode in ("both", "direct"):
        with (out / "jev_predictions.csv").open("w", newline="", encoding="utf-8") as file:
            fields = ["split", "row", "label", "text", "prediction", "confidence", "probabilities_json", "latency_s"]
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for split_records in records.values():
                for item in split_records:
                    answer = item["answers"][DIRECT_ID]
                    probs = answer_probabilities(answer)
                    writer.writerow({"split": item["split"], "row": item["row"], "label": item["label"],
                                     "text": item["text"], "prediction": answer_choice(answer),
                                     "confidence": answer.get("confidence", ""),
                                     "probabilities_json": canonical(probs), "latency_s": item["latency_s"]})


def ensure_out(path: str) -> Path:
    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def classifier_metrics(y_true, y_pred):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))}


def common_confusions(y_true, y_pred, limit=20):
    pairs = Counter((str(y), str(p)) for y, p in zip(y_true, y_pred) if y != p)
    return [{"true": truth, "predicted": pred, "count": count}
            for (truth, pred), count in pairs.most_common(limit)]


def save_predictions(path, split_rows, y_pred):
    with path.open("w", newline="", encoding="utf-8") as file:
        fields = ["row", "true_label", "predicted_label", "text"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row, pred in zip(split_rows, y_pred):
            writer.writerow({"row": row["row"], "true_label": row["label"],
                             "predicted_label": str(pred), "text": row["text"]})


def load_metrics(path: Path):
    if not path.exists():
        return {"dataset": DATASET_ID, "primary_metric": "macro_f1", "results": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_metrics(out: Path, metrics):
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def vectorize(train_rows, test_rows):
    # Fit both text views on training data only.
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 strip_accents="unicode", max_features=300_000)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                                 sublinear_tf=True, max_features=300_000)),
    ], n_jobs=1)
    x_train = features.fit_transform([row["text"] for row in train_rows])
    x_test = features.transform([row["text"] for row in test_rows])
    return x_train, x_test


def run_classical(args):
    rows, labels, official_counts = load_dataset_rows(args.max_rows_per_split)
    train_rows, test_rows = rows["train"], rows["test"]
    if len(train_rows) < 2 or len(test_rows) < 1:
        raise RuntimeError("Need at least two train rows and one test row.")
    out = ensure_out(args.out)
    print(f"Loading word and character TF-IDF: train={len(train_rows)}, test={len(test_rows)}")
    x_train, x_test = vectorize(train_rows, test_rows)
    y_train = np.asarray([row["label"] for row in train_rows])
    y_test = np.asarray([row["label"] for row in test_rows])
    models = {
        "tfidf_logistic_regression": LogisticRegression(C=4.0, max_iter=1000, solver="lbfgs"),
        "tfidf_linear_svm": LinearSVC(C=1.0),
        "tfidf_complement_nb": ComplementNB(alpha=0.5),
    }
    metrics = load_metrics(out / "metrics.json")
    metrics["dataset_rows"] = {"official": official_counts,
                               "evaluated": {k: len(v) for k, v in rows.items()},
                               "smoke_test": bool(args.max_rows_per_split)}
    metrics["classical"] = {}
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_s = time.perf_counter() - started
        started = time.perf_counter()
        pred = model.predict(x_test)
        predict_s = time.perf_counter() - started
        scores = classifier_metrics(y_test, pred)
        scores.update({"fit_seconds": fit_s, "test_predict_seconds": predict_s,
                       "test_predict_ms_per_row": 1000 * predict_s / len(test_rows),
                       "feature_count": int(x_train.shape[1])})
        metrics["classical"][name] = scores
        save_predictions(out / f"{name}_predictions.csv", test_rows, pred)
        metrics.setdefault("diagnostics", {})[name] = common_confusions(y_test, pred)
        print(f"{name}: macro-F1={scores['macro_f1']:.4f} accuracy={scores['accuracy']:.4f}")
    write_metrics(out, metrics)


def run_extract(args):
    rows, labels, _ = load_dataset_rows(args.max_rows_per_split)
    out = ensure_out(args.out)
    mode = args.mode
    try:
        extract_records(rows, labels, out, args.endpoint, args.model, mode,
                        args.token_env, args.workers, args.timeout, args.retries)
    except Exception as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        raise
    print("Jev extraction files and SQLite cache written to", out)


def run_evaluate_jev(args):
    rows, labels, official_counts = load_dataset_rows(args.max_rows_per_split)
    out = ensure_out(args.out)
    mode = args.mode
    questions = question_set(labels, mode)
    if not (out / "jev_cache.sqlite").exists():
        raise RuntimeError("No Jev cache found. Run extract-jev first.")
    cache = ResultCache(out / "jev_cache.sqlite")
    records = {}
    for split, split_rows in rows.items():
        records[split] = []
        for row in split_rows:
            key = cache_key(row["text"], args.model, questions)
            hit = cache.get(key)
            if hit is None:
                cache.close()
                raise RuntimeError(f"Missing cached {split} row {row['row']}. Run extract-jev with the same model and mode first.")
            raw, latency = hit
            records[split].append({**row, "answers": response_answers(raw), "latency_s": latency})
    cache.close()
    write_extraction_csvs(records, out, mode)

    metrics = load_metrics(out / "metrics.json")
    metrics["dataset_rows"] = {"official": official_counts,
                               "evaluated": {k: len(v) for k, v in rows.items()},
                               "smoke_test": bool(args.max_rows_per_split)}
    metrics["semantic_feature_models"] = {}
    if mode in ("both", "features"):
        feature_names = feature_ids()
        x_train = np.asarray([[answer_noul(item["answers"]["probe__" + name])
                               for name in feature_names] for item in records["train"]], dtype=np.float32)
        x_test = np.asarray([[answer_noul(item["answers"]["probe__" + name])
                              for name in feature_names] for item in records["test"]], dtype=np.float32)
        y_train = np.asarray([item["label"] for item in records["train"]])
        y_test = np.asarray([item["label"] for item in records["test"]])
        models = {
            "jev_features_logistic_regression": LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"),
            "jev_features_linear_svm": LinearSVC(C=1.0),
            "jev_features_extra_trees": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                                              max_features="sqrt", n_jobs=-1,
                                                              random_state=77),
        }
        for name, model in models.items():
            started = time.perf_counter(); model.fit(x_train, y_train); fit_s = time.perf_counter() - started
            started = time.perf_counter(); pred = model.predict(x_test); predict_s = time.perf_counter() - started
            scores = classifier_metrics(y_test, pred)
            scores.update({"fit_seconds": fit_s, "test_predict_seconds": predict_s,
                           "test_predict_ms_per_row": 1000 * predict_s / len(records["test"]),
                           "semantic_probe_count": len(feature_names)})
            metrics["semantic_feature_models"][name] = scores
            metrics.setdefault("diagnostics", {})[name] = common_confusions(y_test, pred)
            save_predictions(out / f"{name}_predictions.csv", records["test"], pred)
            print(f"{name}: macro-F1={scores['macro_f1']:.4f} accuracy={scores['accuracy']:.4f}")

    if mode in ("both", "direct"):
        bysplit = {split: [item for item in records[split] if item["split"] == split]
                   for split in records}
        y_test = [item["label"] for item in bysplit["test"]]
        direct = [answer_choice(item["answers"][DIRECT_ID]) for item in bysplit["test"]]
        scores = classifier_metrics(y_test, direct)
        top3 = []
        for item in bysplit["test"]:
            probabilities = answer_probabilities(item["answers"][DIRECT_ID])
            ranked = [key for key, _ in sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)[:3]]
            top3.append(item["label"] in ranked if ranked else None)
        if any(value is not None for value in top3):
            scores["top3_accuracy"] = float(np.mean([value for value in top3 if value is not None]))
        latencies = [item["latency_s"] for item in bysplit["test"]]
        scores.update({"test_rows": len(bysplit["test"]), "request_latency_mean_ms": 1000 * float(np.mean(latencies)),
                       "request_latency_p50_ms": 1000 * float(np.percentile(latencies, 50)),
                       "request_latency_p95_ms": 1000 * float(np.percentile(latencies, 95))})
        metrics["jev_direct_choice"] = scores
        metrics.setdefault("diagnostics", {})["jev_direct_choice"] = common_confusions(y_test, direct)
        print(f"jev_direct_choice: macro-F1={scores['macro_f1']:.4f} accuracy={scores['accuracy']:.4f}")

    metrics["primary_metric"] = "macro_f1"
    write_metrics(out, metrics)
    print("Jev metrics written to", out / "metrics.json")


def parser():
    p = argparse.ArgumentParser(description="BANKING77 classical ML and Jev baseline runner")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("classical", help="run TF-IDF classical baselines")
    c.add_argument("--out", default="runs/banking77")
    c.add_argument("--max-rows-per-split", type=int, default=None,
                   help="smoke test only; truncated-split scores are not benchmark results")
    c.set_defaults(func=run_classical)

    e = sub.add_parser("extract-jev", help="query a System One endpoint and cache outputs")
    e.add_argument("--endpoint", required=True, help="complete URL ending in /v1/systemone")
    e.add_argument("--model", default="parable/tinyjev")
    e.add_argument("--mode", choices=("both", "direct", "features"), default="both")
    e.add_argument("--token-env", default=None, help="name of env var holding an optional Bearer token")
    e.add_argument("--workers", type=int, default=1)
    e.add_argument("--timeout", type=float, default=180)
    e.add_argument("--retries", type=int, default=4)
    e.add_argument("--out", default="runs/banking77")
    e.add_argument("--max-rows-per-split", type=int, default=None)
    e.set_defaults(func=run_extract)

    v = sub.add_parser("evaluate-jev", help="fit semantic-feature learners and score direct Jev")
    v.add_argument("--model", default="parable/tinyjev", help="must match the extraction model")
    v.add_argument("--mode", choices=("both", "direct", "features"), default="both",
                   help="must match the extraction mode")
    v.add_argument("--out", default="runs/banking77")
    v.add_argument("--max-rows-per-split", type=int, default=None)
    v.set_defaults(func=run_evaluate_jev)
    return p


def main():
    args = parser().parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
