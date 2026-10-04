import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from banking77_baseline import (
    ResultCache, answer_choice, answer_noul, cache_key, extract_records,
    json_request, question_set, response_answers,
)


LABELS = ["card_arrival", "failed_transfer"]


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.payload


def make_response(request):
    body = json.loads(request.data)
    answers = {}
    for key, question in body["questions"].items():
        if question["type"] == "choice":
            choice = next(iter(question["criteria"]))
            answers[key] = {"choice": choice, "probabilities": {choice: 0.8}, "confidence": 0.6}
        else:
            answers[key] = {"noul": 0.75}
    return FakeResponse(json.dumps({"states": [{"answers": answers}]}).encode())


class BaselineTests(unittest.TestCase):
    def test_tinyjev_question_schema(self):
        questions = question_set(LABELS, "both")
        self.assertEqual(len(questions), 17)
        self.assertEqual(questions["direct_intent"]["criteria"]["card_arrival"], "card arrival")
        self.assertEqual(questions["probe__asks_for_information"]["type"], "noul")

    def test_parse_systemone_answers(self):
        answers = response_answers({"states": [{"answers": {"x": {"noul": 0.75}, "y": {"choice": "x"}}}]})
        self.assertEqual(answer_noul(answers["x"]), 0.75)
        self.assertEqual(answer_choice(answers["y"]), "x")
        self.assertEqual(response_answers({"answers": {"x": {"choice": "ok"}}})["x"]["choice"], "ok")

    def test_reject_invalid_noul(self):
        with self.assertRaises(ValueError):
            answer_noul({"noul": 1.2})

    def test_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ResultCache(__import__("pathlib").Path(tmp) / "cache.sqlite")
            key = cache_key("text", "model", {"q": {"type": "noul"}})
            self.assertIsNone(cache.get(key))
            cache.put(key, {"answers": {"q": {"noul": 0.5}}}, 0.12)
            payload, elapsed = cache.get(key)
            self.assertEqual(payload["answers"]["q"]["noul"], 0.5)
            self.assertAlmostEqual(elapsed, 0.12)
            cache.close()

    def test_http_smoke_against_systemone_mock(self):
        questions = question_set(LABELS, "both")
        with patch("banking77_baseline.urllib.request.urlopen", side_effect=lambda request, timeout: make_response(request)):
            payload, elapsed = json_request("http://127.0.0.1:11434/v1/systemone",
                                             "mock-model", "example", questions, None, 5, 0)
        answers = response_answers(payload)
        self.assertEqual(answer_choice(answers["direct_intent"]), "card_arrival")
        self.assertEqual(answer_noul(answers["probe__asks_for_information"]), 0.75)
        self.assertGreaterEqual(elapsed, 0)

    def test_extract_writes_resumable_outputs(self):
        rows = {
            "train": [
                {"split": "train", "row": 0, "text": "card issue", "label": "alpha"},
                {"split": "train", "row": 1, "text": "transfer pending", "label": "beta"},
            ],
            "test": [
                {"split": "test", "row": 0, "text": "cash withdrawal", "label": "alpha"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch("banking77_baseline.urllib.request.urlopen", side_effect=lambda request, timeout: make_response(request)) as urlopen:
                records, questions = extract_records(rows, ["alpha", "beta"], out,
                    "http://localhost:11434/v1/systemone", "mock-model", "both", None, 1, 5, 0)
                self.assertEqual(urlopen.call_count, 3)
                self.assertEqual(len(records["train"]), 2)
                self.assertEqual(answer_noul(records["test"][0]["answers"]["probe__asks_for_information"]), 0.75)
            self.assertTrue((out / "jev_features.csv").exists())
            self.assertTrue((out / "jev_predictions.csv").exists())
            manifest = json.loads((out / "jev_manifest.json").read_text())
            self.assertEqual(manifest["endpoint_host"], "localhost")
            self.assertNotIn("official_rows", manifest)
            self.assertEqual(len(questions), 17)
            with patch("banking77_baseline.urllib.request.urlopen", side_effect=AssertionError("cache missed")) as urlopen:
                extract_records(rows, ["alpha", "beta"], out,
                    "http://localhost:11434/v1/systemone", "mock-model", "both", None, 1, 5, 0)
                self.assertEqual(urlopen.call_count, 0)


if __name__ == "__main__":
    unittest.main()
