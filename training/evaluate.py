"""Create a genuine comparison report from available trained components."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from darkguard.models import TextClassifier  # noqa: E402
from darkguard.pipeline import text_rule_detection  # noqa: E402


LABELS = ["benign", "urgency", "scarcity", "social_proof", "misedirection"]


def text_metrics(truth, prediction) -> dict:
    return {
        "macro_f1": f1_score(truth, prediction, labels=LABELS, average="macro", zero_division=0),
        "per_class_f1": {label: values["f1-score"] for label, values in classification_report(truth, prediction, labels=LABELS, output_dict=True, zero_division=0).items() if label in LABELS},
    }


def rule_predictions(texts: list[str]) -> list[str]:
    reverse = {"potential_urgency": "urgency", "potential_scarcity": "scarcity"}
    answers = []
    for text in texts:
        detection = text_rule_detection({"text": text, "bbox": [0, 0, 1, 1]})
        label = reverse.get(detection["label"], detection["label"]) if detection else "benign"
        answers.append("misedirection" if label == "confirmshaming" else label)
    return answers


def request_json(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload else None
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def ensure_server(base_url: str):
    try:
        request_json(f"{base_url}/health")
        return None
    except URLError:
        command = [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8765"]
        process = subprocess.Popen(command, cwd=BASE_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(30):
            time.sleep(0.25)
            try:
                request_json("http://127.0.0.1:8765/health")
                return process
            except URLError:
                continue
        process.terminate()
        raise RuntimeError("The local evaluation server did not start.")


def evaluate_demos(base_url: str) -> dict:
    expected = {
        "safe": None, "urgency": "potential_urgency", "scarcity": "potential_scarcity",
        "social_proof": "social_proof", "confirmshaming": "confirmshaming",
        "preselection": "preselected_addon", "visual_interference": "visual_interference",
        "dynamic_timer": "potential_urgency",
    }
    rows = []
    for name, expected_label in expected.items():
        seconds = 1.2 if name in {"social_proof", "dynamic_timer"} else 0.2
        result = request_json(f"{base_url}/api/analyse/url", {"url": f"{base_url}/demo/{name}", "observe_seconds": seconds})
        labels = [item["label"] for item in result["detections"]]
        passed = not labels if expected_label is None else expected_label in labels
        rows.append({"template": name, "expected": expected_label or "no cue", "detected": labels, "passed": passed})
    return {"template_pass_rate": sum(row["passed"] for row in rows) / len(rows), "templates": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    arguments = parser.parse_args()
    data = pd.read_csv(BASE_DIR / "data" / "text_dataset.csv")
    test = data[data["split"] == "test"]
    report = {"text_test_rows": len(test), "keyword_rules": text_metrics(test["category"], rule_predictions(test["text"].tolist()))}
    baseline = joblib.load(BASE_DIR / "models" / "text_baseline.joblib")["model"]
    report["tfidf_baseline"] = text_metrics(test["category"], baseline.predict(test["text"]))
    if (BASE_DIR / "models" / "distilbert_text" / "config.json").exists():
        classifier = TextClassifier()
        report["distilbert"] = text_metrics(test["category"], [label for label, _ in classifier.classify(test["text"].tolist())])
    structural_path = BASE_DIR / "results" / "structural_metrics.json"
    if structural_path.exists():
        structural = json.loads(structural_path.read_text(encoding="utf-8"))
        report["random_forest_synthetic"] = {"warning": structural["data_warning"], **structural["test"]}
    server = ensure_server(arguments.base_url)
    active_base_url = arguments.base_url if server is None else "http://127.0.0.1:8765"
    try:
        report["fused_demo_templates"] = evaluate_demos(active_base_url)
    finally:
        if server:
            server.terminate()
            server.wait(timeout=10)
    (BASE_DIR / "results" / "evaluation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
