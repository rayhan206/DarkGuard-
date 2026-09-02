"""Train and evaluate the grouped TF-IDF text baseline."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parents[1]
LABELS = ["benign", "urgency", "scarcity", "social_proof", "misedirection"]


def metrics_for(y_true, y_pred) -> dict:
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "per_class": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "labels": LABELS,
    }


if __name__ == "__main__":
    data = pd.read_csv(BASE_DIR / "data" / "text_dataset.csv")
    train = data[data["split"] == "train"]
    validation = data[data["split"] == "validation"]
    test = data[data["split"] == "test"]
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=20_000)),
        ("classifier", LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=42)),
    ])
    model.fit(train["text"], train["category"])
    results = {
        "model": "TF-IDF (1-2 grams) + class-balanced Logistic Regression",
        "split_method": "page_id grouped 70/15/15",
        "validation": metrics_for(validation["category"], model.predict(validation["text"])),
        "test": metrics_for(test["category"], model.predict(test["text"])),
    }
    (BASE_DIR / "models").mkdir(exist_ok=True)
    (BASE_DIR / "results").mkdir(exist_ok=True)
    joblib.dump({"model": model, "labels": LABELS, "source": "EC dark-pattern dataset"}, BASE_DIR / "models" / "text_baseline.joblib")
    (BASE_DIR / "results" / "text_baseline_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
