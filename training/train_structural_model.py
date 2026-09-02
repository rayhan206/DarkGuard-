"""Train the small structural Random Forest on controlled synthetic templates."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

BASE_DIR = Path(__file__).resolve().parents[1]
STRUCTURAL_FEATURES = [
    "is_button", "is_checkbox", "is_checked", "is_popup", "is_optional_addon",
    "contains_subscription_word", "contains_price", "area_ratio", "font_size_ratio",
    "contrast_ratio", "accept_reject_area_ratio", "opacity", "text_change_count", "animation_count",
]


def evaluate(model, frame: pd.DataFrame) -> dict:
    prediction = model.predict(frame[STRUCTURAL_FEATURES])
    labels = list(model.classes_)
    return {
        "macro_f1": f1_score(frame["label"], prediction, labels=labels, average="macro", zero_division=0),
        "per_class": classification_report(frame["label"], prediction, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(frame["label"], prediction, labels=labels).tolist(),
        "labels": labels,
    }


if __name__ == "__main__":
    data = pd.read_csv(BASE_DIR / "data" / "structural_examples.csv")
    train = data[data["split"] == "train"]
    validation = data[data["split"] == "validation"]
    test = data[data["split"] == "test"]
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight="balanced")
    model.fit(train[STRUCTURAL_FEATURES], train["label"])
    importance = sorted(zip(STRUCTURAL_FEATURES, model.feature_importances_), key=lambda pair: pair[1], reverse=True)
    results = {
        "model": "RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)",
        "data_warning": "Controlled synthetic template results; not real-world performance evidence.",
        "split_method": "template_id grouped: 40 train, 10 validation, 10 test templates",
        "validation": evaluate(model, validation),
        "test": evaluate(model, test),
        "feature_importance": [{"feature": name, "importance": value} for name, value in importance],
    }
    (BASE_DIR / "models").mkdir(exist_ok=True)
    (BASE_DIR / "results").mkdir(exist_ok=True)
    joblib.dump({"model": model, "feature_names": STRUCTURAL_FEATURES, "synthetic": True}, BASE_DIR / "models" / "structural_model.joblib")
    (BASE_DIR / "results" / "structural_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(results["feature_importance"]).to_csv(BASE_DIR / "results" / "structural_feature_importance.csv", index=False)
    print(json.dumps(results, indent=2))
