import re
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
TEXT_LABELS = ["benign", "urgency", "scarcity", "social_proof", "misedirection"]
STRUCTURAL_FEATURES = [
    "is_button",
    "is_checkbox",
    "is_checked",
    "is_popup",
    "is_optional_addon",
    "contains_subscription_word",
    "contains_price",
    "area_ratio",
    "font_size_ratio",
    "contrast_ratio",
    "accept_reject_area_ratio",
    "opacity",
    "text_change_count",
    "animation_count",
]


def parse_color(value: str) -> tuple[int, int, int, float]:
    numbers = re.findall(r"[\d.]+", value or "")
    if len(numbers) < 3:
        return 255, 255, 255, 1.0
    alpha = float(numbers[3]) if len(numbers) > 3 else 1.0
    return int(float(numbers[0])), int(float(numbers[1])), int(float(numbers[2])), alpha


def contrast_ratio(foreground: str, background: str) -> float:
    def luminance(color: tuple[int, int, int, float]) -> float:
        channels = []
        for value in color[:3]:
            channel = value / 255
            channels.append(channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4)
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    first = luminance(parse_color(foreground))
    second = luminance(parse_color(background))
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def structural_features(element: dict, viewport: dict | None = None) -> dict[str, float]:
    viewport = viewport or {"width": 1365, "height": 768}
    text = element.get("text", "").lower()
    bbox = element.get("bbox", [0, 0, 0, 0])
    area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
    viewport_area = max(1, viewport["width"] * viewport["height"])
    is_button = element.get("tag") == "button" or element.get("role") == "button"
    return {
        "is_button": float(is_button),
        "is_checkbox": float(element.get("input_type") == "checkbox"),
        "is_checked": float(bool(element.get("checked"))),
        "is_popup": float(element.get("role") == "dialog" or element.get("tag") == "dialog" or element.get("z_index", 0) > 10),
        "is_optional_addon": float(bool(re.search(r"add|optional|protect|insurance|club", text))),
        "contains_subscription_word": float(bool(re.search(r"subscribe|subscription|monthly|renews?|recurring", text))),
        "contains_price": float(bool(re.search(r"(?:₹|\$|£|€)\s?\d|\d[.,]\d{2}", text))),
        "area_ratio": area / viewport_area,
        "font_size_ratio": float(element.get("font_size", 16)) / 16,
        "contrast_ratio": min(21, contrast_ratio(element.get("color", ""), element.get("background_color", ""))),
        "accept_reject_area_ratio": float(element.get("accept_reject_area_ratio", 1)),
        "opacity": float(element.get("opacity", 1)),
        "text_change_count": float(element.get("text_change_count", 0)),
        "animation_count": float(element.get("animation_count", 0)),
    }


def postprocess_text_label(label: str, text: str) -> str:
    guilt_phrases = [
        r"no thanks?.*(prefer|like|want)",
        r"i (?:do not|don't) want to protect",
        r"no,? i hate saving",
        r"i (?:prefer|choose).*(paying|full price|miss)",
    ]
    if label == "misedirection" and any(re.search(pattern, text.lower()) for pattern in guilt_phrases):
        return "confirmshaming"
    return label


class TextClassifier:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.engine = "keyword rules"
        self.model = None
        self.tokenizer = None
        self.labels = TEXT_LABELS
        distilbert_path = model_dir / "distilbert_text"
        baseline_path = model_dir / "text_baseline.joblib"
        if (distilbert_path / "config.json").exists():
            self._load_distilbert(distilbert_path)
        elif baseline_path.exists():
            saved = joblib.load(baseline_path)
            self.model = saved["model"] if isinstance(saved, dict) else saved
            self.labels = saved.get("labels", TEXT_LABELS) if isinstance(saved, dict) else TEXT_LABELS
            self.engine = "TF-IDF + Logistic Regression"

    def _load_distilbert(self, model_path: Path) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.labels = [self.model.config.id2label[index] for index in range(len(self.model.config.id2label))]
        self.engine = "fine-tuned DistilBERT"

    def classify(self, texts: list[str]) -> list[tuple[str, float]]:
        if not texts or self.model is None:
            return [("benign", 0.0) for _ in texts]
        if self.engine.startswith("TF-IDF"):
            probabilities = self.model.predict_proba(texts)
            classes = list(self.model.classes_)
            return [(classes[row.argmax()], float(row.max())) for row in probabilities]
        return self._classify_distilbert(texts)

    def _classify_distilbert(self, texts: list[str]) -> list[tuple[str, float]]:
        import torch

        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            probabilities = self.model(**encoded).logits.softmax(dim=-1)
        answers = []
        for row in probabilities:
            index = int(row.argmax())
            answers.append((self.labels[index], float(row[index])))
        return answers


def load_structural_model(model_dir: Path = MODEL_DIR):
    path = model_dir / "structural_model.joblib"
    return joblib.load(path) if path.exists() else None


def predict_structural(saved_model: dict | None, features: dict[str, float]) -> tuple[str, float]:
    if not saved_model:
        return "benign", 0.0
    names = saved_model.get("feature_names", STRUCTURAL_FEATURES)
    row = pd.DataFrame([[features[name] for name in names]], columns=names)
    model = saved_model["model"]
    probabilities = model.predict_proba(row)[0]
    index = int(probabilities.argmax())
    return str(model.classes_[index]), float(probabilities[index])
