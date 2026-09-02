"""Generate controlled synthetic DOM/layout examples for the Random Forest."""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES = [
    "is_button", "is_checkbox", "is_checked", "is_popup", "is_optional_addon",
    "contains_subscription_word", "contains_price", "area_ratio", "font_size_ratio",
    "contrast_ratio", "accept_reject_area_ratio", "opacity", "text_change_count", "animation_count",
]
LABELS = ["benign", "preselected_addon", "visual_interference", "popup_pressure", "subscription_warning"]


def clipped(rng, mean, spread, low=0, high=21):
    return float(np.clip(rng.normal(mean, spread), low, high))


def base_row(rng) -> dict:
    return {
        "is_button": rng.integers(0, 2), "is_checkbox": 0, "is_checked": 0, "is_popup": 0,
        "is_optional_addon": 0, "contains_subscription_word": 0, "contains_price": rng.integers(0, 2),
        "area_ratio": clipped(rng, 0.025, 0.015, 0.001, 0.2), "font_size_ratio": clipped(rng, 1, 0.18, 0.6, 2.5),
        "contrast_ratio": clipped(rng, 6, 2, 1, 18), "accept_reject_area_ratio": clipped(rng, 1.1, 0.3, 0.5, 2),
        "opacity": clipped(rng, 1, 0.04, 0.75, 1), "text_change_count": 0, "animation_count": 0,
    }


def labelled_row(label: str, rng) -> dict:
    row = base_row(rng)
    if label == "preselected_addon":
        row.update(is_button=0, is_checkbox=1, is_checked=1, is_optional_addon=1, contains_price=1)
        row["contains_subscription_word"] = rng.integers(0, 2)
    elif label == "visual_interference":
        row.update(is_button=1, accept_reject_area_ratio=clipped(rng, 5, 1.8, 2.2, 11), opacity=clipped(rng, 0.6, 0.18, 0.2, 1))
    elif label == "popup_pressure":
        row.update(is_popup=1, area_ratio=clipped(rng, 0.2, 0.08, 0.04, 0.5), animation_count=int(rng.integers(0, 3)))
        row["text_change_count"] = int(rng.integers(0, 3))
    elif label == "subscription_warning":
        row.update(contains_subscription_word=1, contains_price=1, is_optional_addon=int(rng.integers(0, 2)))
    return row


def build_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for label in LABELS:
        for template_index in range(12):
            split = "train" if template_index < 8 else "validation" if template_index < 10 else "test"
            template_id = f"{label}_{template_index:02d}"
            for variation in range(5):
                rows.append({"template_id": template_id, "variation": variation, "split": split, "label": label, **labelled_row(label, rng)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = build_data()
    if data.groupby("template_id")["split"].nunique().max() != 1:
        raise RuntimeError("A structural template leaked between splits.")
    output = BASE_DIR / "data" / "structural_examples.csv"
    data.to_csv(output, index=False)
    print(f"Wrote {len(data)} controlled synthetic rows to {output}")
