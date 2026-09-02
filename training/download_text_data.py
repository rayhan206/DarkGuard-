"""Download and prepare the public text dataset with page-grouped splits."""

import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_URL = "https://raw.githubusercontent.com/yamanalab/ec-darkpattern/refs/heads/master/dataset/dataset.tsv"
CATEGORY_MAP = {
    "Not Dark Pattern": "benign",
    "Urgency": "urgency",
    "Scarcity": "scarcity",
    "Social Proof": "social_proof",
    "Misdirection": "misedirection",
}


def grouped_splits(data: pd.DataFrame) -> pd.Series:
    first_split = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42)
    train_indices, remaining_indices = next(first_split.split(data, groups=data["page_id"]))
    remaining = data.iloc[remaining_indices]
    second_split = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
    validation_relative, test_relative = next(second_split.split(remaining, groups=remaining["page_id"]))
    split = pd.Series(index=data.index, dtype="object")
    split.iloc[train_indices] = "train"
    split.iloc[remaining_indices[validation_relative]] = "validation"
    split.iloc[remaining_indices[test_relative]] = "test"
    return split


def prepare_data() -> pd.DataFrame:
    with urlopen(DATA_URL, timeout=30) as response:
        data = pd.read_csv(response, sep="\t")
    data = data.rename(columns={"Pattern Category": "original_category"})
    data["text"] = data["text"].fillna("").astype(str).str.strip()
    data = data[data["original_category"].isin(CATEGORY_MAP)].copy()
    data = data[data["text"] != ""].drop_duplicates(subset=["text"], keep="first")
    data["category"] = data["original_category"].map(CATEGORY_MAP)
    data["split"] = grouped_splits(data)
    return data[["page_id", "text", "category", "split"]].reset_index(drop=True)


def verify_no_group_leakage(data: pd.DataFrame) -> None:
    counts = data.groupby("page_id")["split"].nunique()
    if counts.max() != 1:
        raise RuntimeError("Grouped split failed: at least one page appears in multiple splits.")


if __name__ == "__main__":
    prepared = prepare_data()
    verify_no_group_leakage(prepared)
    data_path = BASE_DIR / "data" / "text_dataset.csv"
    result_path = BASE_DIR / "results" / "data_split.json"
    result_path.parent.mkdir(exist_ok=True)
    prepared.to_csv(data_path, index=False)
    summary = {
        "source": DATA_URL,
        "duplicate_rule": "exact duplicate after trimming surrounding whitespace",
        "unsupported_source_categories_removed": ["Obstruction", "Sneaking", "Forced Action"],
        "rows": len(prepared),
        "unique_pages": int(prepared["page_id"].nunique()),
        "rows_by_split": prepared["split"].value_counts().to_dict(),
        "pages_by_split": prepared.groupby("split")["page_id"].nunique().to_dict(),
        "class_counts": prepared["category"].value_counts().to_dict(),
        "group_leakage_pages": 0,
    }
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
