"""Fine-tune DistilBERT on the page-grouped text data."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parents[1]
LABELS = ["benign", "urgency", "scarcity", "social_proof", "misedirection"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


class EncodedTexts(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer):
        self.labels = [LABEL_TO_ID[label] for label in frame["category"]]
        self.tokens = tokenizer(frame["text"].tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {name: values[index] for name, values in self.tokens.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def balance_training_rows(train: pd.DataFrame) -> pd.DataFrame:
    target_count = train["category"].value_counts().max()
    balanced = []
    for _, group in train.groupby("category"):
        balanced.append(group.sample(target_count, replace=len(group) < target_count, random_state=42))
    return pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)


def predict(model, dataset, device) -> list[str]:
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=32):
            labels = batch.pop("labels")
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            predictions.extend(LABELS[index] for index in logits.argmax(dim=1).cpu().tolist())
            batch["labels"] = labels
    return predictions


def evaluate(model, dataset, truth, device) -> dict:
    predictions = predict(model, dataset, device)
    return {
        "macro_f1": f1_score(truth, predictions, labels=LABELS, average="macro", zero_division=0),
        "per_class": classification_report(truth, predictions, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=LABELS).tolist(),
        "labels": LABELS,
    }


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    data = pd.read_csv(BASE_DIR / "data" / "text_dataset.csv")
    train = balance_training_rows(data[data["split"] == "train"])
    validation = data[data["split"] == "validation"].reset_index(drop=True)
    test = data[data["split"] == "test"].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_data = EncodedTexts(train, tokenizer)
    validation_data = EncodedTexts(validation, tokenizer)
    test_data = EncodedTexts(test, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id=LABEL_TO_ID,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loader = DataLoader(train_data, batch_size=16, shuffle=True)
    output_dir = BASE_DIR / "models" / "distilbert_text"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    for epoch in range(3):
        model.train()
        for batch in loader:
            optimiser.zero_grad()
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimiser.step()
        validation_metrics = evaluate(model, validation_data, validation["category"].tolist(), device)
        print(f"epoch={epoch + 1} validation_macro_f1={validation_metrics['macro_f1']:.4f}")
        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = validation_metrics["macro_f1"]
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
    best_model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    results = {
        "model": "distilbert-base-uncased fine-tuned for 3 epochs",
        "device": str(device),
        "maximum_token_length": 128,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "imbalance_handling": "training-only random oversampling to the largest class; grouped splits created before oversampling",
        "best_validation_macro_f1": best_f1,
        "test": evaluate(best_model, test_data, test["category"].tolist(), device),
    }
    (output_dir / "label_mapping.json").write_text(json.dumps(LABEL_TO_ID, indent=2), encoding="utf-8")
    (BASE_DIR / "results" / "distilbert_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
