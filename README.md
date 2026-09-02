# DarkGuard

DarkGuard is a small educational AI application that screens e-commerce interfaces for **potential dark-pattern cues**. It accepts a public URL or an uploaded screenshot, draws labelled regions, and explains the evidence in plain language. It does not decide whether a company is deceptive or breaking the law.

## Architecture

URL mode follows one readable path:

```text
URL → Playwright rendered page → DOM, styles, states, mutations, screenshot
    → fine-tuned DistilBERT (language) + Random Forest (structure) + observable rules
    → transparent overlap fusion → annotated screenshot + JSON report
```

Screenshot mode uses:

```text
image → EasyOCR text and coordinates → DistilBERT + text rules
      → annotated screenshot + JSON report
```

The text engine fallback order is honest and visible in the UI: saved DistilBERT, saved TF-IDF + Logistic Regression, then keyword rules. The structural engine similarly reports Random Forest or structural rules. No API key, paid service, database, or external website is needed for the demonstration.

## Installation

Python 3.11 is recommended. From PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m playwright install chromium
```

If PowerShell blocks activation, either use `Set-ExecutionPolicy -Scope Process Bypass` for that terminal or run the environment's Python directly. From Git Bash, activate with `source .venv/Scripts/activate`.

EasyOCR downloads its English recognition weights on the first screenshot request. That first request is therefore slower. Playwright creates a fresh cookie-free browser context for each URL analysis and does not click or submit anything.

## Run

```powershell
python app.py
```

Open <http://127.0.0.1:8000>. Useful local pages are:

- <http://127.0.0.1:8000/demo/safe>
- <http://127.0.0.1:8000/demo/urgency>
- <http://127.0.0.1:8000/demo/scarcity>
- <http://127.0.0.1:8000/demo/social_proof>
- <http://127.0.0.1:8000/demo/confirmshaming>
- <http://127.0.0.1:8000/demo/preselection>
- <http://127.0.0.1:8000/demo/visual_interference>
- <http://127.0.0.1:8000/demo/dynamic_timer>

The demo-only `data-darkguard-label` attributes are used to describe controlled examples when building data. The normal analyser never reads them.

## Train and evaluate

Run these commands from the project root, in order:

```powershell
python training/download_text_data.py
python training/train_text_baseline.py
python training/train_distilbert.py
python training/build_structural_data.py
python training/train_structural_model.py
python training/evaluate.py
```

DistilBERT training uses three epochs, length 128, batch size 16, learning rate `2e-5`, seed 42, and CUDA when available. It oversamples only the training partition to address class imbalance. The download script removes exact duplicate text and assigns rows to 70/15/15 splits with `page_id` grouping before any training.

The checked-in results were genuinely produced in this repository:

| Component | Evaluation | Macro-F1 |
|---|---|---:|
| Keyword rules | 347-row grouped text test set | 0.6298 |
| TF-IDF + Logistic Regression | same grouped text test set | 0.9213 |
| Fine-tuned DistilBERT | same grouped text test set | 0.9663 |
| Random Forest | 50 controlled synthetic test rows | 1.0000* |
| Fused pipeline | 8 controlled rendered demos | 8/8 intended checks |

`*` The structural data is deliberately separable synthetic teaching data. This number is not evidence of performance on arbitrary real websites. Full per-class F1, confusion matrices, split counts, and feature importance are in `results/`.

## Tests

```powershell
pytest -q
```

The browser integration test is optional because it needs a running app:

```powershell
$env:DARKGUARD_BROWSER_TESTS = "1"
pytest -q tests/test_browser_integration.py
```

## Data and model sources

- Yada, Feng, Matsumoto, Fukushima, Kido, and Yamana, [*Dark patterns in e-commerce: a dataset and its baseline evaluations*](https://arxiv.org/abs/2211.06543), with the public [EC dark-pattern dataset](https://github.com/yamanalab/ec-darkpattern). DarkGuard retains only the five supported categories and records removals in `results/data_split.json`.
- Hugging Face model card for [`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased), the pretrained checkpoint fine-tuned locally.
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for screenshot text boxes and confidence.
- [Playwright for Python](https://playwright.dev/python/) for the rendered browser and computed DOM state.

## Limitations and future work

- A screenshot or DOM snapshot cannot prove that a stock or urgency claim is false.
- Multi-page patterns such as basket sneaking and drip pricing require workflow comparison, which this single-page tool does not perform.
- Cross-origin iframes, canvas text, closed shadow DOM, anti-bot pages, and unusual rendering can reduce coverage.
- Screenshot mode cannot reliably recover checked state, semantic roles, or relative controls without DOM information.
- The synthetic structural results do not prove generalisation to every real website.
- Language and layouts outside the English training data may perform poorly.
- DarkGuard is an educational screening tool, not a legal-compliance decision-maker.

YOLO could be explored in future with a properly labelled screenshot dataset to improve screenshot-only structural detection. It is intentionally not included now: URL mode already has exact DOM geometry and the project has no suitable aligned vision dataset. ResNet, vision transformers, LLM APIs, and paid APIs are likewise not part of the core system.
