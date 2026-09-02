# DarkGuard teacher guide

## 1. Two-minute explanation

DarkGuard screens a shopping interface for potential dark-pattern cues and explains why each region was highlighted. In URL mode, Playwright opens the real rendered page at 1365×768, reads visible element text, coordinates, computed styles, checkbox state, animations, and changes observed after load, and captures the matching viewport screenshot. DistilBERT classifies language; a small Random Forest classifies numeric DOM/layout features; direct rules preserve facts such as “this checkbox is checked” or “this text changed twice.” A simple overlap function combines matching evidence. Pillow draws the result.

In screenshot mode, there is no DOM. EasyOCR supplies visible text, confidence, and coordinates; the language model and text rules then use those blocks. The UI explicitly says that checked states and structural patterns are less reliable in this mode. All output says “potential cue”; it never claims a company broke the law or that a timer is definitely fake.

## 2. Important files

| File | Purpose |
|---|---|
| `app.py` | FastAPI routes, upload validation, static files, demo routes, and `python app.py` startup. |
| `darkguard/browser.py` | URL safety, Playwright capture, DOM/style extraction, and mutation observation. |
| `darkguard/models.py` | Colour contrast, structural features, saved-model loading, inference, and confirmshaming post-processing. |
| `darkguard/pipeline.py` | Rules, model calls, fusion, cautious explanations, and the two analysis modes. |
| `darkguard/screenshot.py` | In-memory image validation, one cached EasyOCR reader, annotation, and data URLs. |
| `static/` | Plain responsive HTML, CSS, and JavaScript interface. |
| `demo_pages/` | Eight fictional controlled examples; no internet is needed for the demo. |
| `training/download_text_data.py` | Downloads, deduplicates, maps categories, and creates `page_id`-grouped splits. |
| `training/train_text_baseline.py` | Trains/saves TF-IDF + class-balanced Logistic Regression and honest metrics. |
| `training/train_distilbert.py` | Fine-tunes DistilBERT and saves the best validation checkpoint and label map. |
| `training/build_structural_data.py` | Builds explicitly synthetic numeric variations grouped by template. |
| `training/train_structural_model.py` | Trains the 100-tree, depth-6 Random Forest and saves feature importance. |
| `training/evaluate.py` | Recomputes the rules/model comparison and rendered-demo checks. |
| `tests/` | Focused feature, fusion, safety, API, image, and optional browser tests. |
| `results/` | Genuine JSON metrics and the feature-importance table. |

## 3. Why Playwright instead of raw source

Raw downloaded HTML can miss JavaScript-generated timers, delayed pop-ups, final checkbox state, computed CSS, and on-screen coordinates. Playwright observes the rendered page the user actually sees. The fixed viewport and device scale factor of one make DOM pixels match screenshot pixels. DarkGuard only observes; it does not click, submit, log in, or purchase.

## 4. What DistilBERT learns

The model learns language patterns for `benign`, `urgency`, `scarcity`, `social_proof`, and `misedirection` from the public EC dark-pattern text dataset. Exact duplicate text is removed. Pages, not individual rows, are assigned to train/validation/test partitions, so text from one page cannot leak across splits. Guilt-based misdirection wording is conservatively post-processed as `confirmshaming`.

The genuine grouped test macro-F1 is 0.9663. Per-class F1 is: benign 0.9721, urgency 1.0000, scarcity 0.9831, social proof 0.9800, and misdirection 0.8966. This measures the held-out dataset, not all web text.

## 5. What the Random Forest learns

The Random Forest receives 14 understandable numbers: element type/state, popup/optional/price/subscription indicators, area and font ratios, colour contrast, accept-versus-reject area ratio, opacity, text changes, and animations. It predicts `benign`, `preselected_addon`, `visual_interference`, `popup_pressure`, or `subscription_warning`.

Its 1.0 synthetic test macro-F1 is intentionally easy to interpret and must be described as performance on 50 controlled synthetic variations only. Template IDs are grouped, so five variations of one template never appear across splits. `results/structural_feature_importance.csv` shows exactly which features the trees used.

## 6. How fusion works

Each source produces the same small detection dictionary. Predictions below 0.65 are removed. When boxes substantially overlap, DarkGuard keeps the more specific/higher-confidence label, expands to their union, and preserves evidence and source names from both. No neural fusion network is used. Runtime facts can therefore support a language prediction without being hidden inside another model.

## 7. Why YOLO and ResNet were not necessary

For URLs, Playwright already returns exact element types, states, and coordinates; object detection would try to rediscover weaker versions of those facts. A ResNet crop also lacks the surrounding layout needed to compare accept and reject controls, and this project has no large aligned multimodal dataset. Keeping them out makes every input and decision explainable. YOLO is reasonable future work only if a sufficiently large labelled screenshot dataset is created for screenshot-only structural analysis.

## 8. Exact live demonstration

1. In PowerShell, activate the environment and run `python app.py`.
2. Open `http://127.0.0.1:8000`.
3. Leave the supplied urgency URL and select **Analyse page**. Point out the original and annotated screenshots, active engine, confidence, evidence, and ethical alternative.
4. Change the URL to `http://127.0.0.1:8000/demo/dynamic_timer`. Explain that the MutationObserver records text changes without clicking anything.
5. Analyse `http://127.0.0.1:8000/demo/preselection`. Point out that checked state comes directly from the DOM and that the structural model is trained only on controlled synthetic templates.
6. Analyse `http://127.0.0.1:8000/demo/safe`; it should show no prominent cues.
7. Open the urgency demo in a second tab, take a screenshot, switch DarkGuard to **Screenshot**, and upload it. Point out the EasyOCR boxes and the structural limitation note.
8. If asked for evidence, open `results/evaluation_summary.json` and `results/structural_feature_importance.csv`.

The first screenshot request may take longer while EasyOCR obtains its English weights.

## 9. Likely viva questions

**Why group by `page_id`?** Similar text from the same page can otherwise enter training and testing and inflate performance through leakage.

**Why report macro-F1?** It gives every class equal importance even though benign examples are more common. Accuracy alone could hide weak minority classes.

**Why train Logistic Regression first?** It is a strong, fast, inspectable classical baseline. DistilBERT must demonstrate improvement over it; here test macro-F1 rose from 0.9213 to 0.9663.

**How is class imbalance handled?** Logistic Regression uses balanced class weights. DistilBERT oversamples only training rows after grouped splitting. Validation and test remain untouched.

**Can the tool prove an urgency message is false?** No. It can see time-pressure wording and observe a changing timer, but proving falsity requires temporal or workflow evidence beyond one analysis.

**Why does URL mode reject private addresses?** It reduces server-side request forgery risk. Only the exact built-in localhost demo paths are allowed for teaching.

**Why combine rules with ML?** Direct facts such as a checked checkbox should not be guessed by a language model. Rules also create understandable evidence and a dependable fallback.

**What happens if DistilBERT is absent?** Startup honestly reports and uses the saved TF-IDF model; if that is absent too, text rules remain available.

**Does `data-darkguard-label` help predictions?** No. The analyser never reads it. It exists only to document labels for controlled data generation/evaluation.

## 10. Honest limitations

- A screenshot or DOM snapshot cannot prove that a stock claim is false.
- A timer update does not prove the deadline is fabricated; stronger longitudinal evidence would be needed before even using a phrase such as `likely_resetting_timer`.
- Multi-page patterns such as basket sneaking and drip pricing require workflow comparison.
- Cross-origin iframes, canvas text, closed shadow DOM, anti-bot pages, and dynamic navigation may reduce coverage.
- OCR can split a sentence, miss stylised text, or slightly offset a box.
- Screenshot mode cannot reliably identify checked state or semantic control roles.
- Synthetic structural results do not prove generalisation to every real website.
- The text data and OCR configuration are English-focused.
- DarkGuard is an educational screening tool, not a legal-compliance decision-maker.
