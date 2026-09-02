# DarkGuard progress

## Phase 1 — complete

- Repository inspected with `Get-ChildItem`, `rg --files`, and an `AGENTS.md` search: repository was empty and contained no extra instructions.
- Implemented FastAPI, `/health`, static frontend serving, upload/URL endpoints, and `python app.py` startup.
- Verified with a running server: `/health` returned `{"status":"ok"}` and `/` returned HTTP 200.

## Phase 2 — complete

- Created eight fictional, dependency-free demo pages and `/demo/{name}` routes.
- Verified all demo endpoints and rendered all eight during the end-to-end analyser check.

## Phase 3 — complete

- Installed Playwright 1.54 and Chromium build 1181.
- Implemented a fresh 1365×768 context, visible-element/computed-style extraction, safe URL checks, screenshot capture, post-load mutation observation, and animation counts.
- Live timer check recorded two text changes during a 2.2-second observation.

## Phase 4 — complete

- Implemented cautious text/DOM rules, Pillow annotations, evidence, suggestions, and JSON output.
- End-to-end checks returned the intended cue on each demo; the safe demo returned no detections.

## Phase 5 — complete

- `python training/download_text_data.py`: 2,306 deduplicated supported rows, 1,208 pages, zero page-group leakage.
- `python training/train_text_baseline.py`: trained and saved the genuine TF-IDF + Logistic Regression model; test macro-F1 = 0.921286.
- First DistilBERT attempt failed because preinstalled optional `torchao 0.17.0` required a newer PyTorch API. Removed unused `torchao`; direct DistilBERT import then succeeded.
- Retried `python training/train_distilbert.py`; completed three CUDA epochs and saved the best checkpoint and label map. Best validation macro-F1 = 0.954896; grouped test macro-F1 = 0.966334.

## Phase 6 — complete

- `python training/build_structural_data.py`: wrote 300 explicitly synthetic rows across 60 template IDs; no template crosses a split.
- `python training/train_structural_model.py`: saved the requested 100-tree, depth-6 Random Forest, full report, confusion matrix, and feature-importance CSV.
- Controlled synthetic test macro-F1 = 1.0. The result is clearly labelled as synthetic and not real-world evidence.

## Phase 7 — complete

- Added 0.65 thresholding, overlap-by-smaller-box merging, specificity preference, evidence/source preservation, and readable suggestions.
- First full evaluation exposed safe-page DistilBERT false positives on neutral non-control text. Restricted DOM misdirection integration to choice controls; OCR remains eligible. Re-evaluation passed all 8/8 demo checks with no safe-page cue.

## Phase 8 — complete

- Implemented 8 MB/type/decode validation, in-memory processing, a once-cached English EasyOCR reader, confidence filtering, coordinates, text classification, and annotation.
- A real screenshot captured from the urgency demo returned HTTP 200, 10 OCR blocks, urgency detections, an annotated image, and the explicit missing-DOM limitation.

## Phase 9 — complete

- Built the responsive plain HTML/CSS/JavaScript UI with URL/image modes, loading/errors, active engine, paired images, detection cards, evidence, confidence, suggestions, and limitations.
- Frontend returned HTTP 200 during live-server checks.

## Phase 10 — complete

- Added URL, contrast, feature, post-processing, synthetic grouping, fusion, threshold, safe-page, API, upload, and optional browser tests.
- `pytest -q`: 14 passed, 1 optional integration test skipped.
- With a running app, `$env:DARKGUARD_BROWSER_TESTS='1'; pytest -q tests/test_browser_integration.py`: 1 passed.
- `python training/evaluate.py`: genuinely compared rules (0.629798 macro-F1), TF-IDF (0.921286), DistilBERT (0.966334), synthetic Random Forest (1.0 with warning), and fused rendered demos (8/8).

## Phase 11 — complete

- Added practical `README.md` and the requested ten-part `TEACHER_GUIDE.md` with accurate metrics, commands, citations, design reasons, and limitations.
- Searched for TODOs, placeholder core functions, prohibited API/model code, and swallowed errors; none were present (YOLO/ResNet appear only in the required future-work explanation).
- Final `python -m compileall -q app.py darkguard training tests; pytest -q`: 15 passed, 1 optional integration test skipped.
- Final live run: frontend HTTP 200, `/health` correct, URL mode used fine-tuned DistilBERT + Random Forest, screenshot mode used EasyOCR + fine-tuned DistilBERT, and both returned annotated PNG data. Adjacent OCR banner fragments merged into one readable detection.
- Removed only generated `__pycache__` and `.pytest_cache` directories after final verification; Python/pytest can recreate them.
- The available machine interpreter was Python 3.10.11 and had no `py` launcher; the project targets the documented Python 3.11 stack and was additionally verified compatible with the available 3.10 interpreter.
