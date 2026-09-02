# DarkGuard implementation plan

DarkGuard will be built and verified in the required order. Only one phase is marked in progress at a time.

1. [complete] Minimal FastAPI application foundation
2. [complete] Controlled local demo pages
3. [complete] Playwright rendered-DOM extraction
4. [complete] Explainable rule baseline and annotation
5. [complete] Text dataset, TF-IDF baseline, and optional DistilBERT training
6. [complete] Synthetic structural data and Random Forest training
7. [complete] Transparent fusion
8. [complete] EasyOCR screenshot upload mode
9. [complete] Plain HTML/CSS/JavaScript frontend
10. [complete] Tests and genuine evaluation
11. [complete] Documentation, cleanup, and final verification

Model or dependency failures will be recorded rather than replaced with fabricated results. The working application will retain the documented fallback chain: DistilBERT, then TF-IDF, then rules.
