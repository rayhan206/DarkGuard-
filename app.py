from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from darkguard.pipeline import analyse_screenshot, analyse_url


BASE_DIR = Path(__file__).resolve().parent
DEMO_NAMES = {
    "safe",
    "urgency",
    "scarcity",
    "social_proof",
    "confirmshaming",
    "preselection",
    "visual_interference",
    "dynamic_timer",
}

app = FastAPI(title="DarkGuard", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/demo/{name}")
def demo_page(name: str) -> FileResponse:
    if name not in DEMO_NAMES:
        raise HTTPException(status_code=404, detail="Unknown demonstration page.")
    return FileResponse(BASE_DIR / "demo_pages" / f"{name}.html")


@app.post("/api/analyse/url")
async def analyse_url_endpoint(payload: dict) -> dict:
    url = str(payload.get("url", "")).strip()
    if not url:
        raise HTTPException(status_code=400, detail="Enter a webpage URL to analyse.")
    try:
        observe_seconds = float(payload.get("observe_seconds", 2))
        return await analyse_url(url, max(0.2, min(observe_seconds, 10)))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/analyse/screenshot")
async def analyse_screenshot_endpoint(
    image: UploadFile = File(...),
) -> dict:
    allowed_types = {"image/png", "image/jpeg", "image/webp"}
    if image.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Upload a PNG, JPEG, or WebP image.")
    contents = await image.read(8 * 1024 * 1024 + 1)
    if len(contents) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="The image must be 8 MB or smaller.")
    try:
        return analyse_screenshot(contents)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
