import io

from fastapi.testclient import TestClient
from PIL import Image

from app import app


client = TestClient(app)


def test_health_and_frontend():
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/")
    assert response.status_code == 200
    assert "DarkGuard" in response.text


def test_demo_routes():
    assert client.get("/demo/safe").status_code == 200
    assert client.get("/demo/does-not-exist").status_code == 404


def test_invalid_image_type_and_content():
    wrong_type = client.post("/api/analyse/screenshot", files={"image": ("note.txt", b"hello", "text/plain")})
    assert wrong_type.status_code == 400
    unreadable = client.post("/api/analyse/screenshot", files={"image": ("broken.png", b"not an image", "image/png")})
    assert unreadable.status_code == 400


def test_oversized_image_is_rejected_before_decoding():
    response = client.post("/api/analyse/screenshot", files={"image": ("large.png", b"x" * (8 * 1024 * 1024 + 1), "image/png")})
    assert response.status_code == 400


def test_small_valid_image_can_be_created_for_ocr_tests():
    image = Image.new("RGB", (80, 40), "white")
    contents = io.BytesIO()
    image.save(contents, "PNG")
    assert contents.getvalue().startswith(b"\x89PNG")
