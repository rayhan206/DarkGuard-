import base64
import io
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


COLORS = ["#d62f49", "#8b3fd1", "#006d77", "#b45f06", "#1758a6"]


def image_data_url(image_bytes: bytes, mime_type: str = "image/png") -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def read_image(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded file is not a readable image.") from error


def annotate_image(image_bytes: bytes, detections: list[dict]) -> bytes:
    image = read_image(image_bytes)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=14)
    line_width = max(3, round(min(image.size) / 220))
    for index, detection in enumerate(detections):
        box = [round(value) for value in detection["bbox"]]
        color = COLORS[index % len(COLORS)]
        draw.rectangle(box, outline=color, width=line_width)
        label = f"{index + 1}. {detection['label'].replace('_', ' ')} {detection['confidence']:.0%}"
        text_box = draw.textbbox((box[0], box[1]), label, font=font)
        label_height = text_box[3] - text_box[1] + 8
        label_y = max(0, box[1] - label_height)
        label_width = text_box[2] - text_box[0] + 10
        draw.rectangle([box[0], label_y, box[0] + label_width, label_y + label_height], fill=color)
        draw.text((box[0] + 5, label_y + 3), label, fill="white", font=font)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@lru_cache(maxsize=1)
def get_ocr_reader():
    try:
        import easyocr
    except ImportError as error:
        raise RuntimeError("EasyOCR is not installed. Run: pip install -r requirements.txt") from error
    try:
        return easyocr.Reader(["en"], gpu=False)
    except Exception as error:
        raise RuntimeError(f"EasyOCR could not start: {str(error).splitlines()[0]}") from error


def extract_ocr_blocks(image_bytes: bytes, minimum_confidence: float = 0.35) -> list[dict]:
    import numpy as np

    image = read_image(image_bytes)
    results = get_ocr_reader().readtext(np.array(image), detail=1, paragraph=False)
    blocks = []
    for polygon, text, confidence in results:
        clean_text = " ".join(str(text).split())
        if confidence < minimum_confidence or len(clean_text) < 3:
            continue
        x_values = [point[0] for point in polygon]
        y_values = [point[1] for point in polygon]
        blocks.append({
            "text": clean_text,
            "ocr_confidence": float(confidence),
            "bbox": [min(x_values), min(y_values), max(x_values), max(y_values)],
        })
    return blocks
