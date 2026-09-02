import re
from functools import lru_cache

from darkguard.browser import capture_page
from darkguard.models import (
    TextClassifier,
    load_structural_model,
    postprocess_text_label,
    predict_structural,
    structural_features,
)
from darkguard.screenshot import annotate_image, extract_ocr_blocks, image_data_url, read_image


DEFAULT_THRESHOLD = 0.65
SUGGESTIONS = {
    "potential_urgency": "State the real end date or let the customer decide without a countdown.",
    "potential_scarcity": "Show verifiable availability in neutral language.",
    "social_proof": "Present popularity information with its source and time period.",
    "confirmshaming": "Use a neutral rejection such as: Continue without the discount.",
    "potential_misdirection": "Give each choice direct, neutral wording.",
    "preselected_addon": "Leave optional paid additions unchecked until the customer chooses them.",
    "visual_interference": "Give accept and reject choices comparable size, contrast, and readability.",
    "popup_pressure": "Let visitors browse before showing a dismissible, neutral prompt.",
    "subscription_warning": "State the renewal price and frequency beside the choice in plain language.",
}


@lru_cache(maxsize=1)
def get_text_classifier() -> TextClassifier:
    return TextClassifier()


@lru_cache(maxsize=1)
def get_structural_model():
    return load_structural_model()


def make_detection(
    label: str,
    confidence: float,
    element: dict,
    evidence: list[str],
    sources: list[str],
) -> dict:
    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "bbox": [round(float(value), 2) for value in element.get("bbox", [0, 0, 0, 0])],
        "text": element.get("text", "")[:300],
        "evidence": evidence,
        "suggestion": SUGGESTIONS[label],
        "sources": sources,
        "source": sources[0],
    }


def text_rule_detection(element: dict) -> dict | None:
    text = element.get("text", "").strip()
    lower = text.lower()
    if len(text) < 3:
        return None
    guilt = postprocess_text_label("misedirection", text) == "confirmshaming"
    if guilt:
        return make_detection("confirmshaming", 0.94, element, ["rejection wording uses guilt or self-criticism"], ["rule", "dom"])
    if re.search(r"\b(only \d+ left|few left|selling fast|almost gone|low stock)\b", lower):
        return make_detection("potential_scarcity", 0.88, element, ["scarcity wording was found"], ["rule"])
    social_terms = r"\b(\d+ (?:people|shoppers|customers).*(?:bought|viewing|watching)|popular right now|recently purchased)\b"
    if re.search(social_terms, lower):
        return make_detection("social_proof", 0.87, element, ["popularity or recent-activity wording was found"], ["rule"])
    urgency = re.search(r"\b(limited[- ]time|ends? (?:today|soon|in)|hurry|act now|order today|last chance)\b", lower)
    countdown = re.search(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", lower)
    if urgency or countdown:
        evidence = ["time-pressure wording or a countdown was found"]
        if element.get("text_change_count", 0):
            evidence.append(f"text changed {element['text_change_count']} time(s) during observation")
        return make_detection("potential_urgency", 0.9 if countdown else 0.86, element, evidence, ["rule", "runtime"])
    return None


def structural_rule_detections(elements: list[dict]) -> list[dict]:
    detections = []
    for element in elements:
        text = element.get("text", "").lower()
        if element.get("input_type") == "checkbox" and element.get("checked") and re.search(r"add|optional|protect|insurance|club|₹|\$|£|€", text):
            detections.append(make_detection("preselected_addon", 0.97, element, ["an optional-looking checkbox was already checked"], ["dom", "rule"]))
        if re.search(r"subscription|monthly|renews?|recurring", text) and re.search(r"₹|\$|£|€|price|month|year", text):
            detections.append(make_detection("subscription_warning", 0.78, element, ["renewal or recurring-payment wording was found"], ["dom", "rule"]))
        if element.get("appeared_after_load") and (element.get("role") == "dialog" or element.get("tag") == "dialog"):
            detections.append(make_detection("popup_pressure", 0.84, element, ["dialog appeared after the initial page load"], ["runtime", "dom"]))
    prominence = visual_interference_detection(elements)
    if prominence:
        detections.append(prominence)
    return detections


def is_control(element: dict, terms: str) -> bool:
    is_button = element.get("tag") == "button" or element.get("role") == "button"
    return is_button and bool(re.search(terms, element.get("text", "").lower()))


def box_area(box: list[float]) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def box_union(first: list[float], second: list[float]) -> list[float]:
    return [min(first[0], second[0]), min(first[1], second[1]), max(first[2], second[2]), max(first[3], second[3])]


def visual_interference_detection(elements: list[dict]) -> dict | None:
    accepts = [item for item in elements if is_control(item, r"accept|agree|allow|yes|get my")]
    rejects = [item for item in elements if is_control(item, r"reject|decline|no thanks|not now|continue without")]
    for accept in accepts:
        for reject in rejects:
            if postprocess_text_label("misedirection", reject.get("text", "")) == "confirmshaming":
                continue
            combined = box_union(accept["bbox"], reject["bbox"])
            if box_area(combined) > 0.2 * 1365 * 768:
                continue
            ratio = box_area(accept["bbox"]) / max(1, box_area(reject["bbox"]))
            font_ratio = accept.get("font_size", 16) / max(1, reject.get("font_size", 16))
            if ratio >= 2 or font_ratio >= 1.4 or reject.get("opacity", 1) < 0.65:
                item = {"bbox": combined, "text": f"{accept.get('text', '')} / {reject.get('text', '')}"}
                evidence = [f"accept control is {ratio:.1f}× the reject control area", "choices have unequal visual prominence"]
                return make_detection("visual_interference", min(0.95, 0.68 + ratio / 20), item, evidence, ["dom", "rule"])
    return None


def add_prominence_features(elements: list[dict]) -> None:
    result = visual_interference_detection(elements)
    if not result:
        return
    combined = result["bbox"]
    ratio_text = result["evidence"][0]
    ratio_match = re.search(r"([\d.]+)×", ratio_text)
    ratio = float(ratio_match.group(1)) if ratio_match else 1
    for element in elements:
        if overlap_fraction(element["bbox"], combined) > 0:
            element["accept_reject_area_ratio"] = ratio


def model_text_detections(elements: list[dict], threshold: float) -> list[dict]:
    classifier = get_text_classifier()
    candidates = [item for item in elements if len(item.get("text", "").strip()) >= 3]
    predictions = classifier.classify([item["text"] for item in candidates])
    detections = []
    label_map = {"urgency": "potential_urgency", "scarcity": "potential_scarcity", "misedirection": "potential_misdirection"}
    for element, (label, confidence) in zip(candidates, predictions):
        label = postprocess_text_label(label, element["text"])
        if label == "benign" or confidence < threshold:
            continue
        is_choice = element.get("tag") in {"button", "a", "label", "ocr_text"} or element.get("role") == "button"
        if label == "misedirection" and not is_choice:
            continue
        final_label = label_map.get(label, label)
        evidence = [f"{classifier.engine} classified the wording as {label}"]
        if label == "confirmshaming":
            evidence.append("the wording appears to be a guilt-based rejection")
        detections.append(make_detection(final_label, confidence, element, evidence, ["text_model", "dom"]))
    return detections


def model_structural_detections(elements: list[dict], viewport: dict, threshold: float) -> list[dict]:
    saved_model = get_structural_model()
    if not saved_model:
        return []
    detections = []
    for element in elements:
        features = structural_features(element, viewport)
        label, confidence = predict_structural(saved_model, features)
        if label == "benign" or confidence < threshold:
            continue
        evidence = [f"Random Forest detected {label.replace('_', ' ')} from DOM and layout features"]
        detections.append(make_detection(label, confidence, element, evidence, ["structural_model", "dom"]))
    return detections


def overlap_fraction(first: list[float], second: list[float]) -> float:
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    return intersection / max(1, min(box_area(first), box_area(second)))


def merge_detections(detections: list[dict], threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    specificity = {"confirmshaming": 4, "social_proof": 4, "preselected_addon": 4, "potential_urgency": 3}
    eligible = [item for item in detections if item["confidence"] >= threshold]
    eligible.sort(key=lambda item: (specificity.get(item["label"], 2), item["confidence"]), reverse=True)
    merged = []
    for candidate in eligible:
        existing = next((item for item in merged if overlap_fraction(item["bbox"], candidate["bbox"]) >= 0.45), None)
        if not existing:
            merged.append(candidate.copy())
            continue
        existing["bbox"] = box_union(existing["bbox"], candidate["bbox"])
        existing["confidence"] = max(existing["confidence"], candidate["confidence"])
        existing["evidence"] = list(dict.fromkeys(existing["evidence"] + candidate["evidence"]))
        existing["sources"] = list(dict.fromkeys(existing["sources"] + candidate["sources"]))
        if not existing["text"]:
            existing["text"] = candidate["text"]
    return sorted(merged, key=lambda item: item["confidence"], reverse=True)


def merge_nearby_text(detections: list[dict], maximum_gap: float = 30) -> list[dict]:
    merged = []
    for candidate in sorted(detections, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        existing = next((item for item in merged if same_text_line(item, candidate, maximum_gap)), None)
        if not existing:
            merged.append(candidate.copy())
            continue
        left_text, right_text = (existing["text"], candidate["text"]) if existing["bbox"][0] <= candidate["bbox"][0] else (candidate["text"], existing["text"])
        existing["text"] = f"{left_text} {right_text}".strip()
        existing["bbox"] = box_union(existing["bbox"], candidate["bbox"])
        existing["confidence"] = max(existing["confidence"], candidate["confidence"])
        existing["evidence"] = list(dict.fromkeys(existing["evidence"] + candidate["evidence"]))
        existing["sources"] = list(dict.fromkeys(existing["sources"] + candidate["sources"]))
    return sorted(merged, key=lambda item: item["confidence"], reverse=True)


def same_text_line(first: dict, second: dict, maximum_gap: float) -> bool:
    if first["label"] != second["label"]:
        return False
    first_box, second_box = first["bbox"], second["bbox"]
    vertical_overlap = max(0, min(first_box[3], second_box[3]) - max(first_box[1], second_box[1]))
    smaller_height = max(1, min(first_box[3] - first_box[1], second_box[3] - second_box[1]))
    horizontal_gap = max(0, max(first_box[0], second_box[0]) - min(first_box[2], second_box[2]))
    return vertical_overlap / smaller_height >= 0.5 and horizontal_gap <= maximum_gap


def run_dom_pipeline(elements: list[dict], viewport: dict, threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    add_prominence_features(elements)
    rule_detections = [result for item in elements if (result := text_rule_detection(item))]
    all_detections = rule_detections + structural_rule_detections(elements)
    all_detections += model_text_detections(elements, threshold)
    all_detections += model_structural_detections(elements, viewport, threshold)
    return merge_detections(all_detections, threshold)


async def analyse_url(url: str, observe_seconds: float = 2) -> dict:
    page = await capture_page(url, observe_seconds)
    detections = run_dom_pipeline(page["elements"], page["viewport"])
    annotated = annotate_image(page["screenshot"], detections)
    text_engine = get_text_classifier().engine
    structural_engine = "Random Forest" if get_structural_model() else "structural rules"
    return {
        "mode": "url",
        "active_engine": f"{text_engine} + {structural_engine}",
        "summary": f"Found {len(detections)} potential dark-pattern cue(s)." if detections else "No prominent potential cues were found.",
        "detections": detections,
        "original_image": image_data_url(page["screenshot"]),
        "annotated_image": image_data_url(annotated),
        "page": {"title": page["title"], "url": page["final_url"], "elements_examined": len(page["elements"])},
        "limitations": ["A rendered snapshot cannot prove that urgency or stock claims are false."],
    }


def analyse_screenshot(image_bytes: bytes, threshold: float = DEFAULT_THRESHOLD) -> dict:
    image = read_image(image_bytes)
    blocks = extract_ocr_blocks(image_bytes)
    elements = [{**block, "tag": "ocr_text"} for block in blocks]
    rules = [result for item in elements if (result := text_rule_detection(item))]
    detections = merge_detections(rules + model_text_detections(elements, threshold), threshold)
    detections = merge_nearby_text(detections)
    annotated = annotate_image(image_bytes, detections)
    output = image_data_url(image_bytes, guess_mime_type(image_bytes))
    return {
        "mode": "screenshot",
        "active_engine": f"EasyOCR + {get_text_classifier().engine}",
        "summary": f"Found {len(detections)} potential text cue(s)." if detections else "No prominent potential text cues were found.",
        "detections": detections,
        "original_image": output,
        "annotated_image": image_data_url(annotated),
        "page": {"width": image.width, "height": image.height, "ocr_blocks": len(blocks)},
        "limitations": ["Screenshot mode analyses visible wording only.", "Structural states such as preselected checkboxes cannot always be detected reliably without DOM information."],
    }


def guess_mime_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if image_bytes.startswith(b"RIFF"):
        return "image/webp"
    return "image/png"
