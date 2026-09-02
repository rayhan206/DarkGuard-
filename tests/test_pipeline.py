from darkguard.pipeline import merge_detections, merge_nearby_text, run_dom_pipeline, text_rule_detection


def detection(label, confidence, box, evidence):
    return {
        "label": label, "confidence": confidence, "bbox": box, "text": "example",
        "evidence": [evidence], "suggestion": "Use neutral wording.", "sources": [evidence], "source": evidence,
    }


def test_text_rules_use_cautious_urgency_label():
    result = text_rule_detection({"text": "Hurry, offer ends in 02:10", "bbox": [0, 0, 100, 20]})
    assert result["label"] == "potential_urgency"
    assert "fake" not in result["label"]


def test_fusion_merges_overlapping_evidence():
    text = detection("potential_urgency", 0.86, [10, 10, 200, 50], "text_model")
    runtime = detection("potential_urgency", 0.91, [12, 11, 198, 49], "runtime")
    merged = merge_detections([text, runtime])
    assert len(merged) == 1
    assert merged[0]["confidence"] == 0.91
    assert set(merged[0]["sources"]) == {"text_model", "runtime"}


def test_low_confidence_prediction_is_removed():
    assert merge_detections([detection("potential_scarcity", 0.4, [0, 0, 20, 20], "text_model")]) == []


def test_adjacent_ocr_fragments_on_one_line_are_merged():
    first = detection("potential_urgency", 0.9, [10, 10, 90, 30], "text_model")
    first["text"] = "Limited-time offer"
    second = detection("potential_urgency", 0.86, [105, 11, 240, 31], "rule")
    second["text"] = "order today"
    merged = merge_nearby_text([first, second])
    assert len(merged) == 1
    assert merged[0]["text"] == "Limited-time offer order today"


def test_safe_fixture_has_no_obvious_false_warning():
    elements = [
        {"text": "Durable canvas backpack", "tag": "h1", "bbox": [50, 50, 350, 100], "font_size": 42, "opacity": 1},
        {"text": "Add to basket", "tag": "button", "bbox": [50, 150, 180, 195], "font_size": 16, "opacity": 1},
        {"text": "View dimensions", "tag": "button", "bbox": [190, 150, 330, 195], "font_size": 16, "opacity": 1},
    ]
    assert run_dom_pipeline(elements, {"width": 1365, "height": 768}) == []
