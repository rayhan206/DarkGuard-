import pytest

from darkguard.browser import validate_url
from darkguard.models import contrast_ratio, postprocess_text_label, structural_features
from training.build_structural_data import build_data


def test_url_validation_rejects_unsafe_schemes_and_private_hosts():
    with pytest.raises(ValueError, match="Only http"):
        validate_url("file:///tmp/page.html")
    with pytest.raises(ValueError, match="Private-network"):
        validate_url("http://127.0.0.1/admin")
    with pytest.raises(ValueError, match="Private-network"):
        validate_url("http://127.0.0.1/demo/not-built-in", allow_local_demo=True)
    assert validate_url("http://127.0.0.1:8000/demo/safe", allow_local_demo=True).endswith("/demo/safe")


def test_colour_contrast():
    assert contrast_ratio("rgb(0, 0, 0)", "rgb(255, 255, 255)") == pytest.approx(21.0)
    assert contrast_ratio("rgb(120, 120, 120)", "rgb(120, 120, 120)") == pytest.approx(1.0)


def test_structural_feature_extraction():
    element = {
        "text": "Add protection for ₹99 per month - renews automatically",
        "tag": "input", "input_type": "checkbox", "checked": True,
        "bbox": [10, 10, 30, 30], "font_size": 16,
        "color": "rgb(0, 0, 0)", "background_color": "rgb(255, 255, 255)", "opacity": 1,
    }
    features = structural_features(element, {"width": 100, "height": 100})
    assert features["is_checkbox"] == 1
    assert features["is_checked"] == 1
    assert features["is_optional_addon"] == 1
    assert features["contains_subscription_word"] == 1
    assert features["area_ratio"] == pytest.approx(0.04)


def test_confirmshaming_postprocessing():
    assert postprocess_text_label("misedirection", "No thanks, I prefer paying full price") == "confirmshaming"
    assert postprocess_text_label("misedirection", "Continue without the offer") == "misedirection"


def test_structural_data_is_grouped_and_complete():
    data = build_data()
    assert set(data["label"]) == {"benign", "preselected_addon", "visual_interference", "popup_pressure", "subscription_warning"}
    assert data.groupby("template_id")["split"].nunique().max() == 1
    assert len(data) == 300
