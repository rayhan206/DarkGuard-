import os
import asyncio

import pytest

from darkguard.browser import capture_page


@pytest.mark.skipif(os.getenv("DARKGUARD_BROWSER_TESTS") != "1", reason="set DARKGUARD_BROWSER_TESTS=1 while python app.py is running")
def test_dynamic_demo_records_text_changes():
    page = asyncio.run(capture_page("http://127.0.0.1:8000/demo/dynamic_timer", observe_seconds=1.2))
    timer = next(item for item in page["elements"] if "Offer ends" in item["text"])
    assert timer["text_change_count"] >= 1
    assert page["screenshot"].startswith(b"\x89PNG")
