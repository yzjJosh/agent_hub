from __future__ import annotations

import json

import pytest

from app.lib.browser.analyze_page import analyze_page
from tests.app.conftest import is_llm_available, skip_if_no_chrome

_TITLE_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}},
    "required": ["title"],
}


@pytest.mark.skipif(not is_llm_available(), reason="LLM server not available")
@skip_if_no_chrome
@pytest.mark.live
@pytest.mark.timeout(120)
class TestAnalyzePage:
    """Live tests that need a real browser and a real model."""

    async def test_analyze_page_returns_response(self, test_server_url: str) -> None:
        """Asking about a page gives back some text and the landed address."""
        response, final_url = await analyze_page(test_server_url, "Return the word OK")

        assert isinstance(response, str), f"unexpected return type: {type(response)}"
        assert response.strip(), "expected a non-empty LLM response"
        assert final_url.startswith(test_server_url), (
            f"expected the page to land on {test_server_url}, got {final_url}"
        )

    async def test_analyze_page_honors_output_json_schema(self, test_server_url: str) -> None:
        """Asking for a shape gives back JSON with that key in it."""
        response, _ = await analyze_page(
            test_server_url,
            "Return the page title as a JSON object with a 'title' key",
            output_json_schema=_TITLE_SCHEMA,
        )

        assert isinstance(response, str), f"unexpected return type: {type(response)}"
        parsed = json.loads(response)
        assert isinstance(parsed, dict)
        assert "title" in parsed

    async def test_analyze_page_applies_transform_snapshot(self, test_server_url: str) -> None:
        """Changing the page tree first still gives back text."""
        response, _ = await analyze_page(
            test_server_url,
            "Return the word OK",
            transform_snapshot=lambda _tree: {"role": "web"},
        )

        assert isinstance(response, str), f"unexpected return type: {type(response)}"
        assert response.strip(), "expected a non-empty LLM response"
