import io

import pytest

from app.agents.browser_use.similarity import screen_similarity
from tests.app.agents.browser_use.conftest import paint_canvas

_BLACK = 0
_WHITE = 255
_ALIKE = 1.0
_NOTHING_IN_COMMON = 0.0
_NOTHING_TOLERANCE = 0.01
_ANCHOR_TOLERANCE = 0.02
_EIGHTH_RATIO = 0.125
_QUARTER_RATIO = 0.25
_HALF_RATIO = 0.5
_EIGHTH_SCORE = 0.875
_QUARTER_SCORE = 0.75
_HALF_SCORE = 0.5
_UNDECODABLE = b"not an image at all"


def _screen(color: int = _BLACK, changed_fraction: float = 0.0, patch_color: int = _WHITE) -> bytes:
    """Make the bytes of a flat screenshot with a band across the top.

    Args:
        color: The gray level of the whole canvas.
        changed_fraction: How much of the canvas height the band covers.
        patch_color: The gray level of the band.

    Returns:
        The PNG bytes of the canvas.
    """
    canvas = paint_canvas(color, changed_fraction, patch_color)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


class TestWholeScreens:
    async def test_identical_screens_score_the_top(self) -> None:
        shot = _screen(color=_BLACK, changed_fraction=_QUARTER_RATIO)
        assert await screen_similarity(shot, shot) == pytest.approx(_ALIKE)

    async def test_black_against_white_scores_nothing_in_common(self) -> None:
        score = await screen_similarity(_screen(color=_BLACK), _screen(color=_WHITE))
        assert score == pytest.approx(_NOTHING_IN_COMMON, abs=_NOTHING_TOLERANCE)


class TestUnreadableScreens:
    @pytest.mark.parametrize(
        "old_bytes, new_bytes",
        [
            (b"", _screen(color=_WHITE)),
            (_screen(color=_WHITE), b""),
            (b"", b""),
        ],
    )
    async def test_empty_bytes_score_the_top(self, old_bytes: bytes, new_bytes: bytes) -> None:
        assert await screen_similarity(old_bytes, new_bytes) == pytest.approx(_ALIKE)

    @pytest.mark.parametrize(
        "old_bytes, new_bytes",
        [
            (_UNDECODABLE, _screen(color=_WHITE)),
            (_screen(color=_WHITE), _UNDECODABLE),
            (_UNDECODABLE, _UNDECODABLE),
        ],
    )
    async def test_undecodable_bytes_score_the_top(self, old_bytes: bytes, new_bytes: bytes) -> None:
        assert await screen_similarity(old_bytes, new_bytes) == pytest.approx(_ALIKE)


class TestPartialChange:
    @pytest.mark.parametrize(
        "changed_fraction, expected",
        [
            (_EIGHTH_RATIO, _EIGHTH_SCORE),
            (_QUARTER_RATIO, _QUARTER_SCORE),
            (_HALF_RATIO, _HALF_SCORE),
        ],
    )
    async def test_flat_block_change_scores_the_unchanged_part(
        self,
        changed_fraction: float,
        expected: float,
    ) -> None:
        score = await screen_similarity(_screen(), _screen(changed_fraction=changed_fraction))
        assert score == pytest.approx(expected, abs=_ANCHOR_TOLERANCE)

    async def test_score_falls_as_more_of_the_screen_changes(self) -> None:
        eighth = await screen_similarity(_screen(), _screen(changed_fraction=_EIGHTH_RATIO))
        quarter = await screen_similarity(_screen(), _screen(changed_fraction=_QUARTER_RATIO))
        half = await screen_similarity(_screen(), _screen(changed_fraction=_HALF_RATIO))
        assert eighth > quarter
        assert quarter > half
