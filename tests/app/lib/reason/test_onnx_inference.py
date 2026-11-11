"""Tests for app.lib.reason.onnx_inference module."""

import numpy as np

from app.lib.reason.onnx_inference import softmax

_SOFTMAX_TOLERANCE = 1e-7
_THIRD = 1.0 / 3
_LARGE_LOGIT = 1000.0


class TestSoftmax:
    """Test the softmax function."""

    def test_basic_softmax(self) -> None:
        """The results add up to one and none of them are negative."""
        logits = np.array([2.0, 1.0, 0.0])
        probs = softmax(logits)
        assert abs(np.sum(probs) - 1.0) < _SOFTMAX_TOLERANCE
        assert np.all(probs >= 0)

    def test_equal_logits(self) -> None:
        """Equal scores share the total evenly."""
        logits = np.array([1.0, 1.0, 1.0])
        probs = softmax(logits)
        assert abs(probs[0] - _THIRD) < _SOFTMAX_TOLERANCE

    def test_large_values(self) -> None:
        """Very large scores still add up to one."""
        logits = np.array([_LARGE_LOGIT, _LARGE_LOGIT, _LARGE_LOGIT])
        probs = softmax(logits)
        assert abs(np.sum(probs) - 1.0) < _SOFTMAX_TOLERANCE

    def test_negative_values(self) -> None:
        """Negative scores still add up to one."""
        logits = np.array([-2.0, -1.0, 0.0])
        probs = softmax(logits)
        assert abs(np.sum(probs) - 1.0) < _SOFTMAX_TOLERANCE

    def test_single_value(self) -> None:
        """A lone score gets all of the total."""
        logits = np.array([5.0])
        probs = softmax(logits)
        assert abs(probs[0] - 1.0) < _SOFTMAX_TOLERANCE

    def test_preserves_order(self) -> None:
        """A higher score gets a bigger share."""
        logits = np.array([3.0, 1.0, 0.0])
        probs = softmax(logits)
        assert probs[0] > probs[1] > probs[2]