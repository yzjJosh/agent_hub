"""Run a model on an image and turn its scores into probabilities."""

__all__ = ["run_inference", "softmax"]

import numpy as np
from onnxruntime import InferenceSession

def run_inference(session: InferenceSession, img_array: np.ndarray) -> np.ndarray:
    """Run the model on one image and return its raw scores.

    Args:
        session: The loaded model.
        img_array: Image data holding a single image.

    Returns:
        One raw score per class.
    """
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_array})
    return outputs[0][0]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Turn raw scores into probabilities that add up to 1.

    Args:
        logits: One raw score per class.

    Returns:
        One probability per class.
    """
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)