"""ONNX online recognizer."""

from .loader import OnlineONNXModelLoader
from .recognizer import ONNXOnlineRecognizer

__all__ = ["ONNXOnlineRecognizer", "OnlineONNXModelLoader"]
