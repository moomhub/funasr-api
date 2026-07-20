"""ONLINE ONNX model file requirements."""

REQUIRED_FILES = {
    "streaming_asr": ["model_quant.onnx", "decoder_quant.onnx", "config.yaml", "am.mvn", "tokens.json"],
    "vad": ["model_quant.onnx", "config.yaml", "am.mvn"],
    "final_asr": ["config.yaml", "am.mvn", "tokens.json"],
    "punc": ["model_quant.onnx", "config.yaml", "tokens.json"],
}
