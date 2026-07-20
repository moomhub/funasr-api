"""ONLINE model defaults."""

DEFAULT_MODELS = {
    "pt": {
        "streaming_asr": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "final_asr": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "punc": "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
    },
    "onnx": {
        "streaming_asr": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-onnx",
        "final_asr": "marxyz/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-onnx",
        "punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large-onnx",
    },
}
