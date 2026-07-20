"""OFFLINE model defaults."""

DEFAULT_MODELS = {
    "pt": {
        "asr": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        "spk": "iic/speech_campplus_sv_zh-cn_16k-common",
    },
    "onnx": {
        "asr": "marxyz/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-onnx",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-onnx",
        "punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large-onnx",
        "spk": "iic/speech_campplus_speaker-diarization_common",
    },
}
