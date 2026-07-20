# Online ONNX 2pass 调用链说明

本文说明当前项目 ONLINE ONNX 的实现方式。它参考官方 C++ runtime 2pass 的模型组合与事件语义，但仍运行在当前 Python 进程内，不外接官方 C++ server。

## 1. 模型组合

配置入口是 `engines.models.online.onnx`：

```yaml
online:
  enabled: onnx
  onnx:
    streaming_asr: "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx"
    vad: "iic/speech_fsmn_vad_zh-cn-16k-common-onnx"
    final_asr: "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx"
    punc: "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx"
```

`EngineModelManager.get_online_onnx_model_bundle()` 会加载：

- `streaming_asr`：`funasr_onnx.ParaformerOnline`。
- `vad`：`funasr_onnx.Fsmn_vad_online` 或当前包提供的等价 Fsmn VAD。
- `final_asr`：`funasr_onnx.Paraformer`。
- `punc`：`funasr_onnx.CT_Transformer`。

## 2. 调用链

```text
WebSocket /online/stream
  -> src/api/routes/online.py
  -> OnlineAsrService.create_realtime_session(...)
  -> ONNXOnlineAsrService.create_realtime_session(...)
  -> EngineModelManager.get_online_onnx_recognizer()
  -> ONNXOnlineRecognizer.create_session(...)
  -> EngineModelManager.get_online_onnx_model_bundle()
  -> OnlineONNXModelLoader.load_models(...)
  -> OnlineOnnxRealtimeSession
```

音频处理主链：

```text
binary PCM
  -> OnlineOnnxRealtimeSession.add_audio(...)
  -> ONNXVADWrapper.feed(...)
  -> confirmed VAD segment
  -> ONNXFinalASRWrapper.generate(segment_audio)
  -> ONNXPuncWrapper.generate(final_text)
  -> locked sentence
  -> build_offline_event()
  -> mode = "2pass-offline"
```

实时 partial 链：

```text
OnlineOnnxRealtimeSession.decode_partial(...)
  -> ONNXStreamingASRWrapper.generate(..., cache, is_final=False)
  -> build_online_event()
  -> mode = "2pass-online"
```

## 3. 和官方 C++ runtime 2pass 的对应关系

当前 Python ONNX 2pass 路径加载：

```text
online ASR
VAD
offline ASR
PUNC
```

## 4. 事件语义

项目继续保持现有 WebSocket 协议：

- `ready`
- `started`
- `2pass-online`
- `2pass-offline`
- `stopped`
- `error`

其中 `2pass-online` 是低延迟 partial，`2pass-offline` 是 VAD 结束后 final ASR + PUNC 的稳定句子。

## 5. 关键文件

- `src/engine_runtime/manager.py`
- `src/engine_runtime/engines/online/base.py`
- `src/engine_runtime/engines/online/onnx/loader.py`
- `src/engine_runtime/engines/online/onnx/adapters.py`
- `src/engine_runtime/engines/online/onnx/recognizer.py`
- `src/engine_runtime/engines/online/session_manager.py`
- `src/api/routes/online.py`
