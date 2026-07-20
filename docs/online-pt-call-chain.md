# Online PT 2pass 调用链说明

本文说明当前项目 ONLINE PT 的实现方式。它对齐官方 Python WebSocket 2pass 思路：`online ASR + VAD + offline ASR + PUNC`。

## 1. 模型组合

配置入口是 `engines.models.online.pt`：

```yaml
online:
  enabled: pt
  pt:
    streaming_asr: "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
    vad: "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    final_asr: "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    punc: "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
```

`EngineModelManager.get_online_pt_model_bundle()` 会加载四个 PT `AutoModel`：

- `streaming_asr`：实时 Paraformer，产出 `2pass-online` partial。
- `vad`：FSMN VAD，判定语音端点。
- `final_asr`：离线 Paraformer，对 VAD 完成段重识别。
- `punc`：CT-Transformer，对 final 文本补标点。

## 2. 调用链

```text
WebSocket /online/stream
  -> src/api/routes/online.py
  -> OnlineAsrService.create_realtime_session(...)
  -> PTOnlineAsrService.create_realtime_session(...)
  -> EngineModelManager.get_online_pt_recognizer()
  -> PTOnlineRecognizer.create_session(...)
  -> EngineModelManager.get_online_pt_model_bundle()
  -> OnlineRealtimeSession
```

流式音频进入 session 后：

```text
binary PCM
  -> OnlineRealtimeSession.add_audio(...)
  -> DynamicStreamingVAD.feed(...)
  -> confirmed VAD segment
  -> final_asr.generate(segment_audio)
  -> punc.generate(final_text)
  -> locked sentence
```

并行地，worker 会按 `decode_interval/chunk_ms` 调用：

```text
OnlineRealtimeSession.decode_partial(...)
  -> streaming_asr.generate(..., cache, is_final=False)
  -> build_online_event()
  -> mode = "2pass-online"
```

当 VAD 段完成后：

```text
OnlineRealtimeSession.build_offline_event(...)
  -> consume locked sentences
  -> mode = "2pass-offline"
```

STOP 时：

```text
STOP
  -> OnlineRealtimeSession.finish()
  -> VAD final flush
  -> final ASR + PUNC
  -> final 2pass-offline event
```

## 3. 和官方 Python 2pass 的对应关系

官方 Python WebSocket 2pass 的核心也是四段模型：

```text
online ASR partial
VAD endpoint
offline ASR re-decode
PUNC final text
```

## 4. 关键文件

- `src/engine_runtime/manager.py`
- `src/engine_runtime/engines/online/base.py`
- `src/engine_runtime/engines/online/pt/recognizer.py`
- `src/engine_runtime/engines/online/session_manager.py`
- `src/api/routes/online.py`
