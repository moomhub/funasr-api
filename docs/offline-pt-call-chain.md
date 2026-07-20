# Offline PT 调用链说明

本文说明当前项目中 OFFLINE PT 识别从 API 到 `funasr.AutoModel.generate(...)`、再到 standalone `spk` 二次识别与结果回写的完整路径，并解释为什么 `recognizer` 层当前使用了 `asyncio.to_thread(...)`。

## 1. 总体调用链

当前 OFFLINE PT 主路径不是“HTTP 请求内同步等待识别完成”，而是：

`API -> Scheduler -> Application Service -> Recognizer -> FunASR -> standalone SPK -> merge_result -> ResultHandler`

对应代码入口如下：

1. 上传入口：`src/api/routes/offline.py`
2. 即时调度：`_schedule_immediate_processing(...)`
3. 调度器：`src/scheduler/offline_scheduler.py`
4. 应用服务：`src/application/offline.py`
5. 识别器基类：`src/engine_runtime/engines/offline/base.py`
6. PT 识别器：`src/engine_runtime/engines/offline/pt/recognizer.py`
7. PT 模型加载器：`src/engine_runtime/loaders/pt_loader.py`
8. 底层模型：`funasr.AutoModel`

## 2. 详细时序

### 2.1 API 层

文件：`src/api/routes/offline.py`

`upload_offline_task()` 做的事情是：

1. 保存上传文件
2. 写入任务记录
3. 调用 `_schedule_immediate_processing(task_id)`
4. 立刻返回 HTTP 响应

这里不会同步等待识别结束。

## 2.2 Scheduler 层

文件：`src/scheduler/offline_scheduler.py`

调用链：

- queue worker 取出任务
- `_process_single_task(task_id)`
- `await self.task_service.process_task(task_id)`

这里 scheduler 负责：

- 取任务
- 控制并发信号量
- 调用 application service

它不负责模型推理细节。

## 2.3 Application 层

文件：`src/application/offline.py`

调用链：

- `OfflineTaskService.process_task()`
- `await self.recognition_service.recognize(...)`
- `OfflineRecognitionService.recognize()`
- `await recognizer.recognize(OfflineRecognitionRequest(...))`

这一层负责：

- 组装业务输入
- 处理任务上下文
- 调用 recognizer
- 统一错误结果

它不直接持有 PT/ONNX 推理逻辑。

## 2.4 Recognizer 基类模板

文件：`src/engine_runtime/engines/offline/base.py`

`BaseOfflineRecognizer.recognize()` 是统一模板：

1. `model = self.load_model()`
2. `payload = await self.run_inference(model, request)`
3. `result = self.parse_result(payload)`
4. 返回统一 `RecognitionResult`

这表示：

- `load_model()` 负责拿模型
- `run_inference()` 负责执行推理
- `parse_result()` 负责把底层返回值转换为统一结果对象

对当前 PT 实现来说，`run_inference()` 现在已经包含两步：

1. 先调用 OFFLINE PT 主模型做 ASR
2. 再调用 standalone `spk` 对整段音频做一次独立说话人识别

## 2.5 PT Recognizer

文件：`src/engine_runtime/engines/offline/pt/recognizer.py`

关键代码：

```python
async def run_inference(self, model: Any, request: OfflineRecognitionRequest) -> Any:
    asr_result = await asyncio.to_thread(
        model.generate,
        request.audio_path,
        batch_size_s=300,
        return_spk_res=True,
        hotwords=request.hotwords,
        **request.generate_kwargs,
    )
    speaker_result = await self._recognize_speaker(request.audio_path, request.generate_kwargs)
    return {
        "asr_result": asr_result,
        "speaker_result": speaker_result,
    }
```

这里的下一跳不是项目里的另一个 service，而是直接进入 `funasr.AutoModel.generate(...)`。

当前 PT 路径的项目内调用顺序是：

1. `OfflineRecognitionService.recognize()`
2. `PTOfflineRecognizer.recognize()`
3. `PTOfflineRecognizer.load_model()`
4. `EngineModelManager.get_offline_model()`
5. `PTModelLoader.load_model()`
6. 返回 `funasr.AutoModel(...)`
7. `PTOfflineRecognizer.run_inference()`
8. `await asyncio.to_thread(model.generate, ...)`
9. `PTOfflineRecognizer._recognize_speaker(audio_path, ...)`
10. `SPKRecognizer.recognize(...)`
11. 回到 `BaseOfflineRecognizer.recognize()`
12. `PTOfflineRecognizer.parse_result(payload)`
13. 用 standalone SPK 结果重组 `sentence_info`
14. 返回 `RecognitionResult`

当前 PT 新链路的合并规则是：

- ASR 文本和时间戳仍然来自 PT offline
- standalone `spk` 负责给整段音频重新做 diarization
- 最终 `segments[].speaker` 以 standalone `spk` 为准
- 如果有逐字时间戳，则按时间戳重新切段并合并
- 如果没有逐字时间戳，则退回到句级别重标注

## 2.6 底层 FunASR

本地包位置：

`E:\Web_Projected\python-api\.venv\Lib\site-packages\funasr\auto\auto_model.py`

`AutoModel.generate(...)` 的职责是根据模型装配情况继续分流：

- 没有 `vad_model`：走 `inference(...)`
- 有 `vad_model`：走 `inference_with_vad(...)`

当前项目的 PT offline loader 会把 `asr + vad + punc + spk` 一起传给 `AutoModel(...)`，所以当前 offline PT 通常会走带 VAD 的长音频路径。

## 3. 为什么这里看起来是异步

需要区分两件事：

1. `async` 是接口形式
2. 串行/并行 是实际执行语义

`run_inference()` 虽然是 `async def`，但它不是在做“真正的异步推理框架调用”。  
当前实际执行的是：

- `model.generate(...)` 本身是同步阻塞函数
- standalone `spk` 识别本身也是同步阻塞函数
- `asyncio.to_thread(...)` 只是把这些同步阻塞函数放到线程池里执行
- 当前协程等待线程执行完成

所以它的语义更接近：

“同步推理，只是为了不阻塞事件循环，被包成了 awaitable”

## 4. 它是不是业务并发

不是。

对单个 offline PT 请求来说，当前路径仍然可以理解成串行：

1. 取模型
2. 跑一次 `model.generate(...)`
3. 再跑一次 standalone `spk`
4. 重新按 SPK 合并结果
5. 保存结果

`to_thread(...)` 不会把这条业务链自动变成并行流水线。  
它只是把阻塞 CPU/IO 调用从事件循环线程移开。

## 5. 你可以怎么理解当前架构

如果从“架构理解模型”而不是从 Python 语法看，当前 OFFLINE PT 可以按下面这套 mental model 来理解：

- API 层：异步接入，请求尽快返回
- Scheduler 层：异步调度任务
- Application 层：串行业务编排
- Engine 层：执行具体 backend 推理
- 底层模型层：同步阻塞计算

也就是说：

- “异步”主要用于系统接入层和任务调度层
- “推理本体”仍然是同步模型调用
- `recognizer` 里的 `await asyncio.to_thread(...)` 只是适配层，不是新的业务层

## 6. 当前实现的利弊

### 优点

- `OfflineRecognitionService` 统一通过 `await recognizer.recognize(...)` 调用，不需要分别处理同步/异步 backend
- 事件循环不会被 `model.generate(...)` 长时间阻塞
- scheduler / service / route 的接口形态一致

### 缺点

- `engine` 层混入了 `asyncio.to_thread(...)`，边界不够纯
- 从架构阅读上看，容易让人误以为“推理本身是异步的”
- “接口 async” 和 “执行串行” 混在一起，理解成本会上升

## 7. 当前项目下的一个实用结论

如果你现在是为了理解项目，而不是立即改代码，可以先把这一层认成：

“PT recognizer 对同步 FunASR 模型的一个异步外壳”

也可以简化成一句话：

`API/Scheduler` 是异步的，`FunASR generate` 和 standalone `spk` 都是同步的，`recognizer` 只是把两边接起来。

## 8. 一张简化时序图

```text
HTTP Upload
  -> route/offline.py
  -> TaskSubmissionService.submit_offline(task_id)
  -> OfflineTaskQueue.enqueue_offline(task_id)
  -> queue worker
  -> OfflineTaskService.process_task(task_id)
  -> OfflineRecognitionService.recognize(audio_path)
  -> PTOfflineRecognizer.recognize(request)
  -> load_model()
  -> EngineModelManager.get_offline_model()
  -> PTModelLoader.load_model()
  -> funasr.AutoModel(...)
  -> run_inference()
  -> asyncio.to_thread(model.generate, ...)
  -> funasr.AutoModel.generate(...)
  -> PTOfflineRecognizer._recognize_speaker(...)
  -> SPKRecognizer.recognize(...)
  -> BaseOfflineRecognizer.parse_result(...)
  -> merge sentence_info by standalone spk
  -> ResultHandler.handle_success(...)
  -> repository.save_result(...)
```

## 9. 架构阅读建议

看 offline PT 路径时，建议按下面顺序读：

1. `src/api/routes/offline.py`
2. `src/scheduler/offline_scheduler.py`
3. `src/application/offline.py`
4. `src/engine_runtime/engines/offline/base.py`
5. `src/engine_runtime/engines/offline/pt/recognizer.py`
6. `src/engine_runtime/manager.py`
7. `src/engine_runtime/loaders/pt_loader.py`
8. `funasr/auto/auto_model.py`

这样最容易把“业务编排”和“模型推理”拆开看。
