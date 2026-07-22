# FunASR Multi-Mode API v1.0

基于 FastAPI 的 FunASR 语音识别服务，当前 v1.0 主要覆盖三类能力：

- **OFFLINE 文件转写**：上传音频后进入异步队列，完成 ASR、标点、可用时间戳下的说话人合并。
- **ONLINE 实时转写**：WebSocket 接收 16kHz mono int16 PCM 流，返回实时 partial 与二次 final 结果。
- **SPK 说话人识别**：独立说话人任务接口，同时作为 OFFLINE 的共享说话人运行时。

应用采用单一 Composition Root：`main.py -> src/bootstrap.py -> src/composition.py`。API、应用服务、队列、数据库、运行时模型在启动期统一装配，HTTP 上传任务和后台 worker 复用同一组 runtime service。

> 说明：当前业务 README 按项目 v1.0 编写；代码里的 FastAPI title / `config.yaml` 展示名仍保留 `FunASR v4.0` 字样，以实际运行返回为准。

## 能力概览

| 模式 | 入口 | 默认后端 | 说明 |
| --- | --- | --- | --- |
| OFFLINE | `POST /offline/recognize` | ONNX | 文件上传、队列异步处理、任务查询、结果落库/落盘 |
| ONLINE | `WS /online/stream` | PT（配置中默认） | 16kHz PCM 实时流，START/STOP 控制会话 |
| SPK | `POST /spk/recognize` | PT | standalone 说话人识别任务 |

默认 `config.yaml` 启用：

```yaml
engines:
  enabled:
    - offline
    # - online
    - spk
  models:
    offline:
      enabled: onnx
    online:
      enabled: pt
```

如需开启 ONLINE，将 `engines.enabled` 中的 `online` 取消注释。

## 目录结构

```text
main.py                       FastAPI 入口与 Uvicorn 启动
config.yaml                   主配置文件
src/api/                      HTTP / WebSocket 路由
src/application/              OFFLINE / ONLINE / SPK 业务编排
src/bootstrap.py              应用级依赖构建入口
src/composition.py            service graph 装配
src/core/                     配置、热词、结果类型、基础设施适配
src/database/                 SQLAlchemy 模型、DAO、repository
src/engine_runtime/           模型解析、加载、recognizer、runtime service
src/task_queue/               OFFLINE/SPK 队列、worker、结果 hook
tests/                        单元测试与运行时边界测试
```

关键 runtime 文件：

```text
src/engine_runtime/engines/offline/catalog.py       OFFLINE 默认模型目录
src/engine_runtime/engines/online/catalog.py        ONLINE 默认模型目录
src/engine_runtime/engines/offline/pt/recognizer.py OFFLINE PT 真实 ASR 调用
src/engine_runtime/engines/offline/onnx/recognizer.py OFFLINE ONNX VAD/ASR/PUNC/SPK 编排
src/engine_runtime/engines/online/onnx/final_asr.py ONLINE ONNX final ASR 适配
```

## 环境要求

- Python `>=3.10,<3.12`
- 推荐使用 `uv`
- CPU 可运行；GPU/设备选择由模型依赖与 FunASR/FunASR-ONNX 能力决定
- 首次启动如 `engines.auto_model_download: true`，会联网下载模型到本地缓存目录

安装依赖：

```bash
uv sync
```

安装开发/测试依赖：

```bash
uv sync --group dev
```

仅支持 pip 的部署环境可使用：

```bash
pip install -r requirements.txt
```

## 启动服务

```bash
uv run python main.py
```

服务读取 `config.yaml`，默认地址：

- API 文档：`http://127.0.0.1:8000/docs`
- OpenAPI：`http://127.0.0.1:8000/openapi.json`
- 健康检查：`GET http://127.0.0.1:8000/health`
- 根信息：`GET http://127.0.0.1:8000/`

如果希望直接用 Uvicorn：

```bash
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

服务使用单 worker：每个进程都会拥有自己的模型运行时和任务队列，不能通过增加
Uvicorn worker 数横向扩展。启动后会打印数据库、存储、通知、日志、引擎和队列的
配置/连接状态表；数据库或存储检查失败时服务不会继续启动。

## 配置说明

### 运行期目录

`runtime.root_dir` 是相对路径的根目录：

```yaml
runtime:
  root_dir: "./data"
  models_dir: "models"
  sqlite_path: "sqlite/funasr_tasks.db"
  temp_dir: "temp"
  offline_result_dir: "results/offline"
  local_files_dir: "files"
```

实际落盘结构：

```text
data/
  models/                  模型缓存
  sqlite/funasr_tasks.db   SQLite 数据库
  temp/                    上传临时文件
  results/offline/         OFFLINE 文本结果
  files/                   local 存储的归档目录
```

常用环境变量覆盖：

```powershell
$env:ENGINES_ENABLED="offline,online,spk"
$env:DATABASE_TYPE="sqlite"
$env:STORAGE_TYPE="local"
$env:RUNTIME_MODELS_DIR="models"
$env:RUNTIME_SQLITE_PATH="sqlite/funasr_tasks.db"
$env:RUNTIME_TEMP_DIR="temp"
$env:RUNTIME_OFFLINE_RESULT_DIR="results/offline"
$env:RUNTIME_LOCAL_FILES_DIR="files"
```

### 模型选择

后端通过 `engines.models.<mode>.enabled` 选择：

```yaml
engines:
  device: "cpu"
  disable_model_update: true
  auto_model_download: true
  models:
    offline:
      enabled: onnx
    online:
      enabled: pt
```

默认模型不再要求写进 `config.yaml`，而是集中在 catalog：

- OFFLINE 默认模型：`src/engine_runtime/engines/offline/catalog.py`
- ONLINE 默认模型：`src/engine_runtime/engines/online/catalog.py`

如果确实要覆盖模型，可以在 `config.yaml` 中显式写 `pt` 或 `onnx` 子项；未写时使用 catalog 默认值。

### 当前默认模型

OFFLINE ONNX 默认 ASR 已使用 SeACo：

```text
marxyz/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-onnx
```

ONLINE ONNX 的 final ASR 默认也使用 SeACo：

```text
marxyz/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-onnx
```

ONLINE ONNX 的 streaming ASR 仍是普通 online Paraformer；因此热词主要影响 final segment 的二次识别结果，不保证影响 partial。

### 数据库、存储、通知

- `database.type` 显式选择 `sqlite` 或 `mysql`；未配置时默认 `sqlite`。
- `storage.type` 显式选择 `local` 或 `s3`；未配置时默认 `local`。
- 数据库或存储连接失败时启动失败，运行时存储失败时任务进入失败/重试并保留临时文件；两者都不会切换后端。

已有 MySQL 数据库升级到包含重识别接口的版本时，必须先备份数据库并执行：

```sql
ALTER TABLE `offline_tasks`
  ADD COLUMN `source_task_id` VARCHAR(36) NULL COMMENT '重识别来源 OFFLINE 任务 ID；普通上传任务为空' AFTER `filename`,
  ADD KEY `idx_offline_tasks_source_task_id` (`source_task_id`);

ALTER TABLE `spk_tasks`
  ADD COLUMN `source_task_id` VARCHAR(36) NULL COMMENT '重识别来源 SPK 任务 ID；普通上传任务为空' AFTER `filename`,
  ADD KEY `idx_spk_tasks_source_task_id` (`source_task_id`);
```

应用不会自动迁移已有表；升级顺序为：备份数据库、执行上述 SQL、再启动新代码。

- 应用会创建缺失表，但不会自动补列；已有表结构缺列时会启动失败。
- `notifications.enabled: false` 时跳过 RocketMQ；启用后连接失败只告警，不影响识别主流程。
- RocketMQ 是可选客户端；启用通知的运行环境需安装 `rocketmq-client-python`，缺失时启动表会显示 `client_dependency_missing`。

```yaml
database:
  type: sqlite
  sqlite:
    pool_size: 5
    pool_recycle: 3600
    echo: false
  mysql:
    host: "127.0.0.1"
    port: 3306
    username: "funasr"
    password: ""
    database: "funasr_tasks"

storage:
  type: local
  local:
    prefix: "audio"
  s3:
    endpoint: "https://s3.example.com"
    access_key: ""
    secret_key: ""
    bucket: "funasr-audio"
    region: "us-east-1"
    prefix: "audio"
```

旧的 `database.url`、`database.enabled`、`storage.s3.enabled` 和
`storage.s3.save_original` 已删除。配置加载器会给出迁移提示，不会猜测旧配置的意图。

### 日志

控制台和文件日志由 `logging` 段统一管理。文件日志默认写入
`./logs/funasr.log`，使用 UTF-8 编码，并按 `max_size_mb`、`backup_count`
滚动。目录会自动创建；文件处理器无法建立时服务启动失败。

INFO 日志会记录上传、归档、索引、通知和任务状态，但不会记录原始文件名、用户路径、
识别内容、邮箱、凭证或消息正文；这些诊断明细只在 DEBUG 中以脱敏形式出现。

## HTTP API

### OFFLINE 上传识别

```bash
curl.exe -X POST "http://127.0.0.1:8000/offline/recognize" ^
  -F "file=@test.wav" ^
  -F "vip=false" ^
  -F "hotwords=[{\"weight\":100,\"hotword\":\"篮子\"},{\"weight\":80,\"hotword\":\"直播\"}]"
```

响应示例：

```json
{
  "status": "success",
  "task_id": "task-id",
  "filename": "test.wav",
  "message": "任务已加入队列，正在处理",
  "hotwords": "[{\"weight\":100,\"hotword\":\"篮子\"}]",
  "hotword_id": null
}
```

支持表单字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `file` | file | 是 | 音频文件 |
| `email` | string | 否 | 任务完成后用于后处理通知 |
| `hotwords` | string | 否 | 严格 JSON 热词数组 |
| `hotword_id` | int | 否 | 从数据库读取热词 |
| `vip` | bool | 否 | VIP 优先队列 |

### 查询 OFFLINE 任务

```bash
curl.exe "http://127.0.0.1:8000/tasks/{task_id}"
```

完成后返回：

```json
{
  "task_id": "task-id",
  "source_task_id": null,
  "filename": "test.wav",
  "status": "completed",
  "processing_time": 1.23,
  "error": null,
  "s3_key": "audio/...",
  "file_hash": "sha256...",
  "vip": false,
  "result": {
    "full_text": "识别结果",
    "segments": [],
    "word_timestamps": []
  }
}
```

### OFFLINE 重识别

```bash
curl.exe -X POST "http://127.0.0.1:8000/offline/tasks/{task_id}/rerecognize" ^
  -H "Content-Type: application/json" ^
  -d "{\"vip\":true,\"hotword_id\":2}"
```

请求体可省略。`email`、`vip`、`hotwords`、`hotword_id` 未传或为 `null` 时继承来源任务；
使用非空 `hotwords` 会清除原 `hotword_id`，使用非空 `hotword_id` 会清除原 `hotwords`，
两者不能同时提供非空值。接口从当前配置的 local/S3 归档恢复音频，创建并返回新的
`task_id` 和 `source_task_id`，不会覆盖来源任务。

### SPK 上传识别

```bash
curl.exe -X POST "http://127.0.0.1:8000/spk/recognize" ^
  -F "file=@test.wav" ^
  -F "vip=false"
```

### 查询 SPK 任务

```bash
curl.exe "http://127.0.0.1:8000/spk/tasks/{task_id}"
```

### SPK 重识别

```bash
curl.exe -X POST "http://127.0.0.1:8000/spk/tasks/{task_id}/rerecognize" ^
  -H "Content-Type: application/json" ^
  -d "{\"vip\":true}"
```

请求体可省略。`email`、`vip` 未传或为 `null` 时继承来源任务；接口从当前配置的
local/S3 归档恢复音频并创建新 SPK 任务。

### 健康检查

```bash
curl.exe "http://127.0.0.1:8000/health"
```

健康检查返回当前启用模式、模型加载状态、runtime service 状态和基础设施摘要。

## ONLINE WebSocket

连接地址：

```text
ws://127.0.0.1:8000/online/stream
```

查询参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `sample_rate` | `16000` | 客户端 PCM 采样率 |
| `hotwords` | 空 | 严格 JSON 热词数组 |
| `hotword_id` | 空 | 从数据库读取热词 |

协议：

```text
START                 开始或重置会话
<binary pcm chunk>    16kHz mono int16 PCM 音频分片
STOP                  结束会话并返回最终结果
```

连接建立后会先返回 ready：

```json
{
  "event": "ready",
  "mode": "2pass",
  "sample_rate": 16000,
  "decode_interval": 0.3,
  "queue_max_chunks": 20,
  "hotword_id": null,
  "hotword_count": 2
}
```

实时 partial 示例：

```json
{
  "mode": "2pass-online",
  "text": "正在识别",
  "partial": "正在识别",
  "partial_start_ms": 0,
  "duration_ms": 1200,
  "is_final": false,
  "metrics": {}
}
```

final segment 示例：

```json
{
  "mode": "2pass-offline",
  "text": "最终识别结果。",
  "sentences": [
    {
      "text": "最终识别结果。",
      "raw_text": "最终识别结果",
      "start": 0,
      "end": 1200,
      "is_final": true,
      "timestamp": [],
      "tokens": []
    }
  ],
  "duration_ms": 1200,
  "is_final": true,
  "metrics": {}
}
```

停止后返回：

```json
{
  "event": "stopped",
  "metrics": {}
}
```

## 热词说明

对外 API 统一只接受严格 JSON 数组：

```json
[
  {"weight": 100, "hotword": "篮子"},
  {"weight": 80, "hotword": "直播"},
  {"weight": 60, "hotword": "蛇哥"}
]
```

规则：

- 每个元素必须且只能包含 `weight` 和 `hotword`。
- `weight` 必须是整数，范围 `1-100`。
- `hotword` 必须是非空字符串。
- 自定义 `hotwords` 优先级最高，其次是 `hotword_id`，最后是 `hotwords.default_ids`。

模型侧格式由 `src/core/hotwords/loader.py` 统一转换：

- 默认 plain：`"篮子 直播 蛇哥"`
- weighted：`[[100, "篮子"], [80, "直播"]]`

SeACo ONNX 使用 NN 热词能力，不是 C++ Runtime 的 WFST 加权热词文件语义。当前 SeACo ONNX 调用会把热词整理成空格分隔字符串，例如：

```text
篮子 直播 蛇哥
```

因此：

- 权重会保留在 API/数据库结构中，但 SeACo ONNX 推理时只使用词面。
- 官方 C++ Runtime 的 `热词 权重` 文件格式不等于当前 Python `funasr_onnx` 调用格式。
- OFFLINE ONNX 使用 SeACo 时支持热词偏置，不是后处理文本替换。
- ONLINE ONNX 只有 final ASR 使用 SeACo 热词；streaming partial 不保证被热词影响。

## 模型流程

### OFFLINE PT

```text
audio file
  -> FunASR AutoModel.generate(hotword=...)
  -> ASR sentence_info / timestamp
  -> 有可用时间戳：共享 SPK runtime 二次识别并合并说话人
  -> 无可用时间戳：跳过 SPK，直接返回纯 ASR 结果
```

### OFFLINE ONNX

```text
audio file
  -> funasr_onnx VAD
  -> funasr_onnx Paraformer / SeacoParaformer ASR
  -> funasr_onnx CT-PUNC
  -> 共享 SPK runtime
  -> ASR timestamp + speaker segment alignment
  -> RecognitionResult
```

### ONLINE

```text
PCM stream
  -> streaming ASR partial
  -> streaming VAD 锁定语音段
  -> final ASR 二次识别
  -> punctuation
  -> WebSocket JSON event
```

## OFFLINE 与 SPK 关系

- `offline` 启用后，OFFLINE 会依赖共享 SPK runtime 做说话人二次校验。
- `spk` 启用后，才开放 standalone SPK 上传和查询端点。
- OFFLINE 不调用 standalone SPK HTTP API，也不依赖 standalone `SpkTaskService`。
- 共享 SPK runtime 由 runtime cache 复用，避免同一模型被重复加载。
- OFFLINE PT 在 ASR 没有可用时间戳时不会强行跑 SPK，会返回纯 ASR 结果。

## 队列与结果处理

OFFLINE/SPK 上传任务进入同一个优先队列：

- 新任务 ID 为 32 位小写十六进制 UUID（不含横线），历史 36 位 ID 继续兼容。
- VIP 任务优先。
- 队列 worker 启动后异步消费任务。
- 上传前会检查对应 runtime 和队列 readiness，避免任务入库后无人处理。

OFFLINE 成功后的副作用由 hook 处理：

- 保存识别文本、segments、word timestamps。
- 按 `storage.type` 将原始音频归档到 S3 或本地目录，不做故障回退。
- 写出 `data/results/offline/{task_id}.txt`。
- 成功归档后清理上传临时文件。
- 通知开启时发送 RocketMQ 完成消息。

## 开发与测试

运行测试：

```bash
uv run python -m pytest
```

针对当前热词和 ONNX 相关能力，可优先跑：

```bash
uv run python -m pytest tests/test_hotword_formats.py
uv run python -m pytest tests/test_offline_onnx_recognizer.py
uv run python -m pytest tests/test_online_onnx_adapter_modules.py
```

文档/空白校验：

```bash
git diff --check -- README.md
```

## 排障

### 模型没有下载

确认：

- `engines.auto_model_download: true`
- 网络可访问 ModelScope / HuggingFace 相关源
- `runtime.models_dir` 可写

生产离线部署可提前放入模型目录，并设置：

```yaml
engines:
  auto_model_download: false
```

### ONNX 热词没有效果

优先确认：

- OFFLINE ONNX ASR 模型是 SeACo 模型。
- 日志中出现 `hotword_mode=seaco`。
- 传入的是严格 JSON 数组，而不是 `热词 权重` 文本文件。
- ONLINE ONNX 只看 final ASR 结果，partial 不代表最终热词效果。

### ONLINE 没有响应

确认：

- `engines.enabled` 包含 `online`。
- WebSocket 已发送 `START`。
- 音频是 16kHz mono int16 PCM。
- 发送的是二进制音频分片，不是 wav 文件字节流。

### OFFLINE 只有纯文本没有说话人

OFFLINE PT 如果 ASR 结果没有可用 timestamp，会跳过 SPK 并返回纯 ASR；这是为了避免没有时间轴时错误合并说话人。

### 数据库启动失败

先查看启动状态表中的 `database` 行。应用只使用 `database.type` 选中的数据库，
连接失败时不会创建或切换到 SQLite。应用只创建缺失表，不做自动迁移补列；已有
SQLite/MySQL 表结构缺少当前必需字段时，需要手工迁移或重建数据库。

### 日志文件没有内容

确认 `logging.file.enabled: true`，并检查启动状态表中的 `logging_file` 行是否为
`ready`。内置启动和 `uvicorn main:app` 都会把 Uvicorn 与应用日志交给同一组处理器。

## 版本状态

v1.0 当前稳定边界：

- OFFLINE/PT/ONNX 后端可按模式切换。
- OFFLINE ONNX 默认 SeACo，支持模型级热词偏置。
- ONLINE PT/ONNX 双通路保留；ONNX final ASR 支持 SeACo 热词。
- SPK runtime 与 OFFLINE 共享，避免重复模型实例。
- 上传、队列、任务查询、结果落盘、音频备份、通知链路已形成闭环。
