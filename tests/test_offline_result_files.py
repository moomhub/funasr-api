import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config.loader import ConfigLoader
from src.core.results import RecognitionResult
from src.task_queue.hooks import OfflineTaskContext, TextResultFileHook


@pytest.mark.asyncio
async def test_text_result_file_hook_writes_offline_txt(tmp_path):
    root_dir = tmp_path / "runtime"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
runtime:
  root_dir: "{root_dir.as_posix()}"
  offline_result_dir: results/offline
""",
        encoding="utf-8",
    )
    config = ConfigLoader(str(config_path))

    await TextResultFileHook(config.get_runtime_paths()["offline_result_dir"]).on_success(
        OfflineTaskContext(task_id="task-1", filename="demo.wav", audio_path="demo.wav"),
        RecognitionResult(mode="offline", full_text="识别完成"),
    )

    result_path = root_dir / "results" / "offline" / "task-1.txt"
    assert result_path.read_text(encoding="utf-8") == "识别完成"
