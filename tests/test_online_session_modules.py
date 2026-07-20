import ast
from pathlib import Path

from src.engine_runtime.engines.online.onnx.session import (
    OnlineOnnxRealtimeSession as ConcreteOnnxSession,
)
from src.engine_runtime.engines.online.realtime_session import (
    OnlineRealtimeSession as ConcreteRealtimeSession,
)
from src.engine_runtime.engines.online.session_manager import (
    OnlineOnnxRealtimeSession as LegacyOnnxSession,
    OnlineRealtimeSession as LegacyRealtimeSession,
)


ROOT = Path(__file__).resolve().parents[1]
ONLINE = ROOT / "src" / "engine_runtime" / "engines" / "online"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_legacy_session_manager_reexports_split_sessions():
    assert LegacyRealtimeSession is ConcreteRealtimeSession
    assert LegacyOnnxSession is ConcreteOnnxSession
    assert ConcreteRealtimeSession.__module__.endswith(".realtime_session")
    assert ConcreteOnnxSession.__module__.endswith(".onnx.session")


def test_legacy_session_manager_contains_no_implementation_classes():
    path = ONLINE / "session_manager.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert not any(isinstance(node, ast.ClassDef) for node in tree.body)


def test_common_realtime_session_does_not_depend_on_onnx_backend():
    assert not any(
        imported.startswith("src.engine_runtime.engines.online.onnx")
        for imported in _imports(ONLINE / "realtime_session.py")
    )


def test_recognizers_import_concrete_session_modules():
    pt_imports = _imports(ONLINE / "pt" / "recognizer.py")
    onnx_imports = _imports(ONLINE / "onnx" / "recognizer.py")
    assert "src.engine_runtime.engines.online.realtime_session" in pt_imports
    assert "src.engine_runtime.engines.online.onnx.session" in onnx_imports
    assert "src.engine_runtime.engines.online.session_manager" not in pt_imports
    assert "src.engine_runtime.engines.online.session_manager" not in onnx_imports
