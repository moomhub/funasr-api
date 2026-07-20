"""Import-level architecture guards for the decoupled application."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _violations(directory: str, forbidden: tuple[str, ...]) -> list[str]:
    violations = []
    for path in (SRC / directory).rglob("*.py"):
        for imported in _imports(path):
            if imported.startswith(forbidden):
                violations.append(f"{path.relative_to(ROOT)} -> {imported}")
    return violations


def test_api_depends_only_on_application_and_transport_safe_modules():
    assert _violations("api", ("src.database", "src.storage", "src.task_queue")) == []


def test_application_does_not_depend_on_scheduler_or_concrete_database():
    assert _violations(
        "application",
        ("src.task_queue", "src.database", "src.core.container"),
    ) == []


def test_engine_runtime_is_independent_of_api_queue_and_database():
    assert _violations(
        "engine_runtime",
        ("src.api", "src.task_queue", "src.database"),
    ) == []


def test_module_level_global_initializers_are_gone():
    forbidden_names = {
        "get_config",
        "init_config",
        "get_db",
        "init_database",
        "get_container",
        "init_container",
        "get_task_queue",
        "init_task_queue",
        "get_engine_model_manager",
        "init_engine_model_manager",
    }
    found = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in forbidden_names:
                found.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
    assert found == []


def test_removed_python_compatibility_symbols_are_gone():
    forbidden_by_file = {
        "core/adapters.py": {"save_audio"},
        "engine_runtime/loaders/pt_loader.py": {"load_speaker_model"},
        "engine_runtime/services/__init__.py": {
            "preload_mode",
            "preload_service",
            "_service_by_name",
        },
        "engine_runtime/manager.py": {"get_model_name"},
        "core/config/loader.py": {
            "get_runtime_root_dir",
            "resolve_runtime_path",
            "get_server_config",
            "get_logging_config",
        },
    }
    found = []
    for relative_path, forbidden_names in forbidden_by_file.items():
        path = SRC / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in forbidden_names:
                found.append(f"{relative_path}:{node.lineno}:{node.name}")
    assert found == []


def test_external_only_package_roots_do_not_reexport_implementations():
    package_roots = (
        "application/__init__.py",
        "engine_runtime/__init__.py",
        "task_queue/__init__.py",
        "database/__init__.py",
        "core/config/__init__.py",
        "notifications/__init__.py",
    )
    violations = []
    for relative_path in package_roots:
        path = SRC / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body):
            violations.append(relative_path)
    assert violations == []


def test_strict_contract_layers_do_not_default_missing_members():
    strict_files = (
        "api/online.py",
        "api/system.py",
        "application/offline.py",
        "application/online.py",
        "application/speaker.py",
        "core/container.py",
        "database/repositories/offline_tasks.py",
        "task_queue/hooks.py",
        "task_queue/queue.py",
    )
    forbidden_members = {
        "_loaded",
        "close",
        "critical",
        "db",
        "email",
        "file_hash",
        "hotword_repository",
        "name",
        "recover_stale_processing",
        "vip",
        "word_timestamps",
    }
    violations = []
    for relative_path in strict_files:
        path = SRC / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "getattr" or len(node.args) < 2:
                continue
            member = node.args[1]
            if isinstance(member, ast.Constant) and member.value in forbidden_members:
                violations.append(f"{relative_path}:{node.lineno}:{member.value}")
    assert violations == []
