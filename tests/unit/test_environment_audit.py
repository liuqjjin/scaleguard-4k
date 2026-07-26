import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_audit_module() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts/bootstrap/audit_environment.py"
    spec = importlib.util.spec_from_file_location("scaleguard_environment_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load environment audit module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_AUDIT = _load_audit_module()
InstalledPackage: Any = _AUDIT.InstalledPackage
audit_dependencies: Any = _AUDIT.audit_dependencies
pinned_requirements: Any = _AUDIT.pinned_requirements
exact_pin: Any = _AUDIT.exact_pin
audit_4kagent_entrypoints: Any = _AUDIT.audit_4kagent_entrypoints


def _package(name: str, version: str, *requirements: str) -> InstalledPackage:
    return InstalledPackage(name=name, version=version, requirements=requirements)


def test_dependency_audit_accepts_satisfied_and_inactive_requirements() -> None:
    packages = {
        "parent": _package(
            "parent",
            "1.0",
            "dependency>=2",
            "optional-package; extra == 'training'",
        ),
        "dependency": _package("dependency", "2.1"),
    }

    issues, overrides = audit_dependencies(
        packages,
        allow_4kagent_runtime_overrides=False,
    )

    assert issues == []
    assert overrides == []


def test_dependency_audit_accepts_only_the_exact_4kagent_overrides() -> None:
    packages = {
        "pyiqa": _package("pyiqa", "0.1.13", "transformers==4.37.2"),
        "transformers": _package("transformers", "5.5.0"),
        "hpsv2": _package(
            "hpsv2",
            "1.2.0",
            "protobuf<4",
            "pytest==7.2.0",
            "pytest-split==0.8.0",
        ),
        "protobuf": _package("protobuf", "6.33.5"),
    }

    issues, overrides = audit_dependencies(
        packages,
        allow_4kagent_runtime_overrides=True,
    )

    assert issues == []
    assert overrides == [
        {
            "parent": "hpsv2",
            "parent_version": "1.2.0",
            "dependency": "protobuf",
            "required": "<4",
            "installed": "6.33.5",
            "reason": "HPS scoring does not import protobuf; see ADR 0005",
        },
        {
            "parent": "hpsv2",
            "parent_version": "1.2.0",
            "dependency": "pytest",
            "required": "==7.2.0",
            "installed": None,
            "reason": (
                "test-only HPS metadata omitted from the inference environment; see ADR 0005"
            ),
        },
        {
            "parent": "hpsv2",
            "parent_version": "1.2.0",
            "dependency": "pytest-split",
            "required": "==0.8.0",
            "installed": None,
            "reason": (
                "test-only HPS metadata omitted from the inference environment; see ADR 0005"
            ),
        },
        {
            "parent": "pyiqa",
            "parent_version": "0.1.13",
            "dependency": "transformers",
            "required": "==4.37.2",
            "installed": "5.5.0",
            "reason": "security-updated 4KAgent Qwen runtime; see ADR 0005",
        },
    ]


def test_dependency_audit_rejects_a_changed_or_additional_mismatch() -> None:
    packages = {
        "pyiqa": _package("pyiqa", "0.1.13", "transformers==4.37.2"),
        "transformers": _package("transformers", "5.6.0"),
        "hpsv2": _package(
            "hpsv2",
            "1.2.0",
            "protobuf<4",
            "pytest==7.2.0",
            "pytest-split==0.8.0",
        ),
        "protobuf": _package("protobuf", "6.33.5"),
        "other": _package("other", "1.0", "dependency==2"),
        "dependency": _package("dependency", "3.0"),
    }

    issues, overrides = audit_dependencies(
        packages,
        allow_4kagent_runtime_overrides=True,
    )

    assert {override["dependency"] for override in overrides} == {
        "protobuf",
        "pytest",
        "pytest-split",
    }
    assert {issue["parent"] for issue in issues} == {"pyiqa", "other"}
    assert any(
        "not observed exactly once" in issue["issue"]
        for issue in issues
        if issue["parent"] == "pyiqa"
    )


def test_pin_parser_reads_hashed_requirements_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text(
        """# generated
Package_Name==1.2.3 \\
    --hash=sha256:abc
other==4.5
""",
        encoding="utf-8",
    )

    assert pinned_requirements(lock) == {
        "package-name": "1.2.3",
        "other": "4.5",
    }

    lock.write_text("package==1\npackage==2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="conflicting pins"):
        pinned_requirements(lock)


def test_exact_pin_rejects_ranges_and_markers() -> None:
    assert exact_pin("ScaleGuard_4K==0.1.0.dev0") == (
        "scaleguard-4k",
        "0.1.0.dev0",
    )

    for value in ("pyiqa>=0.1", "pyiqa==0.1; python_version > '3.10'"):
        with pytest.raises(ValueError, match="exact NAME==VERSION"):
            exact_pin(value)


def test_4kagent_entrypoint_audit_uses_a_minimal_offline_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], Path, dict[str, str]]] = []
    for relative in _AUDIT._4KAGENT_ENTRYPOINTS:
        script = tmp_path / relative
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("raise SystemExit(0)\n", encoding="utf-8")

    def run(
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        **_kwargs: object,
    ) -> object:
        calls.append((argv, cwd, env))
        return _AUDIT.subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setenv("SHOULD_NOT_REACH_TOOL", "secret")
    monkeypatch.setattr(_AUDIT.subprocess, "run", run)

    checks, issues = audit_4kagent_entrypoints(tmp_path)

    assert issues == []
    assert checks == [
        {"module": f"entrypoint:{relative}", "symbols": ["--help"]}
        for relative in _AUDIT._4KAGENT_ENTRYPOINTS
    ]
    assert len(calls) == len(_AUDIT._4KAGENT_ENTRYPOINTS)
    for argv, cwd, environment in calls:
        assert argv[-1] == "--help"
        assert cwd == Path(argv[-2]).parent
        assert "SHOULD_NOT_REACH_TOOL" not in environment
        assert environment["HF_HUB_OFFLINE"] == "1"
        assert environment["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] == "1"


def test_4kagent_entrypoint_audit_rejects_missing_scripts(tmp_path: Path) -> None:
    checks, issues = audit_4kagent_entrypoints(tmp_path)

    assert checks == []
    assert [issue["requirement"] for issue in issues] == [
        f"entrypoint:{relative}" for relative in _AUDIT._4KAGENT_ENTRYPOINTS
    ]
