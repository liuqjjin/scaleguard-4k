import importlib.util
import json
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
installation_file_inventory: Any = _AUDIT.installation_file_inventory
runtime_import_origin: Any = _AUDIT.runtime_import_origin
runtime_installation_identity: Any = _AUDIT.runtime_installation_identity
validate_python_identity: Any = _AUDIT.validate_python_identity
write_atomic: Any = _AUDIT.write_atomic


def _package(name: str, version: str, *requirements: str) -> InstalledPackage:
    return InstalledPackage(name=name, version=version, requirements=requirements)


class FakeDistribution:
    def __init__(
        self,
        site_packages: Path,
        files: list[Path],
        *,
        name: str = "example",
        version: str = "1.0",
    ) -> None:
        self.site_packages = site_packages
        self.files = files
        self.metadata = {"Name": name}
        self.version = version

    def locate_file(self, path: Path) -> Path:
        return self.site_packages / path


def test_environment_receipt_writer_uses_a_private_exclusive_temporary_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "must-not-change.txt"
    target.write_text("sentinel\n", encoding="utf-8")
    fixed_temporary = tmp_path / ".environment.json.tmp"
    fixed_temporary.symlink_to(target)
    output = tmp_path / "environment.json"

    write_atomic(output, {"schema_version": 1, "status": "passed"})

    assert target.read_text(encoding="utf-8") == "sentinel\n"
    assert fixed_temporary.is_symlink()
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "status": "passed",
    }
    assert output.stat().st_mode & 0o777 == 0o600

    linked_output = tmp_path / "linked.json"
    linked_output.symlink_to(target)
    with pytest.raises(RuntimeError, match="output is unsafe"):
        write_atomic(linked_output, {"status": "forged"})
    assert target.read_text(encoding="utf-8") == "sentinel\n"


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


def test_environment_identity_requires_the_fixed_venv_but_preserves_python_symlink(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / ".venv"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "pyvenv.cfg").write_text("home = /base\n", encoding="utf-8")
    base_prefix = tmp_path / ".runtime/python/base"
    base_executable = base_prefix / "bin/python"
    base_executable.parent.mkdir(parents=True)
    base_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable = prefix / "bin/python"
    executable.symlink_to(base_executable)
    identity = {
        "executable": str(executable),
        "executable_realpath": str(base_executable),
        "prefix": str(prefix),
        "base_prefix": str(base_prefix),
        "version": "3.10.18",
        "implementation": "CPython",
        "platform": "test-platform",
    }

    validate_python_identity("scaleguard", tmp_path, identity)
    wrong_identity = {**identity, "prefix": str(tmp_path / "unexpected")}
    with pytest.raises(RuntimeError, match="expected the fixed environment"):
        validate_python_identity("scaleguard", tmp_path, wrong_identity)


def test_environment_audit_main_fails_closed_on_the_wrong_venv_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text("", encoding="utf-8")
    output = tmp_path / "receipt.json"
    wrong_prefix = tmp_path / "unexpected"
    identity = {
        "executable": str(wrong_prefix / "bin/python"),
        "executable_realpath": str(tmp_path / "base/bin/python"),
        "prefix": str(wrong_prefix),
        "base_prefix": str(tmp_path / "base"),
        "version": _AUDIT.platform.python_version(),
        "implementation": "CPython",
        "platform": "test-platform",
    }
    monkeypatch.setattr(_AUDIT, "current_python_identity", lambda: identity)
    monkeypatch.setattr(_AUDIT, "installed_packages", dict)
    monkeypatch.setattr(
        _AUDIT,
        "_installation_file_inventory",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("identity rejected")),
    )
    monkeypatch.setattr(_AUDIT, "audit_dependencies", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(_AUDIT, "audit_runtime_imports", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(
        _AUDIT.sys,
        "argv",
        [
            "audit_environment.py",
            "--name",
            "scaleguard",
            "--project-root",
            str(tmp_path),
            "--lock",
            str(lock),
            "--output",
            str(output),
            "--expected-python",
            identity["version"],
        ],
    )

    assert _AUDIT.main() == 1
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 2
    assert receipt["status"] == "failed"
    assert any(
        issue["requirement"] == "scaleguard virtual environment identity"
        for issue in receipt["issues"]
    )


def test_installation_merkle_changes_when_an_installed_source_is_modified(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "environment"
    (environment / "bin").mkdir(parents=True)
    site_packages = environment / "lib/python3.10/site-packages"
    package = site_packages / "example.py"
    record = site_packages / "example-1.0.dist-info/RECORD"
    record.parent.mkdir(parents=True)
    package.write_text("VALUE = 1\n", encoding="utf-8")
    record.write_text("example.py,,\nexample-1.0.dist-info/RECORD,,\n", encoding="utf-8")
    distribution = FakeDistribution(
        site_packages,
        [Path("example.py"), Path("example-1.0.dist-info/RECORD")],
    )

    before = installation_file_inventory([distribution], environment_prefix=environment)
    package.write_text("VALUE = 2\n", encoding="utf-8")
    after = installation_file_inventory([distribution], environment_prefix=environment)

    assert before["algorithm"] == "sha256-merkle-v1"
    assert before["environment_root"] == str(environment)
    assert before["distribution_count"] == 1
    assert before["file_count"] == 2
    assert before["merkle_root"] != after["merkle_root"]
    assert before["distributions"][0]["merkle_root"] != after["distributions"][0]["merkle_root"]


def test_installation_inventory_rejects_record_escape_and_symlink(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "environment"
    (environment / "bin").mkdir(parents=True)
    site_packages = environment / "lib/python3.10/site-packages"
    record = site_packages / "example-1.0.dist-info/RECORD"
    record.parent.mkdir(parents=True)
    record.write_text("example-1.0.dist-info/RECORD,,\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")

    escaping = FakeDistribution(
        site_packages,
        [Path("../../../../outside.py"), Path("example-1.0.dist-info/RECORD")],
    )
    with pytest.raises(RuntimeError, match="escapes its environment"):
        installation_file_inventory([escaping], environment_prefix=environment)

    target = site_packages / "target.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    linked = site_packages / "example.py"
    linked.symlink_to(target)
    symlinked = FakeDistribution(
        site_packages,
        [Path("example.py"), Path("example-1.0.dist-info/RECORD")],
    )
    with pytest.raises(RuntimeError, match="symbolic link"):
        installation_file_inventory([symlinked], environment_prefix=environment)


def test_installation_boundary_rejects_unowned_imports_and_executables(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "environment"
    scripts = environment / "bin"
    scripts.mkdir(parents=True)
    site_packages = environment / "lib/python3.10/site-packages"
    package = site_packages / "example.py"
    record = site_packages / "example-1.0.dist-info/RECORD"
    record.parent.mkdir(parents=True)
    package.write_text("VALUE = 1\n", encoding="utf-8")
    record.write_text("example.py,,\nexample-1.0.dist-info/RECORD,,\n", encoding="utf-8")
    distribution = FakeDistribution(
        site_packages,
        [Path("example.py"), Path("example-1.0.dist-info/RECORD")],
    )

    for name in ("rogue.pth", "rogue.py", "rogue.pyc", "plugin.json"):
        rogue_file = site_packages / name
        rogue_file.write_text("unowned\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="unowned site-packages"):
            installation_file_inventory([distribution], environment_prefix=environment)
        rogue_file.unlink()

    rogue_executable = scripts / "rogue"
    rogue_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    rogue_executable.chmod(0o755)
    with pytest.raises(RuntimeError, match="unowned executable"):
        installation_file_inventory([distribution], environment_prefix=environment)


def test_runtime_installation_identity_changes_when_the_interpreter_is_modified(
    tmp_path: Path,
) -> None:
    base_prefix = tmp_path / ".runtime/python/cpython-3.10"
    base_executable = base_prefix / "bin/python3.10"
    stdlib = base_prefix / "lib/python3.10"
    base_executable.parent.mkdir(parents=True)
    stdlib.mkdir(parents=True)
    base_executable.write_bytes(b"python-runtime-v1")
    (stdlib / "os.py").write_text("name = 'posix'\n", encoding="utf-8")
    prefix = tmp_path / ".venv"
    (prefix / "bin").mkdir(parents=True)
    configuration = prefix / "pyvenv.cfg"
    configuration.write_text(f"home = {base_executable.parent}\n", encoding="utf-8")
    executable = prefix / "bin/python"
    executable.symlink_to(base_executable)
    identity = {
        "executable": str(executable),
        "executable_realpath": str(base_executable),
        "prefix": str(prefix),
        "base_prefix": str(base_prefix),
        "version": "3.10.18",
        "implementation": "CPython",
        "platform": "test-platform",
    }

    before = runtime_installation_identity(
        tmp_path,
        identity,
        name="scaleguard",
        base_executable=base_executable,
        stdlib_root=stdlib,
    )
    base_executable.write_bytes(b"python-runtime-v2")
    after = runtime_installation_identity(
        tmp_path,
        identity,
        name="scaleguard",
        base_executable=base_executable,
        stdlib_root=stdlib,
    )

    assert before["interpreter"]["sha256"] != after["interpreter"]["sha256"]
    assert before["merkle_root"] != after["merkle_root"]
    configuration.unlink()
    forged_configuration = tmp_path / "forged-pyvenv.cfg"
    forged_configuration.write_text("home = /forged\n", encoding="utf-8")
    configuration.symlink_to(forged_configuration)
    with pytest.raises(RuntimeError, match="virtual environment marker is missing or unsafe"):
        runtime_installation_identity(
            tmp_path,
            identity,
            name="scaleguard",
            base_executable=base_executable,
            stdlib_root=stdlib,
        )


def test_runtime_installation_identity_accepts_only_managed_uv_aliases(
    tmp_path: Path,
) -> None:
    managed_root = tmp_path / ".runtime/python"
    base_prefix = managed_root / "cpython-3.10.18-linux-x86_64-gnu"
    base_executable = base_prefix / "bin/python3.10"
    stdlib = base_prefix / "lib/python3.10"
    base_executable.parent.mkdir(parents=True)
    stdlib.mkdir(parents=True)
    base_executable.write_bytes(b"managed-python")
    (stdlib / "os.py").write_text("name = 'posix'\n", encoding="utf-8")
    alias_prefix = managed_root / "cpython-3.10"
    alias_prefix.symlink_to(base_prefix.name, target_is_directory=True)
    alias_executable = alias_prefix / "bin/python3.10"

    prefix = tmp_path / ".venv"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "pyvenv.cfg").write_text(
        f"home = {base_executable.parent}\n",
        encoding="utf-8",
    )
    executable = prefix / "bin/python"
    executable.symlink_to(alias_executable)
    identity = {
        "executable": str(executable),
        "executable_realpath": str(base_executable),
        "prefix": str(prefix),
        "base_prefix": str(base_prefix),
        "version": "3.10.18",
        "implementation": "CPython",
        "platform": "test-platform",
    }

    result = runtime_installation_identity(
        tmp_path,
        identity,
        name="scaleguard",
        base_executable=alias_executable,
        stdlib_root=stdlib,
    )

    assert result["base_runtime"]["executable"] == str(alias_executable)
    assert result["base_runtime"]["executable_realpath"] == str(base_executable)
    assert result["base_runtime"]["executable_alias_count"] == 1
    assert result["base_runtime"]["executable_aliases"] == [
        {
            "path": "cpython-3.10",
            "target": base_prefix.name,
            "resolved": str(base_prefix),
        }
    ]

    outside = tmp_path / "outside"
    outside.mkdir()
    escaped_alias = managed_root / "escaped"
    escaped_alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="alias target escapes"):
        runtime_installation_identity(
            tmp_path,
            identity,
            name="scaleguard",
            base_executable=escaped_alias / "python",
            stdlib_root=stdlib,
        )


def test_runtime_import_origin_is_bound_to_its_environment_or_checkout(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "environment"
    checkout = tmp_path / "checkout"
    environment_module = environment / "lib/python3.10/site-packages/transformers.py"
    checkout_module = checkout / "pipeline/the4kagent_pipeline.py"
    unexpected_module = tmp_path / "unexpected.py"
    for path in (environment_module, checkout_module, unexpected_module):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("VALUE = 1\n", encoding="utf-8")

    transformers = ModuleType("transformers")
    transformers.__spec__ = importlib.util.spec_from_file_location(
        "transformers",
        environment_module,
    )
    pipeline = ModuleType("pipeline.the4kagent_pipeline")
    pipeline.__spec__ = importlib.util.spec_from_file_location(
        "pipeline.the4kagent_pipeline",
        checkout_module,
    )
    unexpected = ModuleType("transformers")
    unexpected.__spec__ = importlib.util.spec_from_file_location(
        "transformers",
        unexpected_module,
    )

    assert runtime_import_origin(
        "4kagent",
        "transformers",
        transformers,
        environment_root=environment,
        checkout_root=checkout,
        owned_environment_files=frozenset({environment_module}),
    ) == str(environment_module)
    assert runtime_import_origin(
        "4kagent",
        "pipeline.the4kagent_pipeline",
        pipeline,
        environment_root=environment,
        checkout_root=checkout,
    ) == str(checkout_module)
    with pytest.raises(RuntimeError, match="outside its expected root"):
        runtime_import_origin(
            "4kagent",
            "transformers",
            unexpected,
            environment_root=environment,
            checkout_root=checkout,
        )

    unowned_environment_module = environment / "lib/python3.10/site-packages/unowned.py"
    unowned_environment_module.write_text("VALUE = 1\n", encoding="utf-8")
    unowned = ModuleType("transformers")
    unowned.__spec__ = importlib.util.spec_from_file_location(
        "transformers",
        unowned_environment_module,
    )
    with pytest.raises(RuntimeError, match="not owned"):
        runtime_import_origin(
            "4kagent",
            "transformers",
            unowned,
            environment_root=environment,
            checkout_root=checkout,
            owned_environment_files=frozenset({environment_module}),
        )


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
        {
            "module": f"entrypoint:{relative}",
            "symbols": ["--help"],
            "origin": str((tmp_path / relative).resolve()),
        }
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
