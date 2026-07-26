#!/usr/bin/env python3
"""Audit one installed runtime against its lock and declared dependencies."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

_AUDITED_4KAGENT_OVERRIDES = (
    {
        "parent": "pyiqa",
        "parent_version": "0.1.13",
        "dependency": "transformers",
        "required": "==4.37.2",
        "installed": "5.5.0",
        "reason": "security-updated 4KAgent Qwen runtime; see ADR 0005",
    },
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
        "reason": "test-only HPS metadata omitted from the inference environment; see ADR 0005",
    },
    {
        "parent": "hpsv2",
        "parent_version": "1.2.0",
        "dependency": "pytest-split",
        "required": "==0.8.0",
        "installed": None,
        "reason": "test-only HPS metadata omitted from the inference environment; see ADR 0005",
    },
)
_OVERRIDE_FIELDS = ("parent", "parent_version", "dependency", "required", "installed")
_4KAGENT_BPE_FILENAME = "bpe_simple_vocab_16e6.txt.gz"
_4KAGENT_BPE_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
_4KAGENT_TORCHVISION_VERSION = "0.25.0"
_4KAGENT_ENTRYPOINTS = (
    "executor/denoising/tools/SwinIR/infer_swinir_4kagent.py",
    "executor/defocus_deblurring/tools/Restormer/infer_restormer_4kagent.py",
    "executor/denoising/tools/MPRNet/infer_mprnet_4kagent.py",
    "executor/dehazing/tools/DehazeFormer/inference.py",
    "executor/jpeg_compression_artifact_removal/tools/FBCNN/infer_fbcnn_4kagent.py",
)
_RUNTIME_IMPORT_SPECS = {
    "scaleguard": (
        ("scaleguard.cli", ("main",)),
        ("scaleguard.provenance", ("validate_runtime_preflight",)),
    ),
    "4kagent": (
        (
            "transformers",
            (
                "AutoProcessor",
                "MllamaForConditionalGeneration",
                "Qwen2_5_VLForConditionalGeneration",
            ),
        ),
        ("outlines.models.transformers_vision", ("transformers_vision",)),
        ("pyiqa.archs.musiq_arch", ("MUSIQ",)),
        ("hpsv2", ("score",)),
        ("llm.qwen_vl", ("PerceptionVLMAgent",)),
        ("pipeline.the4kagent_pipeline", ("The4KAgent",)),
    ),
    "depictqa": (
        ("transformers", ("LlamaTokenizer",)),
        ("peft", ("LoraConfig", "get_peft_model")),
        ("sentence_transformers", ("SentenceTransformer",)),
        ("model.model_llama", ("LlamaForCausalLM",)),
        ("model.depictqa", ("DepictQA",)),
    ),
    "coz": (
        ("transformers", ("AutoProcessor", "Qwen2_5_VLForConditionalGeneration")),
        ("diffusers", ("StableDiffusion3Pipeline",)),
        ("peft", ("PeftModel",)),
        ("osediff_sd3", ("SD3Euler", "OSEDiff_SD3_TEST_TILE")),
    ),
}
_PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^;\\\s]+)")
_EXACT_PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^;\\\s]+)$")


@dataclass(frozen=True)
class InstalledPackage:
    name: str
    version: str
    requirements: tuple[str, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_packages() -> dict[str, InstalledPackage]:
    packages: dict[str, InstalledPackage] = {}
    for distribution in metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = canonicalize_name(raw_name)
        if name in packages:
            raise RuntimeError(f"duplicate installed distribution: {name}")
        packages[name] = InstalledPackage(
            name=name,
            version=distribution.version,
            requirements=tuple(distribution.requires or ()),
        )
    return packages


def pinned_requirements(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _PIN.match(line)
        if match is None:
            continue
        name = canonicalize_name(match.group(1))
        version = match.group(2)
        previous = pins.get(name)
        if previous is not None and previous != version:
            raise RuntimeError(f"conflicting pins for {name} in {path}: {previous} and {version}")
        pins[name] = version
    return pins


def exact_pin(value: str) -> tuple[str, str]:
    match = _EXACT_PIN.fullmatch(value)
    if match is None:
        raise ValueError(f"expected an exact NAME==VERSION pin, found {value!r}")
    return canonicalize_name(match.group(1)), match.group(2)


def install_packaging_compatibility() -> bool:
    try:
        importlib.import_module("pkg_resources")
    except ModuleNotFoundError as error:
        if error.name != "pkg_resources":
            raise
        compatibility = ModuleType("pkg_resources")
        compatibility.packaging = importlib.import_module("packaging")  # type: ignore[attr-defined]
        sys.modules["pkg_resources"] = compatibility
        return True
    return False


def install_torchvision_compatibility() -> bool:
    module_name = "torchvision.transforms.functional_tensor"
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
    else:
        return False

    installed = metadata.version("torchvision").split("+", 1)[0]
    if installed != _4KAGENT_TORCHVISION_VERSION:
        raise RuntimeError(
            f"unsupported torchvision compatibility target: {installed}; "
            f"expected {_4KAGENT_TORCHVISION_VERSION}"
        )
    functional = importlib.import_module("torchvision.transforms.functional")
    rgb_to_grayscale = getattr(functional, "rgb_to_grayscale", None)
    if not callable(rgb_to_grayscale):
        raise RuntimeError("torchvision rgb_to_grayscale API does not match the audited contract")
    compatibility = ModuleType(module_name)
    compatibility.rgb_to_grayscale = rgb_to_grayscale  # type: ignore[attr-defined]
    sys.modules[module_name] = compatibility
    return True


def install_depictqa_inference_package(import_root: Path) -> bool:
    import_root = import_root.resolve()
    model_path = import_root / "model"
    if model_path.is_symlink() or not (model_path / "depictqa.py").is_file():
        raise RuntimeError(f"DepictQA inference package is missing or unsafe: {model_path}")
    model_root = model_path.resolve()
    if not model_root.is_relative_to(import_root):
        raise RuntimeError(f"DepictQA inference package escapes its source root: {model_root}")
    if "model" in sys.modules:
        raise RuntimeError("DepictQA model package was imported before inference isolation")
    package = ModuleType("model")
    package.__package__ = "model"
    package.__path__ = [str(model_root)]  # type: ignore[attr-defined]
    sys.modules["model"] = package
    return True


def audit_4kagent_entrypoints(
    import_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "HOME",
            "LANG",
            "LD_LIBRARY_PATH",
            "LOGNAME",
            "PATH",
            "SHELL",
            "SSL_CERT_DIR",
            "SSL_CERT_FILE",
            "TEMP",
            "TMP",
            "TMPDIR",
            "USER",
        }
        or key.startswith("LC_")
    }
    environment.update(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )

    for relative in _4KAGENT_ENTRYPOINTS:
        script = import_root / relative
        requirement = f"entrypoint:{relative}"
        if script.is_symlink() or not script.is_file():
            issues.append(
                {
                    "requirement": requirement,
                    "issue": f"runtime entrypoint is missing or unsafe: {script}",
                }
            )
            continue
        try:
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=script.parent,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            issues.append(
                {
                    "requirement": requirement,
                    "issue": f"runtime entrypoint failed: {type(error).__name__}: {error}",
                }
            )
            continue
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            issues.append(
                {
                    "requirement": requirement,
                    "issue": (
                        f"runtime entrypoint exited {result.returncode}: "
                        f"{detail[-1] if detail else 'no diagnostic output'}"
                    ),
                }
            )
            continue
        checks.append({"module": requirement, "symbols": ["--help"]})
    return checks, issues


def audit_dependencies(
    packages: dict[str, InstalledPackage],
    *,
    allow_4kagent_runtime_overrides: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    overrides: list[dict[str, Any]] = []
    marker_environment = {key: str(value) for key, value in default_environment().items()}
    marker_environment["extra"] = ""

    for package_name in sorted(packages):
        package = packages[package_name]
        for raw_requirement in package.requirements:
            try:
                requirement = Requirement(raw_requirement)
            except InvalidRequirement as error:
                issues.append(
                    {
                        "parent": package.name,
                        "requirement": raw_requirement,
                        "issue": f"invalid requirement metadata: {error}",
                    }
                )
                continue
            if requirement.marker is not None and not requirement.marker.evaluate(
                marker_environment
            ):
                continue
            dependency_name = canonicalize_name(requirement.name)
            dependency = packages.get(dependency_name)
            if dependency is not None and (
                not requirement.specifier or dependency.version in requirement.specifier
            ):
                continue

            mismatch = {
                "parent": package.name,
                "parent_version": package.version,
                "dependency": dependency_name,
                "required": str(requirement.specifier),
                "installed": dependency.version if dependency is not None else None,
            }
            expected_override = next(
                (
                    expected
                    for expected in _AUDITED_4KAGENT_OVERRIDES
                    if all(mismatch[key] == expected[key] for key in _OVERRIDE_FIELDS)
                ),
                None,
            )
            if allow_4kagent_runtime_overrides and expected_override is not None:
                overrides.append({**mismatch, "reason": expected_override["reason"]})
            else:
                issues.append(
                    {
                        **mismatch,
                        "issue": (
                            f"missing distribution: {dependency_name}"
                            if dependency is None
                            else "installed version does not satisfy requirement"
                        ),
                    }
                )

    if allow_4kagent_runtime_overrides:
        for expected in _AUDITED_4KAGENT_OVERRIDES:
            matches = [
                override
                for override in overrides
                if all(override[key] == expected[key] for key in _OVERRIDE_FIELDS)
            ]
            if len(matches) != 1:
                issues.append(
                    {
                        "parent": expected["parent"],
                        "requirement": expected["required"],
                        "dependency": expected["dependency"],
                        "issue": (
                            "an audited 4KAgent runtime override was requested but was "
                            "not observed exactly once"
                        ),
                    }
                )
    return issues, overrides


def audit_runtime_imports(
    name: str,
    project_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = _RUNTIME_IMPORT_SPECS.get(name)
    if specs is None:
        return [], [{"requirement": name, "issue": "unknown runtime import profile"}]

    project_root = project_root.resolve()
    if name == "4kagent":
        import_root = project_root / "third_party" / "checkouts" / "4KAgent"
    elif name == "depictqa":
        import_root = project_root / "third_party" / "dependencies" / "DepictQA" / "src"
    elif name == "coz":
        import_root = project_root / "third_party" / "checkouts" / "Chain-of-Zoom"
    else:
        import_root = project_root / "src"
    if import_root.is_symlink() or not import_root.is_dir():
        return [], [{"requirement": str(import_root), "issue": "runtime import root is missing"}]

    checks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    previous_cwd = Path.cwd()
    previous_path = list(sys.path)
    previous_dont_write = sys.dont_write_bytecode
    environment_names = (
        "HF_HUB_DISABLE_TELEMETRY",
        "HF_HUB_OFFLINE",
        "OUTLINES_CACHE_DIR",
        "PYTHONDONTWRITEBYTECODE",
        "TRANSFORMERS_OFFLINE",
    )
    previous_environment = {key: os.environ.get(key) for key in environment_names}
    compatibility_modules: list[str] = []
    original_gzip_open = gzip.open
    bpe_redirected = False
    try:
        with tempfile.TemporaryDirectory(
            prefix=f"scaleguard-{name}-import-",
            dir=project_root / ".runtime",
        ) as cache_directory:
            Path(cache_directory).chmod(0o700)
            os.environ.update(
                {
                    "HF_HUB_DISABLE_TELEMETRY": "1",
                    "HF_HUB_OFFLINE": "1",
                    "OUTLINES_CACHE_DIR": cache_directory,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                }
            )
            sys.dont_write_bytecode = True
            sys.path.insert(0, str(import_root))
            os.chdir(import_root)
            if name in {"4kagent", "depictqa"}:
                if install_packaging_compatibility():
                    compatibility_modules.append("pkg_resources")
            if name == "depictqa" and install_depictqa_inference_package(import_root):
                compatibility_modules.append("model")
            if name == "4kagent":
                if install_torchvision_compatibility():
                    compatibility_modules.append("torchvision.transforms.functional_tensor")
                bpe_target = (
                    import_root / "utils" / "clib_fiqa" / "model" / _4KAGENT_BPE_FILENAME
                ).resolve()
                bpe_source = (
                    import_root
                    / "executor"
                    / "super_resolution"
                    / "tools"
                    / "DiffBIR"
                    / "diffbir"
                    / "model"
                    / "open_clip"
                    / _4KAGENT_BPE_FILENAME
                )
                if (
                    bpe_source.is_symlink()
                    or not bpe_source.is_file()
                    or sha256(bpe_source) != _4KAGENT_BPE_SHA256
                ):
                    raise RuntimeError("the pinned 4KAgent checkout has no audited BPE vocabulary")

                def audited_gzip_open(
                    filename: Any,
                    *open_args: Any,
                    **open_kwargs: Any,
                ) -> Any:
                    nonlocal bpe_redirected
                    try:
                        candidate = Path(os.fspath(filename)).resolve()
                    except (TypeError, ValueError):
                        candidate = None
                    if candidate == bpe_target:
                        bpe_redirected = True
                        filename = bpe_source
                    return original_gzip_open(filename, *open_args, **open_kwargs)

                gzip.open = audited_gzip_open
            for module_name, symbols in specs:
                try:
                    module = importlib.import_module(module_name)
                    missing = [symbol for symbol in symbols if not hasattr(module, symbol)]
                    if missing:
                        raise AttributeError(
                            f"missing audited symbols: {', '.join(sorted(missing))}"
                        )
                except Exception as error:
                    issues.append(
                        {
                            "requirement": module_name,
                            "issue": f"runtime import failed: {type(error).__name__}: {error}",
                        }
                    )
                else:
                    checks.append({"module": module_name, "symbols": list(symbols)})
            if name == "4kagent" and not bpe_redirected:
                issues.append(
                    {
                        "requirement": "pipeline.the4kagent_pipeline",
                        "issue": "runtime import did not consume the audited BPE vocabulary",
                    }
                )
            if name == "4kagent":
                entrypoint_checks, entrypoint_issues = audit_4kagent_entrypoints(import_root)
                checks.extend(entrypoint_checks)
                issues.extend(entrypoint_issues)
    finally:
        gzip.open = original_gzip_open
        for module_name in compatibility_modules:
            sys.modules.pop(module_name, None)
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_dont_write
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return checks, issues


def write_atomic(path: Path, document: dict[str, Any]) -> None:
    expanded = path.expanduser()
    candidate = expanded if expanded.is_absolute() else Path.cwd() / expanded
    output = candidate.parent.resolve() / candidate.name
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise RuntimeError(f"environment audit output is unsafe: {output}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        directory_descriptor = os.open(
            output.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--lock", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-python", default="3.10.18")
    parser.add_argument("--expect", action="append", default=[])
    parser.add_argument("--allow-4kagent-runtime-overrides", action="store_true")
    args = parser.parse_args()

    lock_records: list[dict[str, str | int]] = []
    issues: list[dict[str, str]] = []
    expected_packages: dict[str, str] = {}
    for raw_pin in args.expect:
        try:
            name, version = exact_pin(raw_pin)
        except ValueError as error:
            issues.append({"requirement": raw_pin, "issue": str(error)})
            continue
        expected_packages[name] = version
    for lock in args.lock:
        resolved = lock.resolve()
        if not resolved.is_file():
            issues.append(
                {
                    "requirement": str(resolved),
                    "issue": "lock file is missing",
                }
            )
            continue
        try:
            pins = pinned_requirements(resolved)
        except RuntimeError as error:
            issues.append({"requirement": str(resolved), "issue": str(error)})
            pins = {}
        for name, version in pins.items():
            previous = expected_packages.get(name)
            if previous is not None and previous != version:
                issues.append(
                    {
                        "requirement": name,
                        "issue": (f"conflicting versions across locks: {previous} and {version}"),
                    }
                )
            expected_packages[name] = version
        lock_records.append(
            {
                "path": str(resolved),
                "sha256": sha256(resolved),
                "pinned_packages": len(pins),
            }
        )

    current_python = platform.python_version()
    if current_python != args.expected_python:
        issues.append(
            {
                "requirement": f"Python {args.expected_python}",
                "installed": current_python,
                "issue": "unexpected interpreter version",
            }
        )

    try:
        packages = installed_packages()
    except RuntimeError as error:
        packages = {}
        issues.append({"requirement": "installed distributions", "issue": str(error)})

    for name, expected_version in sorted(expected_packages.items()):
        package = packages.get(name)
        if package is None:
            issues.append(
                {
                    "requirement": f"{name}=={expected_version}",
                    "issue": "locked distribution is missing",
                }
            )
        elif package.version != expected_version:
            issues.append(
                {
                    "requirement": f"{name}=={expected_version}",
                    "installed": package.version,
                    "issue": "installed version differs from lock",
                }
            )

    dependency_issues, overrides = audit_dependencies(
        packages,
        allow_4kagent_runtime_overrides=args.allow_4kagent_runtime_overrides,
    )
    issues.extend(dependency_issues)
    runtime_imports, import_issues = audit_runtime_imports(
        args.name,
        args.project_root,
    )
    issues.extend(import_issues)
    status = "passed" if not issues else "failed"
    if not issues and overrides:
        status = "passed_with_audited_override"

    document: dict[str, Any] = {
        "schema_version": 1,
        "name": args.name,
        "status": status,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": current_python,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "locks": lock_records,
        "expected_packages": dict(sorted(expected_packages.items())),
        "packages": {name: package.version for name, package in sorted(packages.items())},
        "runtime_imports": runtime_imports,
        "audited_overrides": overrides,
        "issues": issues,
    }
    write_atomic(args.output, document)
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
