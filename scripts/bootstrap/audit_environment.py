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
import stat
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Iterable, Mapping
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
_INSTALLATION_MERKLE_ALGORITHM = "sha256-merkle-v1"
_EXPECTED_ENVIRONMENTS = {
    "scaleguard": Path(".venv"),
    "4kagent": Path(".runtime/envs/4kagent"),
    "depictqa": Path(".runtime/envs/depictqa"),
    "coz": Path(".runtime/envs/coz"),
}
_CHECKOUT_IMPORT_PREFIXES = {
    "scaleguard": ("scaleguard.",),
    "4kagent": ("llm.", "pipeline."),
    "depictqa": ("model.",),
    "coz": ("osediff_sd3",),
}
_FIXED_VENV_SITE_METADATA = frozenset({"_virtualenv.pth", "_virtualenv.py"})
_FIXED_VENV_BIN_METADATA = frozenset(
    {
        "activate",
        "activate.bat",
        "activate.csh",
        "activate.fish",
        "activate.nu",
        "activate.ps1",
        "activate_this.py",
        "deactivate.bat",
        "pydoc.bat",
    }
)
_VENV_PYTHON_ENTRYPOINT = re.compile(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?")


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


def _lexical_absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def current_python_identity() -> dict[str, str]:
    executable = _lexical_absolute(sys.executable)
    return {
        "executable": str(executable),
        "executable_realpath": str(executable.resolve(strict=True)),
        "prefix": str(_lexical_absolute(sys.prefix)),
        "base_prefix": str(_lexical_absolute(sys.base_prefix)),
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }


def validate_python_identity(
    name: str,
    project_root: Path,
    identity: Mapping[str, str],
) -> None:
    expected_relative = _EXPECTED_ENVIRONMENTS.get(name)
    if expected_relative is None:
        raise RuntimeError(f"unknown runtime environment: {name}")

    expected_prefix = _lexical_absolute(project_root.resolve() / expected_relative)
    managed_python_root = _lexical_absolute(project_root.resolve() / ".runtime/python")
    prefix = Path(identity["prefix"])
    base_prefix = Path(identity["base_prefix"])
    executable = Path(identity["executable"])
    if prefix != expected_prefix:
        raise RuntimeError(
            f"interpreter prefix is {prefix}, expected the fixed environment {expected_prefix}"
        )
    if prefix == base_prefix or prefix.resolve(strict=True) == base_prefix.resolve(strict=True):
        raise RuntimeError("environment audit must run inside a virtual environment")
    if managed_python_root.is_symlink() or not managed_python_root.is_dir():
        raise RuntimeError(f"managed base Python root is missing or unsafe: {managed_python_root}")
    if (
        not base_prefix.is_relative_to(managed_python_root)
        or base_prefix == managed_python_root
        or base_prefix.is_symlink()
        or not base_prefix.is_dir()
    ):
        raise RuntimeError(
            f"base Python prefix {base_prefix} is outside the managed root {managed_python_root}"
        )
    _reject_symlink_components(managed_python_root, base_prefix)
    if prefix.is_symlink() or not prefix.is_dir():
        raise RuntimeError(f"virtual environment prefix is missing or unsafe: {prefix}")
    configuration = prefix / "pyvenv.cfg"
    if configuration.is_symlink() or not configuration.is_file():
        raise RuntimeError(f"virtual environment marker is missing or unsafe: {configuration}")
    if not executable.is_absolute() or executable.parent not in {
        prefix / "bin",
        prefix / "Scripts",
    }:
        raise RuntimeError(
            f"interpreter entry point {executable} is outside the fixed environment bin directory"
        )
    if not executable.is_file():
        raise RuntimeError(f"interpreter entry point is missing: {executable}")
    if not executable.is_symlink():
        raise RuntimeError(f"interpreter entry point must preserve its venv symlink: {executable}")
    executable_realpath = executable.resolve(strict=True)
    if str(executable_realpath) != identity["executable_realpath"]:
        raise RuntimeError("interpreter real path changed while recording its identity")
    if not executable_realpath.is_relative_to(base_prefix):
        raise RuntimeError(
            f"interpreter target is outside its managed base prefix: {executable_realpath}"
        )
    if not executable_realpath.is_file():
        raise RuntimeError(f"interpreter target is missing: {executable_realpath}")


def _merkle_root(payloads: Iterable[bytes]) -> str:
    nodes = [hashlib.sha256(b"\x00" + payload).digest() for payload in payloads]
    if not nodes:
        return hashlib.sha256(b"").hexdigest()
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [
            hashlib.sha256(b"\x01" + nodes[index] + nodes[index + 1]).digest()
            for index in range(0, len(nodes), 2)
        ]
    return nodes[0].hex()


def _canonical_merkle_payload(value: Mapping[str, str | int]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _reject_symlink_components(root: Path, candidate: Path) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"installed file escapes its environment: {candidate}") from error
    current = root
    if current.is_symlink():
        raise RuntimeError(f"environment root must not be a symbolic link: {current}")
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError(f"installed file traverses a symbolic link: {current}")


def _site_package_roots(
    prefix: Path,
    configured: Iterable[Path] | None,
) -> tuple[Path, ...]:
    if configured is None:
        if prefix == _lexical_absolute(sys.prefix):
            candidates = [
                _lexical_absolute(sysconfig.get_path(name)) for name in ("purelib", "platlib")
            ]
        else:
            candidates = [
                *prefix.glob("lib/python*/site-packages"),
                prefix / "Lib/site-packages",
            ]
    else:
        candidates = [_lexical_absolute(path) for path in configured]
    roots = tuple(sorted({path for path in candidates if path.is_dir()}))
    if not roots:
        raise RuntimeError(f"installation environment has no site-packages root: {prefix}")
    for root in roots:
        if root.is_symlink() or not root.is_relative_to(prefix):
            raise RuntimeError(f"site-packages root is outside the environment: {root}")
        _reject_symlink_components(prefix, root)
    return roots


def _scripts_root(prefix: Path, configured: Path | None) -> Path:
    if configured is not None:
        root = _lexical_absolute(configured)
    elif prefix == _lexical_absolute(sys.prefix):
        root = _lexical_absolute(sysconfig.get_path("scripts"))
    else:
        root = prefix / ("Scripts" if os.name == "nt" else "bin")
    if root.is_symlink() or not root.is_dir() or not root.is_relative_to(prefix):
        raise RuntimeError(f"environment scripts root is missing or unsafe: {root}")
    _reject_symlink_components(prefix, root)
    return root


def _file_merkle_payload(path: Path, *, root: Path, kind: str) -> bytes:
    file_stat = path.stat()
    return _canonical_merkle_payload(
        {
            "kind": kind,
            "path": path.relative_to(root).as_posix(),
            "size_bytes": file_stat.st_size,
            "sha256": sha256(path),
        }
    )


def _scan_installation_boundary(
    *,
    prefix: Path,
    owned_files: set[Path],
    site_roots: tuple[Path, ...],
    scripts_root: Path,
    interpreter_realpath: Path | None,
) -> list[bytes]:
    metadata_payloads: list[bytes] = []
    for root in site_roots:
        for candidate in sorted(root.rglob("*")):
            if candidate.is_symlink():
                raise RuntimeError(f"site-packages boundary contains a symbolic link: {candidate}")
            relative_to_root = candidate.relative_to(root)
            if "__pycache__" in relative_to_root.parts:
                continue
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                raise RuntimeError(
                    f"site-packages boundary contains a non-regular file: {candidate}"
                )
            resolved = candidate.resolve(strict=True)
            if resolved in owned_files:
                continue
            if len(relative_to_root.parts) == 1 and candidate.name in _FIXED_VENV_SITE_METADATA:
                metadata_payloads.append(
                    _file_merkle_payload(candidate, root=prefix, kind="venv-site-metadata")
                )
                continue
            raise RuntimeError(f"unowned site-packages file: {candidate}")

    for candidate in sorted(scripts_root.iterdir()):
        if candidate.is_symlink():
            if (
                interpreter_realpath is not None
                and _VENV_PYTHON_ENTRYPOINT.fullmatch(candidate.name)
                and candidate.resolve(strict=True) == interpreter_realpath
            ):
                metadata_payloads.append(
                    _canonical_merkle_payload(
                        {
                            "kind": "venv-python-symlink",
                            "path": candidate.relative_to(prefix).as_posix(),
                            "target": os.readlink(candidate),
                        }
                    )
                )
                continue
            raise RuntimeError(f"environment scripts boundary has an unsafe symlink: {candidate}")
        if candidate.is_dir():
            raise RuntimeError(
                f"environment scripts boundary contains an unexpected directory: {candidate}"
            )
        if not candidate.is_file():
            raise RuntimeError(
                f"environment scripts boundary contains a non-regular file: {candidate}"
            )
        resolved = candidate.resolve(strict=True)
        if resolved in owned_files:
            continue
        if candidate.name in _FIXED_VENV_BIN_METADATA:
            metadata_payloads.append(
                _file_merkle_payload(candidate, root=prefix, kind="venv-bin-metadata")
            )
            continue
        if candidate.stat().st_mode & 0o111:
            raise RuntimeError(f"unowned executable in the environment scripts root: {candidate}")
    metadata_payloads.sort()
    return metadata_payloads


def _installation_file_inventory(
    distributions: Iterable[Any] | None = None,
    *,
    environment_prefix: Path | None = None,
    site_package_roots: Iterable[Path] | None = None,
    scripts_root: Path | None = None,
    interpreter_realpath: Path | None = None,
) -> tuple[dict[str, Any], frozenset[Path]]:
    prefix = _lexical_absolute(environment_prefix or sys.prefix)
    if prefix.is_symlink() or not prefix.is_dir():
        raise RuntimeError(f"installation environment is missing or unsafe: {prefix}")
    real_prefix = prefix.resolve(strict=True)

    discovered: list[tuple[str, str, Any]] = []
    seen_distributions: set[str] = set()
    for distribution in distributions if distributions is not None else metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            raise RuntimeError("installed distribution has no canonical name")
        name = canonicalize_name(raw_name)
        if name in seen_distributions:
            raise RuntimeError(f"duplicate installed distribution: {name}")
        seen_distributions.add(name)
        discovered.append((name, str(distribution.version), distribution))

    distribution_records: list[dict[str, str | int]] = []
    owned_files: set[Path] = set()
    file_owners: dict[Path, str] = {}
    total_files = 0
    for name, version, distribution in sorted(discovered, key=lambda item: item[:2]):
        package_paths = distribution.files
        if package_paths is None:
            raise RuntimeError(f"installed distribution has no RECORD inventory: {name}")
        file_payloads: list[bytes] = []
        record_paths: list[str] = []
        seen_paths: set[str] = set()
        for package_path in package_paths:
            raw_path = Path(str(package_path))
            if raw_path.is_absolute():
                raise RuntimeError(f"{name} RECORD contains an absolute path: {raw_path}")
            candidate = _lexical_absolute(distribution.locate_file(package_path))
            if not candidate.is_relative_to(prefix):
                raise RuntimeError(f"{name} RECORD path escapes its environment: {raw_path}")
            _reject_symlink_components(prefix, candidate)
            relative = candidate.relative_to(prefix)
            if "__pycache__" in relative.parts:
                continue
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as error:
                raise RuntimeError(f"{name} RECORD file is missing: {candidate}") from error
            if not resolved.is_relative_to(real_prefix):
                raise RuntimeError(
                    f"{name} RECORD file resolves outside its environment: {candidate}"
                )
            if not candidate.is_file():
                raise RuntimeError(f"{name} RECORD entry is not a regular file: {candidate}")
            relative_text = relative.as_posix()
            if relative_text in seen_paths:
                raise RuntimeError(f"{name} RECORD contains a duplicate path: {relative_text}")
            seen_paths.add(relative_text)
            previous_owner = file_owners.get(resolved)
            if previous_owner is not None:
                raise RuntimeError(
                    f"installed file is claimed by multiple distributions: "
                    f"{previous_owner}, {name}: {candidate}"
                )
            file_owners[resolved] = name
            owned_files.add(resolved)
            if candidate.name == "RECORD" and candidate.parent.name.endswith(".dist-info"):
                record_paths.append(relative_text)
            file_stat = candidate.stat()
            file_payloads.append(
                _canonical_merkle_payload(
                    {
                        "distribution": name,
                        "version": version,
                        "path": relative_text,
                        "size_bytes": file_stat.st_size,
                        "sha256": sha256(candidate),
                    }
                )
            )
        if len(record_paths) != 1:
            raise RuntimeError(
                f"installed distribution must expose exactly one RECORD file: {name}"
            )
        file_payloads.sort()
        record = {
            "name": name,
            "version": version,
            "record_path": record_paths[0],
            "file_count": len(file_payloads),
            "merkle_root": _merkle_root(file_payloads),
        }
        distribution_records.append(record)
        total_files += len(file_payloads)

    metadata_payloads = _scan_installation_boundary(
        prefix=prefix,
        owned_files=owned_files,
        site_roots=_site_package_roots(prefix, site_package_roots),
        scripts_root=_scripts_root(prefix, scripts_root),
        interpreter_realpath=interpreter_realpath,
    )
    metadata_record = {
        "file_count": len(metadata_payloads),
        "merkle_root": _merkle_root(metadata_payloads),
    }
    overall_payloads = [
        *(_canonical_merkle_payload(record) for record in distribution_records),
        _canonical_merkle_payload(
            {
                "kind": "venv-metadata",
                "file_count": metadata_record["file_count"],
                "merkle_root": metadata_record["merkle_root"],
            }
        ),
    ]
    inventory = {
        "algorithm": _INSTALLATION_MERKLE_ALGORITHM,
        "environment_root": str(prefix),
        "distribution_count": len(distribution_records),
        "distribution_file_count": total_files,
        "file_count": total_files + len(metadata_payloads),
        "merkle_root": _merkle_root(overall_payloads),
        "distributions": distribution_records,
        "venv_metadata": metadata_record,
    }
    return inventory, frozenset(owned_files)


def installation_file_inventory(
    distributions: Iterable[Any] | None = None,
    *,
    environment_prefix: Path | None = None,
    site_package_roots: Iterable[Path] | None = None,
    scripts_root: Path | None = None,
    interpreter_realpath: Path | None = None,
) -> dict[str, Any]:
    if interpreter_realpath is None and _lexical_absolute(
        environment_prefix or sys.prefix
    ) == _lexical_absolute(sys.prefix):
        interpreter_realpath = Path(sys.executable).resolve(strict=True)
    inventory, _owned_files = _installation_file_inventory(
        distributions,
        environment_prefix=environment_prefix,
        site_package_roots=site_package_roots,
        scripts_root=scripts_root,
        interpreter_realpath=interpreter_realpath,
    )
    return inventory


def _regular_file_identity(path: Path, label: str) -> dict[str, str | int]:
    try:
        file_stat = path.lstat()
    except OSError as error:
        raise RuntimeError(f"{label} is missing: {path}") from error
    if not stat.S_ISREG(file_stat.st_mode):
        raise RuntimeError(f"{label} is not a regular file: {path}")
    return {
        "size_bytes": file_stat.st_size,
        "sha256": sha256(path),
    }


def _managed_alias_identity(
    managed_root: Path,
    candidate: Path,
) -> tuple[list[dict[str, str]], str]:
    root = _lexical_absolute(managed_root)
    path = _lexical_absolute(candidate)
    if root.is_symlink() or not root.is_dir() or not path.is_relative_to(root):
        raise RuntimeError(f"managed Python alias is outside its fixed root: {path}")
    real_root = root.resolve(strict=True)
    aliases: list[dict[str, str]] = []
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            target = os.readlink(current)
            raw_target = Path(target)
            lexical_target = _lexical_absolute(
                raw_target if raw_target.is_absolute() else current.parent / raw_target
            )
            if not lexical_target.is_relative_to(root):
                raise RuntimeError(f"managed Python alias target escapes its root: {current}")
            resolved = current.resolve(strict=True)
            if not resolved.is_relative_to(real_root):
                raise RuntimeError(f"managed Python alias resolves outside its root: {current}")
            aliases.append(
                {
                    "path": current.relative_to(root).as_posix(),
                    "target": target,
                    "resolved": str(resolved),
                }
            )
        else:
            try:
                resolved = current.resolve(strict=True)
            except OSError as error:
                raise RuntimeError(f"managed Python alias path is missing: {current}") from error
            if not resolved.is_relative_to(real_root):
                raise RuntimeError(f"managed Python path resolves outside its root: {current}")
    payloads = [
        _canonical_merkle_payload(
            {
                "kind": "base-python-alias",
                "path": alias["path"],
                "target": alias["target"],
                "resolved": alias["resolved"],
            }
        )
        for alias in aliases
    ]
    return aliases, _merkle_root(payloads)


def _stdlib_identity(base_prefix: Path, stdlib_root: Path) -> dict[str, str | int]:
    root = _lexical_absolute(stdlib_root)
    if root.is_symlink() or not root.is_dir() or not root.is_relative_to(base_prefix):
        raise RuntimeError(f"managed base Python stdlib is missing or unsafe: {root}")
    _reject_symlink_components(base_prefix, root)
    payloads: list[bytes] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            raise RuntimeError(f"managed base Python stdlib contains a symlink: {candidate}")
        relative = candidate.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise RuntimeError(
                f"managed base Python stdlib contains a non-regular file: {candidate}"
            )
        payloads.append(_file_merkle_payload(candidate, root=base_prefix, kind="base-stdlib"))
    if not payloads:
        raise RuntimeError(f"managed base Python stdlib is empty: {root}")
    payloads.sort()
    return {
        "stdlib_root": str(root),
        "stdlib_file_count": len(payloads),
        "stdlib_merkle_root": _merkle_root(payloads),
    }


def runtime_installation_identity(
    project_root: Path,
    python_identity: Mapping[str, str],
    *,
    name: str | None = None,
    base_executable: Path | None = None,
    stdlib_root: Path | None = None,
) -> dict[str, Any]:
    if name is None:
        matching_names = [
            candidate_name
            for candidate_name, relative in _EXPECTED_ENVIRONMENTS.items()
            if _lexical_absolute(project_root.resolve() / relative)
            == Path(python_identity["prefix"])
        ]
        if len(matching_names) != 1:
            raise RuntimeError("Python prefix does not identify one managed runtime environment")
        name = matching_names[0]
    validate_python_identity(
        name,
        project_root,
        python_identity,
    )
    prefix = Path(python_identity["prefix"])
    real_executable = Path(python_identity["executable_realpath"])
    interpreter_file = _regular_file_identity(real_executable, "real Python interpreter")
    configuration = prefix / "pyvenv.cfg"
    configuration_file = _regular_file_identity(configuration, "pyvenv.cfg")

    raw_base_executable = base_executable or Path(
        getattr(sys, "_base_executable", python_identity["executable_realpath"])
    )
    base_entrypoint = _lexical_absolute(raw_base_executable)
    managed_python_root = project_root.resolve() / ".runtime/python"
    executable_aliases, executable_alias_merkle_root = _managed_alias_identity(
        managed_python_root,
        base_entrypoint,
    )
    base_realpath = base_entrypoint.resolve(strict=True)
    base_prefix = Path(python_identity["base_prefix"])
    if not base_realpath.is_relative_to(base_prefix) or base_realpath != real_executable:
        raise RuntimeError(
            f"base Python executable differs from the venv interpreter target: {base_entrypoint}"
        )
    base_executable_file = _regular_file_identity(
        base_realpath,
        "managed base Python interpreter",
    )
    stdlib = _stdlib_identity(
        base_prefix,
        stdlib_root or Path(sysconfig.get_path("stdlib")),
    )
    interpreter = {
        "realpath": str(real_executable),
        **interpreter_file,
        "pyvenv_config_path": str(configuration),
        "pyvenv_config_size_bytes": configuration_file["size_bytes"],
        "pyvenv_config_sha256": configuration_file["sha256"],
    }
    base_runtime = {
        "prefix": str(base_prefix),
        "executable": str(base_entrypoint),
        "executable_realpath": str(base_realpath),
        "executable_size_bytes": base_executable_file["size_bytes"],
        "executable_sha256": base_executable_file["sha256"],
        "executable_alias_count": len(executable_aliases),
        "executable_alias_merkle_root": executable_alias_merkle_root,
        "executable_aliases": executable_aliases,
        **stdlib,
    }
    payloads = [
        _canonical_merkle_payload(
            {
                "kind": "interpreter",
                "realpath": interpreter["realpath"],
                "size_bytes": interpreter["size_bytes"],
                "sha256": interpreter["sha256"],
            }
        ),
        _canonical_merkle_payload(
            {
                "kind": "pyvenv-config",
                "path": interpreter["pyvenv_config_path"],
                "size_bytes": interpreter["pyvenv_config_size_bytes"],
                "sha256": interpreter["pyvenv_config_sha256"],
            }
        ),
        _canonical_merkle_payload(
            {
                "kind": "base-runtime",
                "prefix": base_runtime["prefix"],
                "executable_realpath": base_runtime["executable_realpath"],
                "executable_sha256": base_runtime["executable_sha256"],
                "executable_alias_count": base_runtime["executable_alias_count"],
                "executable_alias_merkle_root": base_runtime["executable_alias_merkle_root"],
                "stdlib_file_count": base_runtime["stdlib_file_count"],
                "stdlib_merkle_root": base_runtime["stdlib_merkle_root"],
            }
        ),
    ]
    return {
        "interpreter": interpreter,
        "base_runtime": base_runtime,
        "file_count": (
            2 + int(base_runtime["executable_alias_count"]) + int(base_runtime["stdlib_file_count"])
        ),
        "merkle_root": _merkle_root(payloads),
    }


def bind_runtime_installation(
    inventory: dict[str, Any],
    runtime_identity: dict[str, Any],
) -> dict[str, Any]:
    bound = {
        **inventory,
        "interpreter": runtime_identity["interpreter"],
        "base_runtime": runtime_identity["base_runtime"],
        "file_count": inventory["file_count"] + runtime_identity["file_count"],
    }
    bound["merkle_root"] = _merkle_root(
        [
            _canonical_merkle_payload(
                {
                    "kind": "venv-installation",
                    "file_count": inventory["file_count"],
                    "merkle_root": inventory["merkle_root"],
                }
            ),
            _canonical_merkle_payload(
                {
                    "kind": "python-runtime",
                    "file_count": runtime_identity["file_count"],
                    "merkle_root": runtime_identity["merkle_root"],
                }
            ),
        ]
    )
    return bound


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


def runtime_import_origin(
    profile: str,
    module_name: str,
    module: ModuleType,
    *,
    environment_root: Path,
    checkout_root: Path,
    owned_environment_files: frozenset[Path] | None = None,
) -> str:
    spec = getattr(module, "__spec__", None)
    raw_origin = getattr(spec, "origin", None) or getattr(module, "__file__", None)
    if not isinstance(raw_origin, str) or raw_origin in {"built-in", "frozen"}:
        raise RuntimeError(f"runtime import has no filesystem origin: {module_name}")
    lexical_origin = _lexical_absolute(raw_origin)
    if lexical_origin.is_symlink():
        raise RuntimeError(f"runtime import origin is a symbolic link: {lexical_origin}")
    try:
        origin = lexical_origin.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"runtime import origin is missing: {lexical_origin}") from error
    if not origin.is_file():
        raise RuntimeError(f"runtime import origin is not a regular file: {origin}")

    checkout_prefixes = _CHECKOUT_IMPORT_PREFIXES.get(profile, ())
    from_checkout = any(
        module_name == prefix.rstrip(".") or module_name.startswith(prefix)
        for prefix in checkout_prefixes
    )
    expected_root = checkout_root.resolve() if from_checkout else environment_root.resolve()
    if not origin.is_relative_to(expected_root):
        raise RuntimeError(
            f"runtime import resolved outside its expected root {expected_root}: {origin}"
        )
    if (
        not from_checkout
        and owned_environment_files is not None
        and origin not in owned_environment_files
    ):
        raise RuntimeError(f"runtime import is not owned by an installed RECORD: {origin}")
    return str(origin)


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
        origin = script.resolve()
        if not origin.is_relative_to(import_root.resolve()):
            issues.append(
                {
                    "requirement": requirement,
                    "issue": f"runtime entrypoint escapes its source root: {origin}",
                }
            )
            continue
        checks.append({"module": requirement, "symbols": ["--help"], "origin": str(origin)})
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
    *,
    owned_environment_files: frozenset[Path] | None = None,
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
    environment_root = Path(sys.prefix).resolve()
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
                    try:
                        origin = runtime_import_origin(
                            name,
                            module_name,
                            module,
                            environment_root=environment_root,
                            checkout_root=import_root,
                            owned_environment_files=owned_environment_files,
                        )
                    except RuntimeError as error:
                        issues.append(
                            {
                                "requirement": module_name,
                                "issue": f"runtime import origin is invalid: {error}",
                            }
                        )
                    else:
                        checks.append(
                            {
                                "module": module_name,
                                "symbols": list(symbols),
                                "origin": origin,
                            }
                        )
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
    project_root = args.project_root.resolve()

    lock_records: list[dict[str, str | int]] = []
    issues: list[dict[str, Any]] = []
    expected_packages: dict[str, str] = {}
    python_identity = current_python_identity()
    try:
        validate_python_identity(args.name, project_root, python_identity)
    except (KeyError, OSError, RuntimeError) as error:
        issues.append(
            {
                "requirement": f"{args.name} virtual environment identity",
                "issue": str(error),
            }
        )
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
    owned_environment_files: frozenset[Path] = frozenset()
    try:
        installation_inventory, owned_environment_files = _installation_file_inventory(
            interpreter_realpath=Path(python_identity["executable_realpath"]),
        )
        installation_files = bind_runtime_installation(
            installation_inventory,
            runtime_installation_identity(
                project_root,
                python_identity,
                name=args.name,
            ),
        )
    except (OSError, RuntimeError, ValueError) as error:
        installation_files = {
            "algorithm": _INSTALLATION_MERKLE_ALGORITHM,
            "environment_root": python_identity["prefix"],
            "distribution_count": 0,
            "distribution_file_count": 0,
            "file_count": 0,
            "merkle_root": _merkle_root(()),
            "distributions": [],
            "venv_metadata": {
                "file_count": 0,
                "merkle_root": _merkle_root(()),
            },
            "interpreter": {},
            "base_runtime": {},
        }
        issues.append(
            {
                "requirement": "installed distribution file inventory",
                "issue": str(error),
            }
        )

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
        project_root,
        owned_environment_files=owned_environment_files,
    )
    issues.extend(import_issues)
    status = "passed" if not issues else "failed"
    if not issues and overrides:
        status = "passed_with_audited_override"

    document: dict[str, Any] = {
        "schema_version": 2,
        "name": args.name,
        "status": status,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": python_identity,
        "locks": lock_records,
        "expected_packages": dict(sorted(expected_packages.items())),
        "packages": {name: package.version for name, package in sorted(packages.items())},
        "installation_files": installation_files,
        "runtime_imports": runtime_imports,
        "audited_overrides": overrides,
        "issues": issues,
    }
    write_atomic(args.output, document)
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
