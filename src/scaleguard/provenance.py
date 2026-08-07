"""Validation of AutoDL preflight receipts before evidence promotion."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from scaleguard.config import (
    DASHSCOPE_BASE_URL,
    DASHSCOPE_MODEL,
    DASHSCOPE_PROVIDER,
    DASHSCOPE_REGION,
    EXPERIMENT_GROUP_SEMANTICS,
    PipelineConfig,
    parse_config,
    validate_config,
)
from scaleguard.errors import ScaleGuardError
from scaleguard.runtime.process import minimal_subprocess_environment, project_executable
from scaleguard.strict_json import StrictJSONError, loads
from scaleguard.strict_yaml import StrictYAMLError
from scaleguard.strict_yaml import loads as load_strict_yaml
from scaleguard.upstream import verify_upstreams

LOCK_PATHS = (
    "uv.lock",
    "upstream-lock.yaml",
    "runtime-dependencies.yaml",
    "weights-lock.json",
    "environments/4kagent/requirements.lock",
    "environments/4kagent/requirements.resolved.lock",
    "environments/4kagent/pyiqa.override.lock",
    "environments/4kagent/hpsv2.override.lock",
    "environments/depictqa/requirements.lock",
    "environments/depictqa/requirements.resolved.lock",
    "environments/coz/requirements.lock",
    "environments/coz/requirements.resolved.lock",
)
BOOTSTRAP_LOCK_PATHS = (
    "uv.lock",
    "upstream-lock.yaml",
    "runtime-dependencies.yaml",
    "environments/uv.version",
    "environments/python-downloads.json",
    "environments/bootstrap/uv.lock",
    "environments/bootstrap/uv-binary.sha256",
    "environments/4kagent/requirements.lock",
    "environments/4kagent/requirements.resolved.lock",
    "environments/4kagent/pyiqa.override.lock",
    "environments/4kagent/hpsv2.override.lock",
    "environments/depictqa/requirements.lock",
    "environments/depictqa/requirements.resolved.lock",
    "environments/coz/requirements.lock",
    "environments/coz/requirements.resolved.lock",
)
ENVIRONMENT_LOCK_PATHS = {
    "scaleguard": ("uv.lock",),
    "4kagent": (
        "environments/4kagent/requirements.resolved.lock",
        "environments/4kagent/pyiqa.override.lock",
        "environments/4kagent/hpsv2.override.lock",
    ),
    "depictqa": ("environments/depictqa/requirements.resolved.lock",),
    "coz": ("environments/coz/requirements.resolved.lock",),
}
ENVIRONMENT_RUNTIME_IMPORTS = {
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
        (
            "entrypoint:executor/denoising/tools/SwinIR/infer_swinir_4kagent.py",
            ("--help",),
        ),
        (
            ("entrypoint:executor/defocus_deblurring/tools/Restormer/infer_restormer_4kagent.py"),
            ("--help",),
        ),
        (
            "entrypoint:executor/denoising/tools/MPRNet/infer_mprnet_4kagent.py",
            ("--help",),
        ),
        (
            "entrypoint:executor/dehazing/tools/DehazeFormer/inference.py",
            ("--help",),
        ),
        (
            (
                "entrypoint:executor/jpeg_compression_artifact_removal/tools/FBCNN/"
                "infer_fbcnn_4kagent.py"
            ),
            ("--help",),
        ),
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
FOURKAGENT_AUDITED_OVERRIDES: tuple[dict[str, object], ...] = (
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
    {
        "parent": "pyiqa",
        "parent_version": "0.1.13",
        "dependency": "transformers",
        "required": "==4.37.2",
        "installed": "5.5.0",
        "reason": "security-updated 4KAgent Qwen runtime; see ADR 0005",
    },
)
EXPECTED_PYTHON_VERSION = "3.10.18"
EXPECTED_PYTHON_DISTRIBUTION = "cpython-3.10.18-linux-x86_64-gnu"
RUNTIME_PREFLIGHT_MAX_AGE = timedelta(minutes=15)
RUNTIME_PREFLIGHT_CLOCK_SKEW = timedelta(minutes=1)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_GPU_UUID = re.compile(r"GPU-[A-Za-z0-9][A-Za-z0-9-]*")
_WEIGHT_ARTIFACT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_RUNTIME_CHECKOUTS = {
    "fourkagent": ("repositories", "fourkagent", "third_party/checkouts/4KAgent"),
    "chain_of_zoom": (
        "repositories",
        "chain_of_zoom",
        "third_party/checkouts/Chain-of-Zoom",
    ),
    "depictqa": (
        "dependencies",
        "depictqa",
        "third_party/dependencies/DepictQA",
    ),
}
_RUNTIME_WEIGHT_ARTIFACTS = {
    "coz_sd3": (
        "coz-sd3-medium-diffusers",
        "models/stabilityai/stable-diffusion-3-medium-diffusers",
    ),
    "coz_qwen": (
        "coz-qwen2.5-vl-3b-instruct",
        "models/Qwen/Qwen2.5-VL-3B-Instruct",
    ),
    "fourkagent_qwen": (
        "4kagent-qwen2.5-vl-7b-instruct",
        "4kagent/models/Qwen2.5-VL-7B-Instruct",
    ),
    "coz_sr_lora": (
        "coz-sr-lora",
        "chain-of-zoom/ckpt/SR_LoRA/model_20001.pkl",
    ),
    "coz_vae": (
        "coz-vae-adapter",
        "chain-of-zoom/ckpt/SR_VAE/vae_encoder_20001.pt",
    ),
    "coz_vlm_config": (
        "coz-vlm-lora-config",
        "chain-of-zoom/ckpt/VLM_LoRA/checkpoint-10000/adapter_config.json",
    ),
    "coz_vlm_weights": (
        "coz-vlm-lora-weights",
        "chain-of-zoom/ckpt/VLM_LoRA/checkpoint-10000/adapter_model.safetensors",
    ),
    "fourkagent_hps": (
        "4kagent-hpsv2-2.1",
        "4kagent/hpsv2",
    ),
    "depictqa_vicuna": (
        "4kagent-depictqa-vicuna",
        "4kagent/depictqa/vicuna-7b-v1.5",
    ),
    "depictqa_clip": (
        "4kagent-depictqa-clip-vit-l14",
        "4kagent/depictqa/ViT-L-14.pt",
    ),
    "depictqa_degradation_delta": (
        "4kagent-depictqa-degradation-delta",
        "4kagent/depictqa/delta/degra_eval.pt",
    ),
    "quality_musiq": (
        "scaleguard-pyiqa-musiq-koniq",
        "metrics/pyiqa",
    ),
}
_RUNTIME_WEIGHT_LAYOUTS = {
    "fourkagent_toolbox": (
        "4kagent-toolbox-runtime-root",
        "4kagent/runtime/toolbox-root",
    ),
}
_ENVIRONMENT_IDENTITY_FIELDS = (
    "schema_version",
    "name",
    "status",
    "python",
    "locks",
    "expected_packages",
    "packages",
    "installation_files",
    "runtime_imports",
    "audited_overrides",
    "issues",
)
_ENVIRONMENT_RECEIPT_FIELDS = frozenset((*_ENVIRONMENT_IDENTITY_FIELDS, "created_at_utc"))
_PYTHON_IDENTITY_FIELDS = frozenset(
    {
        "executable",
        "executable_realpath",
        "prefix",
        "base_prefix",
        "version",
        "implementation",
        "platform",
    }
)
_INSTALLATION_FIELDS = frozenset(
    {
        "algorithm",
        "environment_root",
        "distribution_count",
        "distribution_file_count",
        "file_count",
        "merkle_root",
        "distributions",
        "venv_metadata",
        "interpreter",
        "base_runtime",
    }
)
_DISTRIBUTION_IDENTITY_FIELDS = frozenset(
    {"name", "version", "record_path", "file_count", "merkle_root"}
)
_INTERPRETER_IDENTITY_FIELDS = frozenset(
    {
        "realpath",
        "size_bytes",
        "sha256",
        "pyvenv_config_path",
        "pyvenv_config_size_bytes",
        "pyvenv_config_sha256",
    }
)
_BASE_RUNTIME_IDENTITY_FIELDS = frozenset(
    {
        "prefix",
        "executable",
        "executable_realpath",
        "executable_size_bytes",
        "executable_sha256",
        "executable_alias_count",
        "executable_alias_merkle_root",
        "executable_aliases",
        "stdlib_root",
        "stdlib_file_count",
        "stdlib_merkle_root",
    }
)


class RuntimePreflightError(ScaleGuardError):
    """Raised when runtime evidence is absent, stale, or inconsistent."""


def _open_regular_evidence(path: Path, label: str) -> tuple[int, os.stat_result]:
    try:
        before = path.lstat()
    except OSError as error:
        raise RuntimePreflightError(f"cannot inspect {label} {path}: {error}") from error
    if not stat.S_ISREG(before.st_mode):
        raise RuntimePreflightError(f"{label} is not a regular file: {path}")

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimePreflightError(f"cannot open {label} {path}: {error}") from error
    try:
        opened = os.fstat(descriptor)
    except OSError as error:
        os.close(descriptor)
        raise RuntimePreflightError(f"cannot inspect opened {label} {path}: {error}") from error
    if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
        before.st_dev,
        before.st_ino,
    ):
        os.close(descriptor)
        raise RuntimePreflightError(f"{label} changed while it was being opened: {path}")
    return descriptor, opened


def _verify_evidence_snapshot(
    path: Path,
    label: str,
    *,
    opened: os.stat_result,
    completed: os.stat_result,
) -> None:
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if not stat.S_ISREG(completed.st_mode) or any(
        getattr(opened, field) != getattr(completed, field) for field in stable_fields
    ):
        raise RuntimePreflightError(f"{label} changed while it was being read: {path}")
    try:
        current = path.lstat()
    except OSError as error:
        raise RuntimePreflightError(f"cannot recheck {label} {path}: {error}") from error
    if not stat.S_ISREG(current.st_mode) or any(
        getattr(opened, field) != getattr(current, field) for field in stable_fields
    ):
        raise RuntimePreflightError(f"{label} changed while it was being read: {path}")


def _snapshot_regular_evidence(
    path: Path,
    label: str,
    *,
    retain_bytes: bool,
) -> tuple[bytes | None, int, str]:
    descriptor, opened = _open_regular_evidence(path, label)
    digest = hashlib.sha256()
    payload = bytearray() if retain_bytes else None
    try:
        with os.fdopen(descriptor, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                if payload is not None:
                    payload.extend(block)
            completed = os.fstat(handle.fileno())
    except OSError as error:
        raise RuntimePreflightError(f"cannot read and hash {label} {path}: {error}") from error
    _verify_evidence_snapshot(
        path,
        label,
        opened=opened,
        completed=completed,
    )
    return (bytes(payload) if payload is not None else None), opened.st_size, digest.hexdigest()


def _read_regular_evidence(path: Path, label: str) -> tuple[bytes, str]:
    payload, _size, digest = _snapshot_regular_evidence(path, label, retain_bytes=True)
    if payload is None:  # pragma: no cover - retain_bytes guarantees a payload
        raise AssertionError("retained evidence snapshot has no payload")
    return payload, digest


def load_regular_file_snapshot(path: Path, label: str) -> tuple[bytes, str]:
    """Read and hash one regular file from the same race-checked snapshot."""

    return _read_regular_evidence(path, label)


def _evidence_sha256(path: Path, label: str) -> str:
    _payload, _size, digest = _snapshot_regular_evidence(path, label, retain_bytes=False)
    return digest


def _inventory_file_identity(path: Path, label: str) -> tuple[int, str]:
    _payload, size, digest = _snapshot_regular_evidence(path, label, retain_bytes=False)
    return size, digest


def sha256(path: Path) -> str:
    """Hash one regular evidence file without following a symbolic link."""

    return _evidence_sha256(path, "evidence file")


def _decode_evidence_object(payload: bytes, path: Path, label: str) -> dict[str, Any]:
    try:
        value = loads(payload)
    except StrictJSONError as error:
        raise RuntimePreflightError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimePreflightError(f"{label} must be a JSON object")
    return value


def _load_snapshot(path: Path, label: str) -> tuple[dict[str, Any], str]:
    if path.is_symlink():
        raise RuntimePreflightError(f"{label} must not be a symbolic link: {path}")
    payload, digest = _read_regular_evidence(path, label)
    return _decode_evidence_object(payload, path, label), digest


def load_evidence_snapshot(path: Path, label: str) -> tuple[dict[str, Any], str]:
    """Load and hash one strict JSON evidence object from the same file snapshot."""

    return _load_snapshot(path, label)


def _evidence_path(value: object, label: str, *, base: Path | None = None) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimePreflightError(f"{label} has no path")
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() or base is None else base / raw
    if candidate.is_symlink():
        raise RuntimePreflightError(f"{label} must not be a symbolic link: {candidate}")
    return candidate.resolve()


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _merkle_payload(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _merkle_root(payloads: list[bytes]) -> str:
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


def _absolute_lexical_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimePreflightError(f"{label} has no path")
    path = Path(value)
    if not path.is_absolute() or str(path) != os.path.abspath(value):
        raise RuntimePreflightError(f"{label} is not a lexical absolute path")
    return path


def _current_regular_identity(
    path: Path,
    *,
    size: object,
    digest: object,
    label: str,
) -> None:
    if type(size) is not int or size < 0 or not _valid_digest(digest):
        raise RuntimePreflightError(f"{label} has an invalid file identity")
    observed_size, observed_digest = _inventory_file_identity(path, label)
    if (observed_size, observed_digest) != (size, digest):
        raise RuntimePreflightError(f"{label} changed after its environment audit")


def _managed_aliases(
    managed_root: Path,
    candidate: Path,
    *,
    label: str,
) -> list[dict[str, str]]:
    if managed_root.is_symlink() or not managed_root.is_dir():
        raise RuntimePreflightError(f"{label} managed Python root is missing or unsafe")
    if not candidate.is_relative_to(managed_root):
        raise RuntimePreflightError(f"{label} executable escapes the managed Python root")
    real_root = managed_root.resolve()
    aliases: list[dict[str, str]] = []
    current = managed_root
    for part in candidate.relative_to(managed_root).parts:
        current /= part
        try:
            if current.is_symlink():
                target = os.readlink(current)
                raw_target = Path(target)
                lexical_target = Path(
                    os.path.abspath(
                        os.fspath(
                            raw_target if raw_target.is_absolute() else current.parent / raw_target
                        )
                    )
                )
                if not lexical_target.is_relative_to(managed_root):
                    raise RuntimePreflightError(
                        f"{label} executable alias target escapes the managed Python root"
                    )
                resolved = current.resolve(strict=True)
                if not resolved.is_relative_to(real_root):
                    raise RuntimePreflightError(
                        f"{label} executable alias resolves outside the managed Python root"
                    )
                aliases.append(
                    {
                        "path": current.relative_to(managed_root).as_posix(),
                        "target": target,
                        "resolved": str(resolved),
                    }
                )
            else:
                resolved = current.resolve(strict=True)
                if not resolved.is_relative_to(real_root):
                    raise RuntimePreflightError(
                        f"{label} executable path resolves outside the managed Python root"
                    )
        except OSError as error:
            raise RuntimePreflightError(
                f"{label} executable alias cannot be inspected: {current}: {error}"
            ) from error
    return aliases


def _installation_merkle_roots(
    distributions: list[dict[str, Any]],
    venv_metadata: dict[str, Any],
    interpreter: dict[str, Any],
    base_runtime: dict[str, Any],
) -> tuple[str, int, str, int]:
    inventory_file_count = sum(record["file_count"] for record in distributions) + int(
        venv_metadata["file_count"]
    )
    inventory_root = _merkle_root(
        [
            *(_merkle_payload(record) for record in distributions),
            _merkle_payload(
                {
                    "kind": "venv-metadata",
                    "file_count": venv_metadata["file_count"],
                    "merkle_root": venv_metadata["merkle_root"],
                }
            ),
        ]
    )
    runtime_file_count = (
        2 + int(base_runtime["executable_alias_count"]) + int(base_runtime["stdlib_file_count"])
    )
    runtime_root = _merkle_root(
        [
            _merkle_payload(
                {
                    "kind": "interpreter",
                    "realpath": interpreter["realpath"],
                    "size_bytes": interpreter["size_bytes"],
                    "sha256": interpreter["sha256"],
                }
            ),
            _merkle_payload(
                {
                    "kind": "pyvenv-config",
                    "path": interpreter["pyvenv_config_path"],
                    "size_bytes": interpreter["pyvenv_config_size_bytes"],
                    "sha256": interpreter["pyvenv_config_sha256"],
                }
            ),
            _merkle_payload(
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
    )
    return inventory_root, inventory_file_count, runtime_root, runtime_file_count


def _timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise RuntimePreflightError(f"{label} is not an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as error:
        raise RuntimePreflightError(f"{label} is not an ISO-8601 timestamp") from error
    if parsed.utcoffset() != timedelta(0):
        raise RuntimePreflightError(f"{label} must use UTC")
    return parsed


def require_clean_git_commit(project_root: Path) -> str:
    """Return the immutable HEAD only when tracked and untracked state is clean."""

    project_root = project_root.resolve()
    head = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        env=minimal_subprocess_environment(),
    )
    commit = head.stdout.strip()
    if head.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimePreflightError("runtime promotion requires a committed Git HEAD")
    status = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=minimal_subprocess_environment(),
    )
    if status.returncode != 0:
        raise RuntimePreflightError("cannot inspect the Git worktree for runtime promotion")
    if status.stdout:
        raise RuntimePreflightError("runtime promotion requires a clean Git worktree")
    return commit


def _validate_current_lock_set(
    value: object,
    *,
    expected: tuple[str, ...],
    project_root: Path,
    label: str,
    snapshot_digests: dict[str, str] | None = None,
) -> None:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise RuntimePreflightError(f"{label} has an unexpected lock set")
    for relative in expected:
        path = project_root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimePreflightError(f"{label} lock mismatch: {relative}")
        digest = (
            snapshot_digests[relative]
            if snapshot_digests is not None and relative in snapshot_digests
            else _evidence_sha256(path, f"{label} lock")
        )
        if value.get(relative) != digest:
            raise RuntimePreflightError(f"{label} lock mismatch: {relative}")


def _validate_gpu_preflight(
    value: object,
    *,
    receipt_path: Path,
    commit: str,
    stage_started_at: datetime,
    preflight_created_at: datetime,
) -> tuple[str, dict[str, object]]:
    """Validate and normalize the GPU inventory bound into one runtime attempt."""

    if not isinstance(value, dict):
        raise RuntimePreflightError("runtime preflight has no GPU inventory binding")
    gpu_directory = receipt_path.parent / "gpu-preflight"
    if gpu_directory.is_symlink() or not gpu_directory.is_dir():
        raise RuntimePreflightError("GPU preflight directory is missing or unsafe")
    expected_path = gpu_directory / "gpu_check.json"
    recorded_path = _absolute_lexical_path(value.get("path"), "GPU preflight receipt")
    if recorded_path != expected_path:
        raise RuntimePreflightError("GPU preflight receipt is outside this runtime attempt")
    document, digest = _load_snapshot(recorded_path, "GPU preflight receipt")
    if not _valid_digest(value.get("sha256")) or value.get("sha256") != digest:
        raise RuntimePreflightError("GPU preflight receipt digest mismatch")
    checked_at = _timestamp(document.get("checked_at_utc"), "GPU preflight checked_at_utc")
    requirements = document.get("requirements")
    selected = document.get("selected_gpus")
    if (
        document.get("schema_version") != 1
        or document.get("status") != "passed"
        or document.get("git_commit") != commit
        or checked_at < stage_started_at
        or checked_at > preflight_created_at
        or not isinstance(requirements, dict)
        or requirements.get("minimum_gpu_count") != 2
        or not isinstance(selected, list)
        or len(selected) != 2
    ):
        raise RuntimePreflightError(
            "GPU preflight is not a current, passed, source-bound dual-GPU inventory"
        )

    normalized: list[dict[str, object]] = []
    for logical_index, item in enumerate(selected):
        if not isinstance(item, dict):
            raise RuntimePreflightError("GPU preflight selected_gpus is malformed")
        physical_index = item.get("physical_index")
        uuid = item.get("uuid")
        name = item.get("name")
        memory_total_mib = item.get("memory_total_mib")
        driver_version = item.get("driver_version")
        if (
            item.get("logical_index") != logical_index
            or physical_index != str(logical_index)
            or not isinstance(uuid, str)
            or _GPU_UUID.fullmatch(uuid) is None
            or not isinstance(name, str)
            or not name
            or type(memory_total_mib) is not int
            or memory_total_mib <= 0
            or not isinstance(driver_version, str)
            or not driver_version
        ):
            raise RuntimePreflightError(
                "GPU preflight does not match the canonical logical and physical 0,1 topology"
            )
        normalized.append(
            {
                "logical_index": logical_index,
                "physical_index": physical_index,
                "uuid": uuid,
                "name": name,
                "memory_total_mib": memory_total_mib,
                "driver_version": driver_version,
            }
        )
    if len({str(item["uuid"]) for item in normalized}) != 2:
        raise RuntimePreflightError("GPU preflight selected a UUID more than once")

    visible = document.get("cuda_visible_devices")
    selectors = [str(item["physical_index"]) for item in normalized]
    if visible is not None:
        if not isinstance(visible, str):
            raise RuntimePreflightError("GPU preflight CUDA selector binding is malformed")
        selectors = visible.split(",")
        if len(selectors) != 2:
            raise RuntimePreflightError("GPU preflight must bind exactly two CUDA selectors")
        for selector, gpu in zip(selectors, normalized, strict=True):
            if selector not in {gpu["physical_index"], gpu["uuid"]}:
                raise RuntimePreflightError(
                    "GPU preflight CUDA selectors do not match selected GPU identities"
                )

    recorded_selected = value.get("selected_gpus")
    if (
        value.get("cuda_visible_devices") != visible
        or not isinstance(recorded_selected, list)
        or recorded_selected != normalized
    ):
        raise RuntimePreflightError("runtime preflight GPU binding differs from its receipt")
    return digest, {
        "schema_version": 1,
        "receipt_path": str(recorded_path),
        "receipt_sha256": digest,
        "cuda_visible_devices": visible,
        "selectors": selectors,
        "selected_gpus": normalized,
    }


def _validate_installation_identity(
    name: str,
    *,
    python: dict[str, Any],
    installation: object,
    packages: dict[str, Any],
    project_root: Path,
    context: str,
) -> None:
    if not isinstance(installation, dict) or set(installation) != _INSTALLATION_FIELDS:
        raise RuntimePreflightError(f"{context} has an invalid installation inventory")
    prefix = _absolute_lexical_path(python["prefix"], f"{context} Python prefix")
    expected_prefix = (
        project_root / ".venv"
        if name == "scaleguard"
        else project_root / ".runtime" / "envs" / name
    )
    expected_prefix = Path(os.path.abspath(os.fspath(expected_prefix)))
    executable = _absolute_lexical_path(
        python["executable"],
        f"{context} Python executable",
    )
    executable_realpath = _absolute_lexical_path(
        python["executable_realpath"],
        f"{context} Python executable realpath",
    )
    base_prefix = _absolute_lexical_path(
        python["base_prefix"],
        f"{context} Python base prefix",
    )
    if (
        prefix != expected_prefix
        or prefix.is_symlink()
        or not prefix.is_dir()
        or executable != prefix / "bin/python"
        or not executable.is_symlink()
        or executable.resolve(strict=True) != executable_realpath
        or executable_realpath.is_symlink()
        or not executable_realpath.is_file()
        or installation.get("algorithm") != "sha256-merkle-v1"
        or installation.get("environment_root") != str(prefix)
    ):
        raise RuntimePreflightError(f"{context} is not bound to its fixed virtual environment")

    distributions = installation.get("distributions")
    if not isinstance(distributions, list) or not distributions:
        raise RuntimePreflightError(f"{context} has no installed distribution inventory")
    distribution_names: list[str] = []
    distribution_file_count = 0
    for index, value in enumerate(distributions):
        label = f"{context} distribution {index}"
        if not isinstance(value, dict) or set(value) != _DISTRIBUTION_IDENTITY_FIELDS:
            raise RuntimePreflightError(f"{label} has an invalid identity")
        distribution_name = value.get("name")
        version = value.get("version")
        record_path = value.get("record_path")
        file_count = value.get("file_count")
        relative_record = PurePosixPath(record_path) if isinstance(record_path, str) else None
        if (
            not isinstance(distribution_name, str)
            or not distribution_name
            or distribution_name in distribution_names
            or not isinstance(version, str)
            or not version
            or packages.get(distribution_name) != version
            or relative_record is None
            or relative_record.is_absolute()
            or ".." in relative_record.parts
            or relative_record.name != "RECORD"
            or type(file_count) is not int
            or file_count <= 0
            or not _valid_digest(value.get("merkle_root"))
        ):
            raise RuntimePreflightError(f"{label} has an invalid identity")
        distribution_names.append(distribution_name)
        distribution_file_count += file_count
    if distribution_names != sorted(distribution_names):
        raise RuntimePreflightError(f"{context} distribution inventory is not canonical")

    venv_metadata = installation.get("venv_metadata")
    if (
        not isinstance(venv_metadata, dict)
        or set(venv_metadata) != {"file_count", "merkle_root"}
        or type(venv_metadata.get("file_count")) is not int
        or venv_metadata["file_count"] <= 0
        or not _valid_digest(venv_metadata.get("merkle_root"))
    ):
        raise RuntimePreflightError(f"{context} has invalid virtual-environment metadata")

    interpreter = installation.get("interpreter")
    if (
        not isinstance(interpreter, dict)
        or set(interpreter) != _INTERPRETER_IDENTITY_FIELDS
        or interpreter.get("realpath") != str(executable_realpath)
    ):
        raise RuntimePreflightError(f"{context} has an invalid interpreter identity")
    _current_regular_identity(
        executable_realpath,
        size=interpreter.get("size_bytes"),
        digest=interpreter.get("sha256"),
        label=f"{context} real Python interpreter",
    )
    pyvenv_config = _absolute_lexical_path(
        interpreter.get("pyvenv_config_path"),
        f"{context} pyvenv.cfg",
    )
    if pyvenv_config != prefix / "pyvenv.cfg":
        raise RuntimePreflightError(f"{context} references an unexpected pyvenv.cfg")
    _current_regular_identity(
        pyvenv_config,
        size=interpreter.get("pyvenv_config_size_bytes"),
        digest=interpreter.get("pyvenv_config_sha256"),
        label=f"{context} pyvenv.cfg",
    )

    base_runtime = installation.get("base_runtime")
    if (
        not isinstance(base_runtime, dict)
        or set(base_runtime) != _BASE_RUNTIME_IDENTITY_FIELDS
        or base_runtime.get("prefix") != str(base_prefix)
        or base_runtime.get("executable_realpath") != str(executable_realpath)
    ):
        raise RuntimePreflightError(f"{context} has an invalid managed base runtime")
    managed_root = project_root / ".runtime/python"
    if (
        base_prefix.is_symlink()
        or not base_prefix.is_dir()
        or not base_prefix.is_relative_to(managed_root)
        or base_prefix == managed_root
        or executable_realpath.resolve(strict=True) != executable_realpath
        or not executable_realpath.is_relative_to(base_prefix)
    ):
        raise RuntimePreflightError(f"{context} base Python escapes its managed root")
    base_executable = _absolute_lexical_path(
        base_runtime.get("executable"),
        f"{context} base Python executable",
    )
    aliases = base_runtime.get("executable_aliases")
    observed_aliases = _managed_aliases(
        managed_root,
        base_executable,
        label=context,
    )
    alias_payloads = [
        _merkle_payload(
            {
                "kind": "base-python-alias",
                "path": alias["path"],
                "target": alias["target"],
                "resolved": alias["resolved"],
            }
        )
        for alias in observed_aliases
    ]
    if (
        aliases != observed_aliases
        or type(base_runtime.get("executable_alias_count")) is not int
        or base_runtime["executable_alias_count"] != len(observed_aliases)
        or base_runtime.get("executable_alias_merkle_root") != _merkle_root(alias_payloads)
        or base_executable.resolve(strict=True) != executable_realpath
    ):
        raise RuntimePreflightError(f"{context} managed Python alias identity changed")
    _current_regular_identity(
        executable_realpath,
        size=base_runtime.get("executable_size_bytes"),
        digest=base_runtime.get("executable_sha256"),
        label=f"{context} managed base Python executable",
    )
    stdlib_root = _absolute_lexical_path(
        base_runtime.get("stdlib_root"),
        f"{context} managed Python stdlib",
    )
    if (
        stdlib_root.is_symlink()
        or not stdlib_root.is_dir()
        or not stdlib_root.is_relative_to(base_prefix)
        or type(base_runtime.get("stdlib_file_count")) is not int
        or base_runtime["stdlib_file_count"] <= 0
        or not _valid_digest(base_runtime.get("stdlib_merkle_root"))
    ):
        raise RuntimePreflightError(f"{context} has an invalid managed Python stdlib identity")

    inventory_root, inventory_count, runtime_root, runtime_count = _installation_merkle_roots(
        distributions,
        venv_metadata,
        interpreter,
        base_runtime,
    )
    expected_root = _merkle_root(
        [
            _merkle_payload(
                {
                    "kind": "venv-installation",
                    "file_count": inventory_count,
                    "merkle_root": inventory_root,
                }
            ),
            _merkle_payload(
                {
                    "kind": "python-runtime",
                    "file_count": runtime_count,
                    "merkle_root": runtime_root,
                }
            ),
        ]
    )
    if (
        installation.get("distribution_count") != len(distributions)
        or installation.get("distribution_file_count") != distribution_file_count
        or inventory_count != distribution_file_count + int(venv_metadata["file_count"])
        or installation.get("file_count") != inventory_count + runtime_count
        or installation.get("merkle_root") != expected_root
    ):
        raise RuntimePreflightError(f"{context} installation Merkle identity is inconsistent")


def _validate_runtime_import_origins(
    name: str,
    value: object,
    *,
    project_root: Path,
    prefix: Path,
    context: str,
) -> None:
    expected = ENVIRONMENT_RUNTIME_IMPORTS[name]
    if not isinstance(value, list) or len(value) != len(expected):
        raise RuntimePreflightError(f"{context} has an unexpected runtime import set")
    checkout_roots = {
        "scaleguard": project_root / "src",
        "4kagent": project_root / "third_party/checkouts/4KAgent",
        "depictqa": project_root / "third_party/dependencies/DepictQA/src",
        "coz": project_root / "third_party/checkouts/Chain-of-Zoom",
    }
    for index, ((module, symbols), record) in enumerate(zip(expected, value, strict=True)):
        label = f"{context} runtime import {index}"
        if (
            not isinstance(record, dict)
            or set(record) != {"module", "symbols", "origin"}
            or record.get("module") != module
            or record.get("symbols") != list(symbols)
        ):
            raise RuntimePreflightError(f"{label} has an invalid contract")
        origin = _absolute_lexical_path(record.get("origin"), f"{label} origin")
        try:
            resolved = origin.resolve(strict=True)
        except OSError as error:
            raise RuntimePreflightError(f"{label} origin is missing: {origin}") from error
        if origin.is_symlink() or origin != resolved or not origin.is_file():
            raise RuntimePreflightError(f"{label} origin is unsafe")
        from_checkout = (
            name == "scaleguard"
            or (name == "4kagent" and module.startswith(("llm.", "pipeline.", "entrypoint:")))
            or (name == "depictqa" and module.startswith("model."))
            or (name == "coz" and module == "osediff_sd3")
        )
        expected_root = checkout_roots[name].resolve() if from_checkout else prefix.resolve()
        if not origin.is_relative_to(expected_root):
            raise RuntimePreflightError(f"{label} origin escapes its fixed source boundary")
        if module.startswith("entrypoint:"):
            expected_origin = expected_root / module.removeprefix("entrypoint:")
            if origin != expected_origin:
                raise RuntimePreflightError(f"{label} origin disagrees with its entrypoint")


def validate_environment_receipt(
    name: str,
    record: object,
    *,
    project_root: Path,
    expected_path: Path,
    context: str,
) -> tuple[Path, dict[str, Any], str]:
    """Validate one schema-v2 environment receipt against current managed bytes."""
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "status"}:
        raise RuntimePreflightError(f"{context} has an invalid evidence record")
    receipt_path = _evidence_path(record["path"], f"{context} receipt", base=project_root)
    if expected_path.is_symlink() or receipt_path != expected_path.resolve():
        raise RuntimePreflightError(f"{context} references an unexpected receipt")
    status = record.get("status")
    expected_status = "passed_with_audited_override" if name == "4kagent" else "passed"
    if status != expected_status:
        raise RuntimePreflightError(f"{context} has an invalid status")
    receipt, digest = _load_snapshot(receipt_path, f"{context} receipt")
    if not _valid_digest(record.get("sha256")) or record["sha256"] != digest:
        raise RuntimePreflightError(f"{context} receipt hash mismatch")

    if (
        set(receipt) != _ENVIRONMENT_RECEIPT_FIELDS
        or receipt.get("schema_version") != 2
        or receipt.get("name") != name
        or receipt.get("status") != status
        or receipt.get("issues") != []
    ):
        raise RuntimePreflightError(f"{context} receipt content mismatch")
    _timestamp(receipt.get("created_at_utc"), f"{context} created_at_utc")

    python = receipt.get("python")
    if (
        not isinstance(python, dict)
        or set(python) != _PYTHON_IDENTITY_FIELDS
        or python.get("version") != EXPECTED_PYTHON_VERSION
        or python.get("implementation") != "CPython"
        or not isinstance(python.get("platform"), str)
        or not python["platform"]
    ):
        raise RuntimePreflightError(f"{context} has an invalid Python identity")

    lock_records = receipt.get("locks")
    expected_locks = ENVIRONMENT_LOCK_PATHS[name]
    if not isinstance(lock_records, list) or len(lock_records) != len(expected_locks):
        raise RuntimePreflightError(f"{context} has an unexpected lock inventory")
    observed_locks: set[str] = set()
    for lock_record in lock_records:
        if not isinstance(lock_record, dict):
            raise RuntimePreflightError(f"{context} has a malformed lock record")
        lock_path = _evidence_path(lock_record.get("path"), f"{context} lock")
        try:
            relative = lock_path.relative_to(project_root).as_posix()
        except ValueError as error:
            raise RuntimePreflightError(f"{context} lock escapes the project") from error
        pinned = lock_record.get("pinned_packages")
        if (
            relative not in expected_locks
            or relative in observed_locks
            or lock_path.is_symlink()
            or not lock_path.is_file()
            or not _valid_digest(lock_record.get("sha256"))
            or lock_record["sha256"] != _evidence_sha256(lock_path, f"{context} lock")
            or type(pinned) is not int
            or pinned < 0
        ):
            raise RuntimePreflightError(f"{context} lock evidence mismatch: {relative}")
        observed_locks.add(relative)
    if observed_locks != set(expected_locks):
        raise RuntimePreflightError(f"{context} has an unexpected lock inventory")

    expected_packages = receipt.get("expected_packages")
    packages = receipt.get("packages")
    runtime_imports = receipt.get("runtime_imports")
    overrides = receipt.get("audited_overrides")
    expected_overrides = list(FOURKAGENT_AUDITED_OVERRIDES) if name == "4kagent" else []
    if (
        not isinstance(expected_packages, dict)
        or not expected_packages
        or not isinstance(packages, dict)
        or not packages
        or not all(
            isinstance(package, str) and package and isinstance(version, str) and version
            for package, version in packages.items()
        )
        or not all(
            isinstance(package, str)
            and package
            and isinstance(version, str)
            and version
            and packages.get(package) == version
            for package, version in expected_packages.items()
        )
        or overrides != expected_overrides
    ):
        raise RuntimePreflightError(f"{context} package audit is inconsistent")
    _validate_installation_identity(
        name,
        python=python,
        installation=receipt.get("installation_files"),
        packages=packages,
        project_root=project_root,
        context=context,
    )
    _validate_runtime_import_origins(
        name,
        runtime_imports,
        project_root=project_root,
        prefix=Path(python["prefix"]),
        context=context,
    )
    return receipt_path, receipt, digest


def _validate_bootstrap(
    record: object,
    *,
    project_root: Path,
    commit: str,
) -> tuple[Path, str, dict[str, dict[str, Any]]]:
    if not isinstance(record, dict):
        raise RuntimePreflightError("runtime preflight receipt has no bootstrap binding")
    path = _evidence_path(record.get("path"), "bootstrap receipt")
    expected_path = (project_root / ".runtime" / "receipts" / "bootstrap.json").resolve()
    if path != expected_path:
        raise RuntimePreflightError("runtime preflight references an unexpected bootstrap receipt")
    bootstrap, digest = _load_snapshot(path, "bootstrap receipt")
    if not _valid_digest(record.get("sha256")) or record["sha256"] != digest:
        raise RuntimePreflightError("bootstrap receipt is stale or did not pass")
    if (
        bootstrap.get("schema_version") != 1
        or bootstrap.get("status") != "passed"
        or bootstrap.get("project_commit") != commit
    ):
        raise RuntimePreflightError("bootstrap receipt is stale or did not pass")
    _timestamp(bootstrap.get("created_at_utc"), "bootstrap created_at_utc")
    if bootstrap.get("python_version") != EXPECTED_PYTHON_VERSION:
        raise RuntimePreflightError("bootstrap receipt has an unexpected Python version")
    uv_path = project_root / "environments" / "uv.version"
    uv_payload, uv_digest = _read_regular_evidence(uv_path, "uv version lock")
    try:
        expected_uv = uv_payload.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise RuntimePreflightError(f"cannot read uv version lock {uv_path}: {error}") from error
    if not expected_uv or bootstrap.get("uv_version") != expected_uv:
        raise RuntimePreflightError("bootstrap receipt has an unexpected uv version")
    uv_binary_path = project_root / "environments/bootstrap/uv-binary.sha256"
    uv_binary_payload, uv_binary_lock_digest = _read_regular_evidence(
        uv_binary_path,
        "uv binary identity lock",
    )
    try:
        expected_uv_binary = uv_binary_payload.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise RuntimePreflightError(
            f"cannot read uv binary identity lock {uv_binary_path}: {error}"
        ) from error
    if (
        not _valid_digest(expected_uv_binary)
        or bootstrap.get("uv_binary_sha256") != expected_uv_binary
    ):
        raise RuntimePreflightError("bootstrap receipt has an unexpected uv binary identity")
    python_downloads_path = project_root / "environments/python-downloads.json"
    python_downloads, python_downloads_digest = _load_snapshot(
        python_downloads_path,
        "managed Python distribution lock",
    )
    if set(python_downloads) != {EXPECTED_PYTHON_DISTRIBUTION}:
        raise RuntimePreflightError("managed Python distribution lock has an unexpected build set")
    python_distribution = python_downloads[EXPECTED_PYTHON_DISTRIBUTION]
    if not isinstance(python_distribution, dict):
        raise RuntimePreflightError("managed Python distribution lock is invalid")
    expected_distribution = {
        "key": EXPECTED_PYTHON_DISTRIBUTION,
        "build": python_distribution.get("build"),
        "url": python_distribution.get("url"),
        "archive_sha256": python_distribution.get("sha256"),
    }
    if (
        python_distribution.get("name") != "cpython"
        or python_distribution.get("major") != 3
        or python_distribution.get("minor") != 10
        or python_distribution.get("patch") != 18
        or python_distribution.get("os") != "linux"
        or python_distribution.get("arch")
        != {
            "family": "x86_64",
            "variant": None,
        }
        or python_distribution.get("libc") != "gnu"
        or not isinstance(expected_distribution["build"], str)
        or not isinstance(expected_distribution["url"], str)
        or not str(expected_distribution["url"]).startswith(
            "https://github.com/astral-sh/python-build-standalone/releases/download/"
        )
        or not _valid_digest(expected_distribution["archive_sha256"])
        or bootstrap.get("python_distribution") != expected_distribution
    ):
        raise RuntimePreflightError(
            "bootstrap receipt has an unexpected managed Python distribution"
        )
    platform = bootstrap.get("platform")
    glibc = platform.get("glibc") if isinstance(platform, dict) else None
    if (
        not isinstance(platform, dict)
        or set(platform) != {"system", "machine", "glibc"}
        or platform.get("system") != "Linux"
        or platform.get("machine") != "x86_64"
        or not isinstance(glibc, str)
        or re.fullmatch(r"\d+(?:\.\d+)+", glibc) is None
        or tuple(int(part) for part in glibc.split(".")[:2]) < (2, 28)
    ):
        raise RuntimePreflightError("bootstrap receipt has an unexpected platform")
    _validate_current_lock_set(
        bootstrap.get("locks"),
        expected=BOOTSTRAP_LOCK_PATHS,
        project_root=project_root,
        label="bootstrap receipt",
        snapshot_digests={
            "environments/uv.version": uv_digest,
            "environments/bootstrap/uv-binary.sha256": uv_binary_lock_digest,
            "environments/python-downloads.json": python_downloads_digest,
        },
    )
    environments = bootstrap.get("environments")
    if not isinstance(environments, dict) or set(environments) != set(ENVIRONMENT_LOCK_PATHS):
        raise RuntimePreflightError("bootstrap receipt has an unexpected environment set")
    environment_receipts: dict[str, dict[str, Any]] = {}
    for name in ENVIRONMENT_LOCK_PATHS:
        _path, environment_receipts[name], _digest = validate_environment_receipt(
            name,
            environments[name],
            project_root=project_root,
            expected_path=project_root / ".runtime" / "receipts" / f"{name}.json",
            context=f"bootstrap environment {name}",
        )
    return path, digest, environment_receipts


def _validate_runtime_environments(
    value: object,
    *,
    receipt_path: Path,
    project_root: Path,
    baseline: dict[str, dict[str, Any]],
    stage_started_at: datetime,
    preflight_created_at: datetime,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    if not isinstance(value, dict) or set(value) != set(ENVIRONMENT_LOCK_PATHS):
        raise RuntimePreflightError(
            "runtime preflight receipt has an unexpected runtime environment set"
        )
    environment_root = receipt_path.parent / "runtime-environments"
    if environment_root.is_symlink() or not environment_root.is_dir():
        raise RuntimePreflightError(
            f"runtime environment receipt directory is missing or unsafe: {environment_root}"
        )
    expected_files = {f"{name}.json" for name in ENVIRONMENT_LOCK_PATHS}
    observed_files = {path.name for path in environment_root.iterdir()}
    if observed_files != expected_files:
        raise RuntimePreflightError(
            "runtime environment receipt directory has an unexpected file set"
        )

    digests: dict[str, str] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for name in ENVIRONMENT_LOCK_PATHS:
        context = f"runtime environment {name}"
        _path, current, digest = validate_environment_receipt(
            name,
            value[name],
            project_root=project_root,
            expected_path=environment_root / f"{name}.json",
            context=context,
        )
        created_at = _timestamp(current.get("created_at_utc"), f"{context} created_at_utc")
        if created_at < stage_started_at or created_at > preflight_created_at:
            raise RuntimePreflightError(
                f"{context} receipt was not created during this preflight stage"
            )
        baseline_receipt = baseline[name]
        for field in _ENVIRONMENT_IDENTITY_FIELDS:
            if current.get(field) != baseline_receipt.get(field):
                raise RuntimePreflightError(
                    f"{context} {field} differs from the bootstrap baseline"
                )
        digests[name] = digest
        receipts[name] = current
    return digests, receipts


def _reaudit_runtime_environments(
    expected: dict[str, dict[str, Any]],
    *,
    project_root: Path,
    receipt_parent: Path,
) -> None:
    audit_script = project_root / "scripts/bootstrap/audit_environment.py"
    if audit_script.is_symlink() or not audit_script.is_file():
        raise RuntimePreflightError("runtime environment auditor is missing or unsafe")
    environment = minimal_subprocess_environment(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    profiles: dict[str, tuple[Path, tuple[str, ...]]] = {
        "scaleguard": (
            project_root / ".venv/bin/python",
            (
                "--lock",
                str(project_root / "uv.lock"),
                "--expect",
                "scaleguard-4k==0.1.0.dev0",
                "--expect",
                "pyiqa==0.1.16",
            ),
        ),
        "4kagent": (
            project_root / ".runtime/envs/4kagent/bin/python",
            (
                "--lock",
                str(project_root / "environments/4kagent/requirements.resolved.lock"),
                "--lock",
                str(project_root / "environments/4kagent/pyiqa.override.lock"),
                "--lock",
                str(project_root / "environments/4kagent/hpsv2.override.lock"),
                "--allow-4kagent-runtime-overrides",
            ),
        ),
        "depictqa": (
            project_root / ".runtime/envs/depictqa/bin/python",
            (
                "--lock",
                str(project_root / "environments/depictqa/requirements.resolved.lock"),
            ),
        ),
        "coz": (
            project_root / ".runtime/envs/coz/bin/python",
            (
                "--lock",
                str(project_root / "environments/coz/requirements.resolved.lock"),
            ),
        ),
    }
    with tempfile.TemporaryDirectory(
        prefix=".runtime-environment-reaudit-",
        dir=receipt_parent,
    ) as temporary:
        output_root = Path(temporary)
        output_root.chmod(0o700)
        for name, (python, profile_args) in profiles.items():
            output = output_root / f"{name}.json"
            command = (
                str(python),
                "-I",
                str(audit_script),
                "--name",
                name,
                "--project-root",
                str(project_root),
                *profile_args,
                "--output",
                str(output),
                "--expected-python",
                EXPECTED_PYTHON_VERSION,
            )
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    cwd=project_root,
                    env=environment,
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise RuntimePreflightError(
                    f"cannot re-audit runtime environment {name}: {error}"
                ) from error
            if result.returncode != 0:
                detail = result.stderr.strip().splitlines()
                suffix = f": {detail[-1][:500]}" if detail else ""
                raise RuntimePreflightError(
                    f"runtime environment {name} failed independent re-audit{suffix}"
                )
            status = "passed_with_audited_override" if name == "4kagent" else "passed"
            _path, observed, _digest = validate_environment_receipt(
                name,
                {
                    "path": str(output),
                    "sha256": _evidence_sha256(output, f"{name} re-audit receipt"),
                    "status": status,
                },
                project_root=project_root,
                expected_path=output,
                context=f"independent runtime environment {name} re-audit",
            )
            mismatched = [
                field
                for field in _ENVIRONMENT_IDENTITY_FIELDS
                if observed.get(field) != expected[name].get(field)
            ]
            if mismatched:
                raise RuntimePreflightError(
                    f"runtime environment {name} differs from independent re-audit: "
                    + ", ".join(mismatched)
                )


def _safe_destination(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or value == ".":
        raise RuntimePreflightError(f"{label} has no destination")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimePreflightError(f"{label} has an unsafe destination: {value!r}")
    candidate = root
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise RuntimePreflightError(f"{label} destination contains a symlink: {value!r}")
    destination = candidate.resolve()
    if not destination.is_relative_to(root):
        raise RuntimePreflightError(f"{label} destination escapes the weight root")
    return destination


def _current_inventory(
    destination: Path,
    *,
    label: str,
    ignore_cache_metadata: bool,
) -> list[dict[str, object]]:
    if destination.is_symlink():
        raise RuntimePreflightError(f"{label} destination is a symbolic link")
    if destination.is_file():
        paths = [destination]
        inventory_root = destination.parent
    elif destination.is_dir():
        discovered = sorted(destination.rglob("*"))
        if ignore_cache_metadata:
            discovered = [
                path
                for path in discovered
                if ".cache" not in path.relative_to(destination).parts
                and ".git" not in path.relative_to(destination).parts
            ]
        symlinks = [path for path in discovered if path.is_symlink()]
        if symlinks:
            raise RuntimePreflightError(f"{label} inventory contains a symlink")
        paths = [path for path in discovered if path.is_file()]
        inventory_root = destination
    else:
        raise RuntimePreflightError(f"{label} destination is missing")
    inventory: list[dict[str, object]] = []
    for path in paths:
        size, digest = _inventory_file_identity(path, f"{label} inventory file")
        inventory.append(
            {
                "path": path.relative_to(inventory_root).as_posix(),
                "size_bytes": size,
                "sha256": digest,
            }
        )
    return inventory


def _weights_root(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimePreflightError("materialization receipt has no weight root")
    raw = Path(value).expanduser()
    if not raw.is_absolute() or raw.is_symlink():
        raise RuntimePreflightError("materialization receipt has an unsafe weight root")
    root = raw.resolve()
    if not root.is_dir():
        raise RuntimePreflightError("materialization weight root is missing")
    return root


def _locked_relative_path(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value == "." or "\\" in value:
        raise RuntimePreflightError(f"{context} is not a safe relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise RuntimePreflightError(f"{context} is not a safe relative path")
    return value


def _locked_file_inventory(
    artifact: dict[str, Any],
    context: str,
) -> tuple[list[str], dict[str, str]]:
    files = artifact.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimePreflightError(f"{context} has no declared files")
    paths: list[str] = []
    hashes: dict[str, str] = {}
    for index, item in enumerate(files):
        item_context = f"{context} file {index}"
        if isinstance(item, str):
            path = _locked_relative_path(item, item_context)
            digest: object = None
        elif isinstance(item, dict):
            path = _locked_relative_path(item.get("path"), item_context)
            digest = item.get("sha256")
            if digest is not None and not _valid_digest(digest):
                raise RuntimePreflightError(f"{item_context} has an invalid SHA-256")
        else:
            raise RuntimePreflightError(f"{item_context} is malformed")
        if path in paths:
            raise RuntimePreflightError(f"{context} declares duplicate file paths")
        paths.append(path)
        if isinstance(digest, str):
            hashes[path] = digest
    return paths, hashes


def _locked_expected_hashes(
    artifact: dict[str, Any],
    context: str,
) -> dict[str, str]:
    file_paths, expected = _locked_file_inventory(artifact, context)
    if "known_sha256" not in artifact:
        raise RuntimePreflightError(f"{context} has no known_sha256 declaration")
    known = artifact["known_sha256"]
    if isinstance(known, str):
        if not _valid_digest(known) or len(file_paths) != 1:
            raise RuntimePreflightError(f"{context} has an invalid known_sha256")
        previous = expected.get(file_paths[0])
        if previous is not None and previous != known:
            raise RuntimePreflightError(f"{context} has conflicting locked SHA-256 values")
        expected[file_paths[0]] = known
    elif isinstance(known, dict):
        for raw_path, digest in known.items():
            path = _locked_relative_path(raw_path, f"{context} known_sha256 path")
            if not _valid_digest(digest):
                raise RuntimePreflightError(f"{context} has an invalid known_sha256")
            previous = expected.get(path)
            if previous is not None and previous != digest:
                raise RuntimePreflightError(f"{context} has conflicting locked SHA-256 values")
            expected[path] = digest
    elif known is not None:
        raise RuntimePreflightError(f"{context} has an invalid known_sha256")
    if not set(expected).issubset(file_paths):
        raise RuntimePreflightError(f"{context} hashes files outside its declared inventory")
    return dict(sorted(expected.items()))


def _locked_weight_artifacts(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = document.get("artifacts")
    if document.get("schema_version") != 1 or not isinstance(artifacts, list) or not artifacts:
        raise RuntimePreflightError("weights lock has an invalid artifact inventory")
    records: dict[str, dict[str, Any]] = {}
    destinations: dict[PurePosixPath, str] = {}
    for index, artifact in enumerate(artifacts):
        context = f"weights lock artifact {index}"
        if not isinstance(artifact, dict):
            raise RuntimePreflightError(f"{context} is malformed")
        artifact_id = artifact.get("id")
        provider = artifact.get("provider")
        required = artifact.get("required")
        destination = artifact.get("destination")
        if (
            not isinstance(artifact_id, str)
            or _WEIGHT_ARTIFACT_ID.fullmatch(artifact_id) is None
            or artifact_id in records
            or provider not in {"huggingface", "https", "manual"}
            or type(required) is not bool
        ):
            raise RuntimePreflightError(f"{context} is malformed")
        destination_value = _locked_relative_path(destination, f"{context} destination")
        destination_path = PurePosixPath(destination_value)
        if destination_path in destinations:
            raise RuntimePreflightError(f"{context} duplicates another artifact destination")
        for other_path, other_id in destinations.items():
            if destination_path in other_path.parents or other_path in destination_path.parents:
                raise RuntimePreflightError(
                    f"{context} destination is nested with artifact {other_id!r}"
                )
        if artifact.get("verify_on_download") is not True:
            raise RuntimePreflightError(f"{context} does not require download verification")
        expected_hashes = _locked_expected_hashes(artifact, context)
        if provider == "huggingface" and (
            not isinstance(artifact.get("repo_id"), str)
            or re.fullmatch(r"[^/\s]+/[^/\s]+", artifact["repo_id"]) is None
            or not isinstance(artifact.get("revision"), str)
            or _GIT_COMMIT.fullmatch(artifact["revision"]) is None
        ):
            raise RuntimePreflightError(f"{context} has no immutable Hugging Face identity")
        if provider == "https":
            url = artifact.get("url")
            sha256 = artifact.get("sha256")
            known_sha256 = artifact.get("known_sha256")
            try:
                parsed = urlsplit(url) if isinstance(url, str) else None
            except ValueError:
                parsed = None
            if (
                parsed is None
                or parsed.scheme != "https"
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or not _valid_digest(sha256)
                or known_sha256 != sha256
                or PurePosixPath(destination_value).suffix == ""
                or list(expected_hashes.items())
                != [(PurePosixPath(destination_value).name, sha256)]
            ):
                raise RuntimePreflightError(f"{context} has no immutable HTTPS identity")
        revision = artifact.get("revision")
        if (
            provider != "huggingface"
            and revision is not None
            and (not isinstance(revision, str) or _GIT_COMMIT.fullmatch(revision) is None)
        ):
            raise RuntimePreflightError(f"{context} has an invalid source revision")
        destinations[destination_path] = artifact_id
        records[artifact_id] = artifact
    return records


def _inventory_hashes(
    inventory: object,
    context: str,
) -> dict[str, str]:
    if not isinstance(inventory, list) or not inventory:
        raise RuntimePreflightError(f"{context} has no file inventory")
    hashes: dict[str, str] = {}
    for entry in inventory:
        if not isinstance(entry, dict):
            raise RuntimePreflightError(f"{context} file inventory is malformed")
        path = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if (
            not isinstance(path, str)
            or not path
            or path in hashes
            or type(size) is not int
            or size < 0
            or not _valid_digest(digest)
        ):
            raise RuntimePreflightError(f"{context} file inventory is malformed")
        assert isinstance(digest, str)
        hashes[path] = digest
    return hashes


def _validate_locked_artifact_hashes(
    *,
    locked: dict[str, Any],
    artifact: dict[str, Any],
    recorded_files: object,
    current_files: object,
    context: str,
) -> None:
    expected = _locked_expected_hashes(locked, context)
    verified = artifact.get("known_hashes_verified")
    if verified != sorted(expected) or artifact.get("verify_on_download") is not True:
        raise RuntimePreflightError(f"{context} hash verification disagrees with weights lock")
    if locked.get("provider") == "manual" and artifact.get(
        "upstream_digest_authenticated"
    ) is not bool(expected):
        raise RuntimePreflightError(f"{context} authentication claim disagrees with weights lock")
    for inventory_name, inventory in (
        ("receipt", recorded_files),
        ("current", current_files),
    ):
        observed = _inventory_hashes(inventory, context)
        mismatched = [path for path, digest in expected.items() if observed.get(path) != digest]
        if mismatched:
            raise RuntimePreflightError(
                f"{context} {inventory_name} inventory disagrees with locked SHA-256"
            )


def resolve_materialization_sources(
    materialization_path: Path,
    *,
    artifact_root: Path,
    weight_receipt_override: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve the fixed marker and its digest-bound download receipt."""

    materialization_path = _evidence_path(
        str(materialization_path),
        "materialization verification receipt",
    )
    materialization, _materialization_digest = _load_snapshot(
        materialization_path,
        "materialization verification receipt",
    )
    root = _weights_root(materialization.get("weights_root"))
    marker_path = root / ".scaleguard-materialization.json"
    marker, _marker_digest = _load_snapshot(marker_path, "fixed materialization marker")
    if materialization != marker:
        raise RuntimePreflightError(
            "materialization verification receipt differs from the fixed marker"
        )
    expected = marker.get("source_weights_receipt_sha256")
    if not _valid_digest(expected):
        raise RuntimePreflightError("materialization marker has no source receipt digest")

    if weight_receipt_override is not None:
        candidates = [weight_receipt_override.expanduser()]
    else:
        artifact_root = artifact_root.expanduser().resolve()
        candidates = sorted(
            artifact_root.glob("weight-download/*/weights-receipt.json"),
            reverse=True,
        )
    matches: list[Path] = []
    for candidate in candidates:
        if candidate.is_symlink() or not candidate.is_file():
            continue
        resolved = candidate.resolve()
        payload, digest = _read_regular_evidence(resolved, "source weights receipt candidate")
        if digest == expected:
            _decode_evidence_object(payload, resolved, "source weights receipt candidate")
            matches.append(resolved)
    if not matches:
        raise RuntimePreflightError(
            "no weight download receipt matches the fixed materialization marker"
        )
    return marker_path.resolve(), matches[0]


def _validate_materialization(
    materialization_record: object,
    marker_record: object,
    weights_record: object,
    *,
    project_root: Path,
    commit: str,
    weights_lock: dict[str, Any],
    weights_lock_digest: str,
) -> tuple[Path, str, str, str, dict[str, dict[str, str]]]:
    if (
        not isinstance(materialization_record, dict)
        or not isinstance(marker_record, dict)
        or not isinstance(weights_record, dict)
    ):
        raise RuntimePreflightError("runtime preflight receipt is missing weight evidence")

    materialization_path = _evidence_path(
        materialization_record.get("path"),
        "materialization verification receipt",
    )
    materialization, materialization_digest = _load_snapshot(
        materialization_path,
        "materialization verification receipt",
    )
    if (
        not _valid_digest(materialization_record.get("sha256"))
        or materialization_record["sha256"] != materialization_digest
    ):
        raise RuntimePreflightError("materialization verification receipt is stale or did not pass")
    weights_root = _weights_root(materialization.get("weights_root"))

    marker_path = _evidence_path(marker_record.get("path"), "fixed materialization marker")
    expected_marker = (weights_root / ".scaleguard-materialization.json").resolve()
    if marker_path != expected_marker:
        raise RuntimePreflightError("runtime preflight references an unexpected fixed marker")
    marker, marker_digest = _load_snapshot(marker_path, "fixed materialization marker")
    if not _valid_digest(marker_record.get("sha256")) or marker_record["sha256"] != marker_digest:
        raise RuntimePreflightError("fixed materialization marker is stale")
    if materialization != marker:
        raise RuntimePreflightError(
            "materialization verification receipt differs from the fixed marker"
        )
    if (
        marker.get("schema_version") != 1
        or marker.get("status") != "passed"
        or marker.get("source_git_commit") != commit
        or marker.get("checkout_mutations") is not False
        or marker.get("errors") != []
    ):
        raise RuntimePreflightError("materialization verification receipt did not pass")
    _timestamp(marker.get("completed_at_utc"), "materialization completed_at_utc")

    layouts = marker.get("layouts")
    if not isinstance(layouts, list) or not layouts:
        raise RuntimePreflightError("materialization receipt has no verified layouts")
    layout_ids: set[str] = set()
    layout_sources: set[str] = set()
    layout_destinations: set[str] = set()
    layout_paths: dict[str, str] = {}
    for index, layout in enumerate(layouts):
        context = f"materialized layout {index}"
        if not isinstance(layout, dict):
            raise RuntimePreflightError(f"{context} is malformed")
        layout_id = layout.get("id")
        source_id = layout.get("source_artifact_id")
        destination_value = layout.get("destination")
        if (
            not isinstance(layout_id, str)
            or not layout_id
            or layout_id in layout_ids
            or not isinstance(source_id, str)
            or not source_id
            or not isinstance(destination_value, str)
            or destination_value in layout_destinations
        ):
            raise RuntimePreflightError(f"{context} is malformed")
        destination = _safe_destination(weights_root, destination_value, context)
        recorded_files = layout.get("files")
        current_files = _current_inventory(
            destination,
            label=context,
            ignore_cache_metadata=False,
        )
        if not isinstance(recorded_files, list) or not recorded_files:
            raise RuntimePreflightError(f"{context} has no file inventory")
        if current_files != recorded_files:
            raise RuntimePreflightError(f"{context} no longer matches its inventory")
        layout_ids.add(layout_id)
        layout_sources.add(source_id)
        layout_destinations.add(destination_value)
        layout_paths[layout_id] = str(destination)

    weights_path = _evidence_path(weights_record.get("path"), "source weights receipt")
    weights, weights_digest = _load_snapshot(weights_path, "source weights receipt")
    if (
        not _valid_digest(weights_record.get("sha256"))
        or weights_record["sha256"] != weights_digest
        or marker.get("source_weights_receipt_sha256") != weights_digest
    ):
        raise RuntimePreflightError("source weights receipt is stale or not bound")
    if (
        weights.get("schema_version") != 1
        or weights.get("status") != "passed"
        or weights.get("git_commit") != commit
        or weights.get("manual_gates") != []
    ):
        raise RuntimePreflightError("source weights receipt did not pass")
    _timestamp(weights.get("completed_at_utc"), "source weights completed_at_utc")
    recorded_root = weights.get("weight_root")
    if (
        not isinstance(recorded_root, str)
        or Path(recorded_root).expanduser().resolve() != weights_root
    ):
        raise RuntimePreflightError("source weights receipt is bound to another weight root")
    if (
        weights.get("source_manifest") != "weights-lock.json"
        or weights.get("source_manifest_sha256") != weights_lock_digest
        or type(weights.get("optional_artifacts_requested")) is not bool
    ):
        raise RuntimePreflightError("source weights receipt has an invalid manifest binding")

    artifacts = weights.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimePreflightError("source weights receipt has no artifact records")
    locked_artifacts = _locked_weight_artifacts(weights_lock)
    receipt_artifacts: dict[str, dict[str, Any]] = {}
    for index, artifact in enumerate(artifacts):
        artifact_id = artifact.get("id") if isinstance(artifact, dict) else None
        if (
            not isinstance(artifact, dict)
            or not isinstance(artifact_id, str)
            or not artifact_id
            or artifact_id in receipt_artifacts
        ):
            raise RuntimePreflightError(f"source weight artifact {index} is malformed")
        receipt_artifacts[artifact_id] = artifact
    if set(receipt_artifacts) != set(locked_artifacts):
        raise RuntimePreflightError(
            "source weights receipt artifact IDs disagree with weights lock"
        )

    completed_artifact_ids: set[str] = set()
    artifact_destinations: set[str] = set()
    artifact_paths: dict[str, str] = {}
    for artifact_id, locked in locked_artifacts.items():
        artifact = receipt_artifacts[artifact_id]
        context = f"source weight artifact {artifact_id!r}"
        status = artifact.get("status")
        required = artifact.get("required")
        provider = artifact.get("provider")
        if (
            provider != locked.get("provider")
            or required != locked.get("required")
            or status not in {"downloaded", "recorded_manual", "skipped"}
            or (required and status not in {"downloaded", "recorded_manual"})
            or (provider == "manual" and status == "downloaded")
            or (provider != "manual" and status == "recorded_manual")
        ):
            raise RuntimePreflightError(
                f"{context} identity or completion disagrees with weights lock"
            )
        if status in {"downloaded", "recorded_manual"}:
            destination_value = artifact.get("destination")
            if (
                not isinstance(destination_value, str)
                or destination_value != locked.get("destination")
                or destination_value in artifact_destinations
            ):
                raise RuntimePreflightError(f"{context} identity disagrees with weights lock")
            if provider == "huggingface" and (
                artifact.get("repo_id") != locked.get("repo_id")
                or artifact.get("revision") != locked.get("revision")
            ):
                raise RuntimePreflightError(f"{context} identity disagrees with weights lock")
            if provider == "https" and artifact.get("url") != locked.get("url"):
                raise RuntimePreflightError(f"{context} identity disagrees with weights lock")
            destination = _safe_destination(weights_root, destination_value, context)
            recorded_files = artifact.get("files")
            current_files = _current_inventory(
                destination,
                label=context,
                ignore_cache_metadata=True,
            )
            if not isinstance(recorded_files, list) or not recorded_files:
                raise RuntimePreflightError(f"{context} has no file inventory")
            if current_files != recorded_files:
                raise RuntimePreflightError(f"{context} no longer matches its inventory")
            _validate_locked_artifact_hashes(
                locked=locked,
                artifact=artifact,
                recorded_files=recorded_files,
                current_files=current_files,
                context=context,
            )
            artifact_destinations.add(destination_value)
            completed_artifact_ids.add(artifact_id)
            artifact_paths[artifact_id] = str(destination)
        elif artifact.get("destination") not in {None, locked.get("destination")}:
            raise RuntimePreflightError(f"{context} identity disagrees with weights lock")
    if not layout_sources.issubset(completed_artifact_ids):
        raise RuntimePreflightError("materialized layouts reference incomplete source artifacts")
    return (
        weights_root,
        materialization_digest,
        marker_digest,
        weights_digest,
        {
            "artifacts": dict(sorted(artifact_paths.items())),
            "layouts": dict(sorted(layout_paths.items())),
        },
    )


def _load_yaml_snapshot(path: Path, label: str) -> tuple[dict[str, Any], str]:
    payload, digest = _read_regular_evidence(path, label)
    try:
        document = load_strict_yaml(payload)
    except StrictYAMLError as error:
        raise RuntimePreflightError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(document, dict):
        raise RuntimePreflightError(f"{label} must be a YAML mapping")
    return document, digest


def _locked_checkout(
    document: dict[str, Any],
    *,
    mapping: str,
    key: str,
    expected_relative: str,
    project_root: Path,
) -> Path:
    records = document.get(mapping)
    record = records.get(key) if isinstance(records, dict) else None
    if (
        document.get("schema_version") != 1
        or not isinstance(record, dict)
        or record.get("checkout") != expected_relative
    ):
        raise RuntimePreflightError(f"runtime profile has an invalid {key} checkout lock")
    if mapping == "dependencies" and (
        record.get("role") != "4kagent_transitive_perception_service"
        or record.get("parent") != "fourkagent"
    ):
        raise RuntimePreflightError("DepictQA must remain a 4KAgent transitive dependency")
    relative = Path(expected_relative)
    checkout = project_root / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or checkout.is_symlink()
        or not checkout.is_dir()
    ):
        raise RuntimePreflightError(f"runtime profile checkout is missing or unsafe: {checkout}")
    resolved = checkout.resolve()
    if not resolved.is_relative_to(project_root):
        raise RuntimePreflightError(f"runtime profile checkout escapes the project: {checkout}")
    return resolved


def _require_verified_upstreams(
    *,
    project_root: Path,
    lock_digests: dict[str, str],
) -> None:
    for relative, mapping in (
        ("upstream-lock.yaml", "repositories"),
        ("runtime-dependencies.yaml", "dependencies"),
    ):
        try:
            results = verify_upstreams(project_root / relative, project_root, mapping)
        except ScaleGuardError as error:
            raise RuntimePreflightError(
                f"cannot verify runtime profile checkout lock {relative}: {error}"
            ) from error
        failures = [f"{result.target}.{result.check}" for result in results if not result.ok]
        if not results or failures:
            detail = ", ".join(failures) if failures else "no verification results"
            raise RuntimePreflightError(
                f"runtime profile checkout verification failed for {relative}: {detail}"
            )
        if (
            _evidence_sha256(project_root / relative, "runtime profile lock")
            != lock_digests[relative]
        ):
            raise RuntimePreflightError(
                f"runtime profile lock changed during checkout verification: {relative}"
            )


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _required_runtime_paths(
    *,
    weights_root: Path,
    weights_lock: dict[str, Any],
    materialized_paths: dict[str, dict[str, str]],
) -> dict[str, str]:
    locked = _locked_weight_artifacts(weights_lock)
    artifacts = materialized_paths.get("artifacts")
    layouts = materialized_paths.get("layouts")
    if not isinstance(artifacts, dict) or not isinstance(layouts, dict):
        raise RuntimePreflightError("materialization did not expose runtime path bindings")

    paths: dict[str, str] = {}
    for role, (artifact_id, relative) in _RUNTIME_WEIGHT_ARTIFACTS.items():
        record = locked.get(artifact_id)
        expected = _safe_destination(weights_root, relative, f"runtime asset {role}")
        if (
            record is None
            or record.get("destination") != relative
            or artifacts.get(artifact_id) != str(expected)
        ):
            raise RuntimePreflightError(f"runtime asset {role} is not bound to its locked bytes")
        paths[role] = str(expected)
    for role, (layout_id, relative) in _RUNTIME_WEIGHT_LAYOUTS.items():
        expected = _safe_destination(weights_root, relative, f"runtime layout {role}")
        if layouts.get(layout_id) != str(expected):
            raise RuntimePreflightError(f"runtime layout {role} is not bound to materialization")
        paths[role] = str(expected)

    quality_root = Path(paths["quality_musiq"])
    paths["quality_musiq"] = str(quality_root / "musiq_koniq_ckpt-e95806b9.pth")
    vlm_config = Path(paths.pop("coz_vlm_config"))
    vlm_weights = Path(paths.pop("coz_vlm_weights"))
    if vlm_config.parent != vlm_weights.parent:
        raise RuntimePreflightError("CoZ VLM LoRA files do not share one locked directory")
    paths["coz_vlm_lora"] = str(vlm_config.parent)
    depictqa_paths = (
        Path(paths.pop("depictqa_vicuna")),
        Path(paths.pop("depictqa_clip")),
        Path(paths.pop("depictqa_degradation_delta")),
    )
    depictqa_root = weights_root / "4kagent" / "depictqa"
    if any(not path.is_relative_to(depictqa_root) for path in depictqa_paths):
        raise RuntimePreflightError("DepictQA assets do not share the locked runtime root")
    paths["depictqa_root"] = str(depictqa_root)

    for role, raw_path in paths.items():
        path = Path(raw_path)
        if path.is_symlink() or not path.exists():
            raise RuntimePreflightError(f"runtime asset {role} is missing or unsafe: {path}")
    return dict(sorted(paths.items()))


def _environment_binding(
    receipts: dict[str, dict[str, Any]],
    *,
    project_root: Path,
    require_current_scaleguard: bool,
) -> dict[str, dict[str, Any]]:
    expected_prefixes = {
        "scaleguard": project_root / ".venv",
        "4kagent": project_root / ".runtime" / "envs" / "4kagent",
        "depictqa": project_root / ".runtime" / "envs" / "depictqa",
        "coz": project_root / ".runtime" / "envs" / "coz",
    }
    bindings: dict[str, dict[str, Any]] = {}
    for name, expected_prefix in expected_prefixes.items():
        receipt = receipts.get(name)
        python = receipt.get("python") if isinstance(receipt, dict) else None
        installation = receipt.get("installation_files") if isinstance(receipt, dict) else None
        if not isinstance(python, dict) or not isinstance(installation, dict):
            raise RuntimePreflightError(f"runtime environment {name} has no execution identity")
        interpreter = installation.get("interpreter")
        base_runtime = installation.get("base_runtime")
        if not isinstance(interpreter, dict) or not isinstance(base_runtime, dict):
            raise RuntimePreflightError(f"runtime environment {name} has no Python byte identity")
        expected_prefix = expected_prefix.resolve()
        expected_executable = expected_prefix / "bin" / "python"
        if (
            python.get("executable") != str(expected_executable)
            or python.get("prefix") != str(expected_prefix)
            or python.get("executable_realpath") != str(expected_executable.resolve())
            or installation.get("environment_root") != str(expected_prefix)
            or installation.get("algorithm") != "sha256-merkle-v1"
            or not _valid_digest(installation.get("merkle_root"))
            or type(installation.get("file_count")) is not int
            or installation["file_count"] <= 0
        ):
            raise RuntimePreflightError(
                f"runtime environment {name} is not bound to its fixed virtual environment"
            )
        binding = {
            "executable": python["executable"],
            "executable_realpath": python["executable_realpath"],
            "prefix": python["prefix"],
            "base_prefix": python.get("base_prefix"),
            "installation_merkle_root": installation["merkle_root"],
            "installation_file_count": installation["file_count"],
            "interpreter_sha256": interpreter.get("sha256"),
            "pyvenv_config_sha256": interpreter.get("pyvenv_config_sha256"),
            "base_executable_sha256": base_runtime.get("executable_sha256"),
            "base_executable_alias_merkle_root": base_runtime.get("executable_alias_merkle_root"),
            "base_stdlib_merkle_root": base_runtime.get("stdlib_merkle_root"),
        }
        digest_fields = (
            "interpreter_sha256",
            "pyvenv_config_sha256",
            "base_executable_sha256",
            "base_executable_alias_merkle_root",
            "base_stdlib_merkle_root",
        )
        if (
            not isinstance(binding["base_prefix"], str)
            or not binding["base_prefix"]
            or any(not _valid_digest(binding[field]) for field in digest_fields)
        ):
            raise RuntimePreflightError(f"runtime environment {name} has no base Python identity")
        bindings[name] = binding

    if require_current_scaleguard:
        current_executable = Path(sys.executable)
        current_executable = current_executable.parent.resolve() / current_executable.name
        current_prefix = Path(sys.prefix).resolve()
        expected = bindings["scaleguard"]
        if (
            str(current_executable) != expected["executable"]
            or str(current_prefix) != expected["prefix"]
        ):
            raise RuntimePreflightError(
                "real runtime validation is not executing inside the audited ScaleGuard environment"
            )
    return bindings


def _resolved_config_path(project_root: Path, value: Path | str, label: str) -> Path:
    path = Path(value).expanduser()
    candidate = path if path.is_absolute() else project_root / path
    resolved = candidate.resolve()
    if not resolved.exists():
        raise RuntimePreflightError(f"{label} is missing: {resolved}")
    return resolved


def _binding_mapping(binding: object, key: str) -> dict[str, Any]:
    if not isinstance(binding, dict):
        raise RuntimePreflightError("runtime execution binding is malformed")
    value = binding.get(key)
    if not isinstance(value, dict):
        raise RuntimePreflightError(f"runtime execution binding has no {key}")
    return value


def bind_runtime_config(
    config: PipelineConfig,
    *,
    project_root: Path,
    binding: dict[str, Any],
) -> PipelineConfig:
    """Validate a real profile and replace free paths with its audited absolute paths."""

    project_root = project_root.resolve()
    validate_config(config)
    checkouts = _binding_mapping(binding, "checkouts")
    environments = _binding_mapping(binding, "environments")
    overlays = _binding_mapping(binding, "overlays")
    assets = _binding_mapping(binding, "assets")
    fourkagent_environment = environments.get("4kagent")
    depictqa_environment = environments.get("depictqa")
    coz_environment = environments.get("coz")
    if not all(
        isinstance(value, dict)
        for value in (fourkagent_environment, depictqa_environment, coz_environment)
    ):
        raise RuntimePreflightError("runtime execution binding has incomplete environments")
    assert isinstance(fourkagent_environment, dict)
    assert isinstance(depictqa_environment, dict)
    assert isinstance(coz_environment, dict)

    expected_paths: dict[str, tuple[Path | str | None, str]] = {
        "fourkagent.checkout": (config.fourkagent.checkout, str(checkouts.get("fourkagent", ""))),
        "fourkagent.depictqa_cwd": (
            config.fourkagent.depictqa_cwd,
            str(checkouts.get("depictqa", "")),
        ),
        "fourkagent.perception_model_path": (
            config.fourkagent.perception_model_path,
            str(assets.get("fourkagent_qwen", "")),
        ),
        "fourkagent.toolbox_root": (
            config.fourkagent.toolbox_root,
            str(assets.get("fourkagent_toolbox", "")),
        ),
        "fourkagent.hps_root": (
            config.fourkagent.hps_root,
            str(assets.get("fourkagent_hps", "")),
        ),
        "fourkagent.quality_model_path": (
            config.fourkagent.quality_model_path,
            str(assets.get("quality_musiq", "")),
        ),
        "coz.checkout": (config.coz.checkout, str(checkouts.get("chain_of_zoom", ""))),
        "coz.model_path": (config.coz.model_path, str(assets.get("coz_sd3", ""))),
        "coz.qwen_model_path": (config.coz.qwen_model_path, str(assets.get("coz_qwen", ""))),
        "coz.sr_lora_path": (config.coz.sr_lora_path, str(assets.get("coz_sr_lora", ""))),
        "coz.vae_path": (config.coz.vae_path, str(assets.get("coz_vae", ""))),
        "coz.vlm_lora_path": (
            config.coz.vlm_lora_path,
            str(assets.get("coz_vlm_lora", "")),
        ),
        "metrics.quality_model_path": (
            config.metrics.quality_model_path,
            str(assets.get("quality_musiq", "")),
        ),
    }
    for label, (configured, expected_text) in expected_paths.items():
        if configured is None or not expected_text:
            raise RuntimePreflightError(f"runtime profile binding is incomplete: {label}")
        if _resolved_config_path(project_root, configured, label) != Path(expected_text):
            raise RuntimePreflightError(f"{label} is not bound to the audited runtime")

    expected_fourkagent_python = str(fourkagent_environment.get("executable", ""))
    expected_coz_python = str(coz_environment.get("executable", ""))
    if (
        project_executable(project_root, config.fourkagent.python_executable)
        != expected_fourkagent_python
        or project_executable(project_root, config.coz.python_executable) != expected_coz_python
    ):
        raise RuntimePreflightError(
            "runtime profile interpreter does not match its audited environment"
        )
    semantics = EXPERIMENT_GROUP_SEMANTICS.get(config.runtime.experiment_group or "")
    expected_fourkagent_mode = semantics[0] if semantics is not None else "upstream"
    expected_acceptance_policy = semantics[4] if semantics is not None else "trusted"
    if (
        config.fourkagent.mode != expected_fourkagent_mode
        or config.fourkagent.profile != "FastGen4K_P"
        or config.fourkagent.llm_provider != DASHSCOPE_PROVIDER
        or config.fourkagent.llm_base_url != DASHSCOPE_BASE_URL
        or config.fourkagent.llm_region != DASHSCOPE_REGION
        or config.fourkagent.llm_model != DASHSCOPE_MODEL
        or config.fourkagent.api_key_env != "DASHSCOPE_API_KEY"
        or config.fourkagent.llm_connect_timeout_seconds != 10.0
        or config.fourkagent.llm_read_timeout_seconds != 120.0
        or config.fourkagent.llm_max_transport_retries != 4
        or config.fourkagent.llm_max_structure_retries != 2
        or config.fourkagent.llm_max_completion_tokens != 1024
        or config.fourkagent.llm_temperature != 0.0
        or config.fourkagent.command
        or config.fourkagent.tool_gpu != "0"
        or config.fourkagent.depictqa_visible_devices != "1"
        or config.fourkagent.depictqa_host != "127.0.0.1"
        or config.fourkagent.depictqa_port != 5001
        or config.coz.mode != "persistent"
        or config.coz.command
        or config.coz.visible_devices != "0,1"
        or config.coz.prompt_type != "vlm"
        or config.metrics.quality_backend != "pyiqa"
        or config.metrics.quality_metric != "musiq"
        or config.metrics.quality_device != "cpu"
        or config.controller.acceptance_policy != expected_acceptance_policy
        or config.controller.accept_unvalidated_quality_proxy
    ):
        raise RuntimePreflightError("runtime profile semantics differ from the audited topology")

    service_work_dir = project_root / ".runtime-binding-service-work-dir"
    checkout = Path(str(checkouts["fourkagent"]))
    try:
        from scaleguard.runtime.process import format_command

        expanded_depictqa = format_command(
            config.fourkagent.depictqa_command,
            {
                "project_root": str(project_root),
                "checkout": str(checkout),
                "service_work_dir": str(service_work_dir),
            },
        )
    except ScaleGuardError as error:
        raise RuntimePreflightError(f"invalid managed DepictQA command: {error}") from error
    if len(expanded_depictqa) != 12:
        raise RuntimePreflightError("managed DepictQA command has an unexpected argument set")
    expected_depictqa = (
        str(depictqa_environment.get("executable", "")),
        str(overlays.get("depictqa", "")),
        "--depictqa-checkout",
        str(checkouts.get("depictqa", "")),
        "--app-script",
        str(checkout / "installation" / "custom_depictqa_scripts" / "app_eval.py"),
        "--base-config",
        str(checkout / "installation" / "custom_depictqa_scripts" / "config_eval.yaml"),
        "--weights-root",
        str(assets.get("depictqa_root", "")),
        "--session-dir",
        str(service_work_dir),
    )
    normalized_depictqa = list(expanded_depictqa)
    normalized_depictqa[0] = project_executable(project_root, normalized_depictqa[0])
    for index in (1, 3, 5, 7, 9):
        normalized_depictqa[index] = str(Path(normalized_depictqa[index]).resolve())
    if tuple(normalized_depictqa) != expected_depictqa:
        raise RuntimePreflightError("managed DepictQA command is not bound to the audited runtime")

    normalized_command = (*expected_depictqa[:-1], "{service_work_dir}")
    return replace(
        config,
        fourkagent=replace(
            config.fourkagent,
            checkout=Path(str(checkouts["fourkagent"])),
            python_executable=expected_fourkagent_python,
            depictqa_command=normalized_command,
            depictqa_cwd=Path(str(checkouts["depictqa"])),
            perception_model_path=str(assets["fourkagent_qwen"]),
            toolbox_root=Path(str(assets["fourkagent_toolbox"])),
            hps_root=Path(str(assets["fourkagent_hps"])),
            quality_model_path=Path(str(assets["quality_musiq"])),
        ),
        coz=replace(
            config.coz,
            checkout=Path(str(checkouts["chain_of_zoom"])),
            python_executable=expected_coz_python,
            model_path=str(assets["coz_sd3"]),
            qwen_model_path=str(assets["coz_qwen"]),
            sr_lora_path=Path(str(assets["coz_sr_lora"])),
            vae_path=Path(str(assets["coz_vae"])),
            vlm_lora_path=Path(str(assets["coz_vlm_lora"])),
        ),
        metrics=replace(
            config.metrics,
            quality_model_path=Path(str(assets["quality_musiq"])),
        ),
    )


def _runtime_profile_binding(
    *,
    config_payload: bytes,
    config_path: Path,
    project_root: Path,
    weights_root: Path,
    weights_lock: dict[str, Any],
    upstream_lock: dict[str, Any],
    dependency_lock: dict[str, Any],
    runtime_environments: dict[str, dict[str, Any]],
    materialized_paths: dict[str, dict[str, str]],
    lock_digests: dict[str, str],
    require_current_scaleguard: bool,
) -> tuple[dict[str, Any], PipelineConfig]:
    repositories = upstream_lock.get("repositories")
    dependencies = dependency_lock.get("dependencies")
    if not isinstance(repositories, dict) or set(repositories) != {
        "fourkagent",
        "chain_of_zoom",
    }:
        raise RuntimePreflightError("4KAgent and Chain-of-Zoom must remain the only core upstreams")
    if not isinstance(dependencies, dict) or set(dependencies) != {"depictqa"}:
        raise RuntimePreflightError(
            "DepictQA must remain the only pinned transitive runtime repository"
        )
    checkouts = {
        name: str(
            _locked_checkout(
                upstream_lock if mapping == "repositories" else dependency_lock,
                mapping=mapping,
                key=key,
                expected_relative=relative,
                project_root=project_root,
            )
        )
        for name, (mapping, key, relative) in _RUNTIME_CHECKOUTS.items()
    }
    _require_verified_upstreams(project_root=project_root, lock_digests=lock_digests)
    environments = _environment_binding(
        runtime_environments,
        project_root=project_root,
        require_current_scaleguard=require_current_scaleguard,
    )
    overlays = {
        "fourkagent": str(
            (project_root / "third_party/overlays/4kagent/run_native_restoration.py").resolve()
        ),
        "scheduler": str(
            (project_root / "third_party/overlays/4kagent/scheduler_client.py").resolve()
        ),
        "depictqa": str(
            (project_root / "third_party/overlays/4kagent/serve_depictqa_eval.py").resolve()
        ),
        "chain_of_zoom": str(
            (project_root / "third_party/overlays/chain-of-zoom/coz_session_worker.py").resolve()
        ),
    }
    if any(Path(path).is_symlink() or not Path(path).is_file() for path in overlays.values()):
        raise RuntimePreflightError("runtime profile overlay is missing or unsafe")
    assets = _required_runtime_paths(
        weights_root=weights_root,
        weights_lock=weights_lock,
        materialized_paths=materialized_paths,
    )
    binding: dict[str, Any] = {
        "schema_version": 1,
        "checkouts": dict(sorted(checkouts.items())),
        "environments": dict(sorted(environments.items())),
        "overlays": dict(sorted(overlays.items())),
        "assets": assets,
        "weights_root": str(weights_root),
        "upstreams": {
            "fourkagent": {
                "commit": repositories["fourkagent"].get("commit"),
                "tree": repositories["fourkagent"].get("tree"),
            },
            "chain_of_zoom": {
                "commit": repositories["chain_of_zoom"].get("commit"),
                "tree": repositories["chain_of_zoom"].get("tree"),
            },
            "depictqa": {
                "commit": dependencies["depictqa"].get("commit"),
                "tree": dependencies["depictqa"].get("tree"),
                "role": dependencies["depictqa"].get("role"),
            },
        },
    }
    try:
        parsed = parse_config(config_payload, source=config_path)
    except ScaleGuardError as error:
        raise RuntimePreflightError(f"runtime profile configuration is invalid: {error}") from error
    return binding, bind_runtime_config(parsed, project_root=project_root, binding=binding)


def validate_runtime_preflight(
    receipt_path: Path,
    *,
    config_path: Path | None,
    project_root: Path,
    require_recent: bool = False,
    require_runtime_profile: bool = True,
    reaudit_environments: bool | None = None,
) -> dict[str, Any]:
    """Bind one run to clean source, config, environments, and current weight bytes."""

    project_root = project_root.resolve()
    receipt_path = _evidence_path(str(receipt_path), "runtime preflight receipt")
    receipt, receipt_digest = _load_snapshot(receipt_path, "runtime preflight receipt")
    if receipt.get("schema_version") != 2 or receipt.get("status") != "passed":
        raise RuntimePreflightError("runtime preflight receipt did not pass")
    created_at = _timestamp(
        receipt.get("created_at_utc"),
        "runtime preflight created_at_utc",
    )
    stage_started_at = _timestamp(
        receipt.get("stage_started_at_utc"),
        "runtime preflight stage_started_at_utc",
    )
    if stage_started_at > created_at:
        raise RuntimePreflightError("runtime preflight predates its stage start")
    if created_at - stage_started_at > RUNTIME_PREFLIGHT_MAX_AGE:
        raise RuntimePreflightError("runtime preflight stage exceeded its maximum evidence window")
    if require_recent:
        now = datetime.now(timezone.utc)
        if (
            created_at > now + RUNTIME_PREFLIGHT_CLOCK_SKEW
            or now - created_at > RUNTIME_PREFLIGHT_MAX_AGE
            or stage_started_at > now + RUNTIME_PREFLIGHT_CLOCK_SKEW
            or now - stage_started_at > RUNTIME_PREFLIGHT_MAX_AGE
        ):
            raise RuntimePreflightError(
                "runtime preflight is not recent enough to start a real run"
            )
    commit = require_clean_git_commit(project_root)
    if receipt.get("project_commit") != commit:
        raise RuntimePreflightError("runtime preflight receipt is bound to another commit")
    gpu_preflight_digest, gpu_preflight_binding = _validate_gpu_preflight(
        receipt.get("gpu_preflight"),
        receipt_path=receipt_path,
        commit=commit,
        stage_started_at=stage_started_at,
        preflight_created_at=created_at,
    )

    config = receipt.get("config")
    if not isinstance(config, dict):
        raise RuntimePreflightError("runtime preflight receipt has no config binding")
    recorded_config_path = _evidence_path(config.get("path"), "runtime config")
    config_path = recorded_config_path if config_path is None else config_path.resolve()
    if recorded_config_path != config_path:
        raise RuntimePreflightError("runtime preflight receipt is bound to another config")
    config_payload, config_digest = _read_regular_evidence(config_path, "runtime config")
    if not _valid_digest(config.get("sha256")) or config["sha256"] != config_digest:
        raise RuntimePreflightError("runtime config changed after preflight")

    upstream_lock, upstream_lock_digest = _load_yaml_snapshot(
        project_root / "upstream-lock.yaml",
        "upstream lock",
    )
    dependency_lock, dependency_lock_digest = _load_yaml_snapshot(
        project_root / "runtime-dependencies.yaml",
        "runtime dependencies lock",
    )
    weights_lock, weights_lock_digest = _load_snapshot(
        project_root / "weights-lock.json",
        "weights lock",
    )
    _validate_current_lock_set(
        receipt.get("locks"),
        expected=LOCK_PATHS,
        project_root=project_root,
        label="runtime preflight receipt",
        snapshot_digests={
            "upstream-lock.yaml": upstream_lock_digest,
            "runtime-dependencies.yaml": dependency_lock_digest,
            "weights-lock.json": weights_lock_digest,
        },
    )
    _bootstrap_path, bootstrap_digest, baseline_environments = _validate_bootstrap(
        receipt.get("bootstrap"),
        project_root=project_root,
        commit=commit,
    )
    runtime_environment_digests, runtime_environments = _validate_runtime_environments(
        receipt.get("runtime_environments"),
        receipt_path=receipt_path,
        project_root=project_root,
        baseline=baseline_environments,
        stage_started_at=stage_started_at,
        preflight_created_at=created_at,
    )
    if reaudit_environments is None:
        reaudit_environments = require_recent and require_runtime_profile
    if reaudit_environments:
        _reaudit_runtime_environments(
            runtime_environments,
            project_root=project_root,
            receipt_parent=receipt_path.parent,
        )
    (
        weights_root,
        materialization_digest,
        marker_digest,
        weights_digest,
        materialized_paths,
    ) = _validate_materialization(
        receipt.get("materialization"),
        receipt.get("materialization_marker"),
        receipt.get("source_weights_receipt"),
        project_root=project_root,
        commit=commit,
        weights_lock=weights_lock,
        weights_lock_digest=weights_lock_digest,
    )
    result: dict[str, Any] = {
        "runtime_evidence_verified": True,
        "runtime_preflight_receipt": str(receipt_path),
        "runtime_preflight_sha256": receipt_digest,
        "gpu_preflight_receipt_sha256": gpu_preflight_digest,
        "gpu_preflight_binding": gpu_preflight_binding,
        "bootstrap_receipt_sha256": bootstrap_digest,
        "runtime_environment_receipt_sha256": runtime_environment_digests,
        "materialization_receipt_sha256": materialization_digest,
        "materialization_marker_sha256": marker_digest,
        "source_weights_receipt_sha256": weights_digest,
        "weights_root": str(weights_root),
        "project_commit": commit,
        "project_root": str(project_root),
        "runtime_config_path": str(config_path),
        "runtime_config_sha256": config_digest,
        "runtime_stage_started_at": stage_started_at.isoformat(),
    }
    if require_runtime_profile:
        lock_digests = {
            "upstream-lock.yaml": upstream_lock_digest,
            "runtime-dependencies.yaml": dependency_lock_digest,
        }
        binding, _bound_config = _runtime_profile_binding(
            config_payload=config_payload,
            config_path=config_path,
            project_root=project_root,
            weights_root=weights_root,
            weights_lock=weights_lock,
            upstream_lock=upstream_lock,
            dependency_lock=dependency_lock,
            runtime_environments=runtime_environments,
            materialized_paths=materialized_paths,
            lock_digests=lock_digests,
            require_current_scaleguard=require_recent,
        )
        binding["gpu_preflight"] = gpu_preflight_binding
        result.update(
            {
                "runtime_profile_bound": True,
                "runtime_execution_binding": binding,
                "runtime_execution_binding_sha256": _canonical_sha256(binding),
            }
        )
    return result
