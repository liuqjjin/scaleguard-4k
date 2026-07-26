"""Validation of AutoDL preflight receipts before evidence promotion."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from scaleguard.errors import ScaleGuardError
from scaleguard.runtime.process import minimal_subprocess_environment
from scaleguard.strict_json import StrictJSONError, loads

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
    "environments/bootstrap/uv.lock",
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
RUNTIME_PREFLIGHT_MAX_AGE = timedelta(minutes=15)
RUNTIME_PREFLIGHT_CLOCK_SKEW = timedelta(minutes=1)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ENVIRONMENT_IDENTITY_FIELDS = (
    "schema_version",
    "name",
    "status",
    "python",
    "locks",
    "expected_packages",
    "packages",
    "runtime_imports",
    "audited_overrides",
    "issues",
)
_ENVIRONMENT_RECEIPT_FIELDS = frozenset((*_ENVIRONMENT_IDENTITY_FIELDS, "created_at_utc"))


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
    if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (
        opened.st_dev,
        opened.st_ino,
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


def _validate_environment_receipt(
    name: str,
    record: object,
    *,
    project_root: Path,
    expected_path: Path,
    context: str,
) -> tuple[Path, dict[str, Any], str]:
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
        or receipt.get("schema_version") != 1
        or receipt.get("name") != name
        or receipt.get("status") != status
        or receipt.get("issues") != []
    ):
        raise RuntimePreflightError(f"{context} receipt content mismatch")
    _timestamp(receipt.get("created_at_utc"), f"{context} created_at_utc")

    python = receipt.get("python")
    if (
        not isinstance(python, dict)
        or set(python) != {"executable", "version", "implementation", "platform"}
        or python.get("version") != EXPECTED_PYTHON_VERSION
        or python.get("implementation") != "CPython"
        or not isinstance(python.get("platform"), str)
        or not python["platform"]
    ):
        raise RuntimePreflightError(f"{context} has an invalid Python identity")
    executable = _evidence_path(python.get("executable"), f"{context} Python")
    if not executable.is_file():
        raise RuntimePreflightError(f"{context} Python is missing: {executable}")

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
    expected_imports = [
        {"module": module, "symbols": list(symbols)}
        for module, symbols in ENVIRONMENT_RUNTIME_IMPORTS[name]
    ]
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
        or runtime_imports != expected_imports
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
        snapshot_digests={"environments/uv.version": uv_digest},
    )
    environments = bootstrap.get("environments")
    if not isinstance(environments, dict) or set(environments) != set(ENVIRONMENT_LOCK_PATHS):
        raise RuntimePreflightError("bootstrap receipt has an unexpected environment set")
    environment_receipts: dict[str, dict[str, Any]] = {}
    for name in ENVIRONMENT_LOCK_PATHS:
        _path, environment_receipts[name], _digest = _validate_environment_receipt(
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
) -> dict[str, str]:
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
    for name in ENVIRONMENT_LOCK_PATHS:
        context = f"runtime environment {name}"
        _path, current, digest = _validate_environment_receipt(
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
    return digests


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


def _locked_weight_artifacts(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = document.get("artifacts")
    if document.get("schema_version") != 1 or not isinstance(artifacts, list) or not artifacts:
        raise RuntimePreflightError("weights lock has an invalid artifact inventory")
    records: dict[str, dict[str, Any]] = {}
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
            or not artifact_id
            or artifact_id in records
            or provider not in {"huggingface", "https", "manual"}
            or type(required) is not bool
            or not isinstance(destination, str)
            or not destination
        ):
            raise RuntimePreflightError(f"{context} is malformed")
        if provider == "huggingface" and (
            not isinstance(artifact.get("repo_id"), str)
            or not artifact["repo_id"]
            or not isinstance(artifact.get("revision"), str)
            or not artifact["revision"]
        ):
            raise RuntimePreflightError(f"{context} has no immutable Hugging Face identity")
        if provider == "https" and (
            not isinstance(artifact.get("url"), str) or not artifact["url"]
        ):
            raise RuntimePreflightError(f"{context} has no immutable HTTPS identity")
        records[artifact_id] = artifact
    return records


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
) -> tuple[Path, str, str, str]:
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
            artifact_destinations.add(destination_value)
            completed_artifact_ids.add(artifact_id)
        elif artifact.get("destination") not in {None, locked.get("destination")}:
            raise RuntimePreflightError(f"{context} identity disagrees with weights lock")
    if not layout_sources.issubset(completed_artifact_ids):
        raise RuntimePreflightError("materialized layouts reference incomplete source artifacts")
    return weights_root, materialization_digest, marker_digest, weights_digest


def validate_runtime_preflight(
    receipt_path: Path,
    *,
    config_path: Path | None,
    project_root: Path,
    require_recent: bool = False,
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
    if require_recent:
        now = datetime.now(timezone.utc)
        if (
            created_at > now + RUNTIME_PREFLIGHT_CLOCK_SKEW
            or now - created_at > RUNTIME_PREFLIGHT_MAX_AGE
        ):
            raise RuntimePreflightError(
                "runtime preflight is not recent enough to start a real run"
            )
    commit = require_clean_git_commit(project_root)
    if receipt.get("project_commit") != commit:
        raise RuntimePreflightError("runtime preflight receipt is bound to another commit")

    config = receipt.get("config")
    if not isinstance(config, dict):
        raise RuntimePreflightError("runtime preflight receipt has no config binding")
    recorded_config_path = _evidence_path(config.get("path"), "runtime config")
    config_path = recorded_config_path if config_path is None else config_path.resolve()
    if recorded_config_path != config_path:
        raise RuntimePreflightError("runtime preflight receipt is bound to another config")
    _config_payload, config_digest = _read_regular_evidence(config_path, "runtime config")
    if not _valid_digest(config.get("sha256")) or config["sha256"] != config_digest:
        raise RuntimePreflightError("runtime config changed after preflight")

    weights_lock, weights_lock_digest = _load_snapshot(
        project_root / "weights-lock.json",
        "weights lock",
    )
    _validate_current_lock_set(
        receipt.get("locks"),
        expected=LOCK_PATHS,
        project_root=project_root,
        label="runtime preflight receipt",
        snapshot_digests={"weights-lock.json": weights_lock_digest},
    )
    _bootstrap_path, bootstrap_digest, baseline_environments = _validate_bootstrap(
        receipt.get("bootstrap"),
        project_root=project_root,
        commit=commit,
    )
    runtime_environment_digests = _validate_runtime_environments(
        receipt.get("runtime_environments"),
        receipt_path=receipt_path,
        project_root=project_root,
        baseline=baseline_environments,
        stage_started_at=stage_started_at,
        preflight_created_at=created_at,
    )
    (
        weights_root,
        materialization_digest,
        marker_digest,
        weights_digest,
    ) = _validate_materialization(
        receipt.get("materialization"),
        receipt.get("materialization_marker"),
        receipt.get("source_weights_receipt"),
        project_root=project_root,
        commit=commit,
        weights_lock=weights_lock,
        weights_lock_digest=weights_lock_digest,
    )
    return {
        "runtime_evidence_verified": True,
        "runtime_preflight_receipt": str(receipt_path),
        "runtime_preflight_sha256": receipt_digest,
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
