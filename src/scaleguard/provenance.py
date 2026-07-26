"""Validation of AutoDL preflight receipts before evidence promotion."""

from __future__ import annotations

import hashlib
import re
import subprocess
from datetime import datetime, timedelta
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
_SHA256 = re.compile(r"[0-9a-f]{64}")


class RuntimePreflightError(ScaleGuardError):
    """Raised when runtime evidence is absent, stale, or inconsistent."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise RuntimePreflightError(f"{label} must not be a symbolic link: {path}")
    try:
        value = loads(path.read_text(encoding="utf-8"))
    except (OSError, StrictJSONError) as error:
        raise RuntimePreflightError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimePreflightError(f"{label} must be a JSON object")
    return value


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
) -> None:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise RuntimePreflightError(f"{label} has an unexpected lock set")
    for relative in expected:
        path = project_root / relative
        if path.is_symlink() or not path.is_file() or value.get(relative) != sha256(path):
            raise RuntimePreflightError(f"{label} lock mismatch: {relative}")


def _validate_environment_receipt(
    name: str,
    record: object,
    *,
    project_root: Path,
) -> None:
    context = f"bootstrap environment {name}"
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "status"}:
        raise RuntimePreflightError(f"{context} has an invalid evidence record")
    expected_relative = f".runtime/receipts/{name}.json"
    if record.get("path") != expected_relative:
        raise RuntimePreflightError(f"{context} references an unexpected receipt")
    path = _evidence_path(record["path"], f"{context} receipt", base=project_root)
    expected_path = (project_root / expected_relative).resolve()
    if path != expected_path:
        raise RuntimePreflightError(f"{context} references an unexpected receipt")
    status = record.get("status")
    expected_status = "passed_with_audited_override" if name == "4kagent" else "passed"
    if status != expected_status:
        raise RuntimePreflightError(f"{context} has an invalid status")
    if not _valid_digest(record.get("sha256")) or record["sha256"] != sha256(path):
        raise RuntimePreflightError(f"{context} receipt hash mismatch")

    receipt = _load(path, f"{context} receipt")
    if (
        receipt.get("schema_version") != 1
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
        path = _evidence_path(lock_record.get("path"), f"{context} lock")
        try:
            relative = path.relative_to(project_root).as_posix()
        except ValueError as error:
            raise RuntimePreflightError(f"{context} lock escapes the project") from error
        pinned = lock_record.get("pinned_packages")
        if (
            relative not in expected_locks
            or relative in observed_locks
            or path.is_symlink()
            or not path.is_file()
            or not _valid_digest(lock_record.get("sha256"))
            or lock_record["sha256"] != sha256(path)
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


def _validate_bootstrap(
    record: object,
    *,
    project_root: Path,
    commit: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, dict):
        raise RuntimePreflightError("runtime preflight receipt has no bootstrap binding")
    path = _evidence_path(record.get("path"), "bootstrap receipt")
    expected_path = (project_root / ".runtime" / "receipts" / "bootstrap.json").resolve()
    if path != expected_path:
        raise RuntimePreflightError("runtime preflight references an unexpected bootstrap receipt")
    if not _valid_digest(record.get("sha256")) or record["sha256"] != sha256(path):
        raise RuntimePreflightError("bootstrap receipt is stale or did not pass")
    bootstrap = _load(path, "bootstrap receipt")
    if (
        bootstrap.get("schema_version") != 1
        or bootstrap.get("status") != "passed"
        or bootstrap.get("project_commit") != commit
    ):
        raise RuntimePreflightError("bootstrap receipt is stale or did not pass")
    _timestamp(bootstrap.get("created_at_utc"), "bootstrap created_at_utc")
    if bootstrap.get("python_version") != EXPECTED_PYTHON_VERSION:
        raise RuntimePreflightError("bootstrap receipt has an unexpected Python version")
    expected_uv = (project_root / "environments" / "uv.version").read_text(encoding="utf-8").strip()
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
    )
    environments = bootstrap.get("environments")
    if not isinstance(environments, dict) or set(environments) != set(ENVIRONMENT_LOCK_PATHS):
        raise RuntimePreflightError("bootstrap receipt has an unexpected environment set")
    for name in ENVIRONMENT_LOCK_PATHS:
        _validate_environment_receipt(name, environments[name], project_root=project_root)
    return path, bootstrap


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
    return [
        {
            "path": path.relative_to(inventory_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]


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


def _locked_weight_artifacts(project_root: Path) -> dict[str, dict[str, Any]]:
    document = _load(project_root / "weights-lock.json", "weights lock")
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
    materialization = _load(materialization_path, "materialization verification receipt")
    root = _weights_root(materialization.get("weights_root"))
    marker_path = root / ".scaleguard-materialization.json"
    marker = _load(marker_path, "fixed materialization marker")
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
        if sha256(resolved) == expected:
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
) -> tuple[Path, Path, Path, Path]:
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
    if not _valid_digest(materialization_record.get("sha256")) or materialization_record[
        "sha256"
    ] != sha256(materialization_path):
        raise RuntimePreflightError("materialization verification receipt is stale or did not pass")
    materialization = _load(
        materialization_path,
        "materialization verification receipt",
    )
    weights_root = _weights_root(materialization.get("weights_root"))

    marker_path = _evidence_path(marker_record.get("path"), "fixed materialization marker")
    expected_marker = (weights_root / ".scaleguard-materialization.json").resolve()
    if marker_path != expected_marker:
        raise RuntimePreflightError("runtime preflight references an unexpected fixed marker")
    if not _valid_digest(marker_record.get("sha256")) or marker_record["sha256"] != sha256(
        marker_path
    ):
        raise RuntimePreflightError("fixed materialization marker is stale")
    marker = _load(marker_path, "fixed materialization marker")
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
    if (
        not _valid_digest(weights_record.get("sha256"))
        or weights_record["sha256"] != sha256(weights_path)
        or marker.get("source_weights_receipt_sha256") != sha256(weights_path)
    ):
        raise RuntimePreflightError("source weights receipt is stale or not bound")
    weights = _load(weights_path, "source weights receipt")
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
        or weights.get("source_manifest_sha256") != sha256(project_root / "weights-lock.json")
        or type(weights.get("optional_artifacts_requested")) is not bool
    ):
        raise RuntimePreflightError("source weights receipt has an invalid manifest binding")

    artifacts = weights.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimePreflightError("source weights receipt has no artifact records")
    locked_artifacts = _locked_weight_artifacts(project_root)
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
    return materialization_path, marker_path, weights_path, weights_root


def validate_runtime_preflight(
    receipt_path: Path,
    *,
    config_path: Path | None,
    project_root: Path,
) -> dict[str, Any]:
    """Bind one run to clean source, config, environments, and current weight bytes."""

    project_root = project_root.resolve()
    receipt_path = _evidence_path(str(receipt_path), "runtime preflight receipt")
    receipt = _load(receipt_path, "runtime preflight receipt")
    if receipt.get("schema_version") != 1 or receipt.get("status") != "passed":
        raise RuntimePreflightError("runtime preflight receipt did not pass")
    _timestamp(receipt.get("created_at_utc"), "runtime preflight created_at_utc")
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
    if not _valid_digest(config.get("sha256")) or config["sha256"] != sha256(config_path):
        raise RuntimePreflightError("runtime config changed after preflight")

    _validate_current_lock_set(
        receipt.get("locks"),
        expected=LOCK_PATHS,
        project_root=project_root,
        label="runtime preflight receipt",
    )
    bootstrap_path, _ = _validate_bootstrap(
        receipt.get("bootstrap"),
        project_root=project_root,
        commit=commit,
    )
    (
        materialization_path,
        marker_path,
        weights_path,
        weights_root,
    ) = _validate_materialization(
        receipt.get("materialization"),
        receipt.get("materialization_marker"),
        receipt.get("source_weights_receipt"),
        project_root=project_root,
        commit=commit,
    )
    return {
        "runtime_evidence_verified": True,
        "runtime_preflight_receipt": str(receipt_path),
        "runtime_preflight_sha256": sha256(receipt_path),
        "bootstrap_receipt_sha256": sha256(bootstrap_path),
        "materialization_receipt_sha256": sha256(materialization_path),
        "materialization_marker_sha256": sha256(marker_path),
        "source_weights_receipt_sha256": sha256(weights_path),
        "weights_root": str(weights_root),
        "project_commit": commit,
        "project_root": str(project_root),
        "runtime_config_path": str(config_path),
    }
