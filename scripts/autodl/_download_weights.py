#!/usr/bin/env python3
"""Download immutable weight artifacts and write a content-hash receipt."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from typing import Any

from scaleguard.strict_json import StrictJSONError, loads_object


class ManifestError(ValueError):
    """Raised when a weight manifest is unsafe or not reproducible."""


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: pathlib.Path) -> dict[str, Any]:
    try:
        raw_document = loads_object(path.read_text(encoding="utf-8"))
    except (OSError, StrictJSONError) as exc:
        raise ManifestError(f"cannot read JSON manifest {path}: {exc}") from exc
    document: dict[str, Any] = raw_document
    if document.get("schema_version") not in {1, "1", "1.0"}:
        raise ManifestError("weight manifest schema_version must be 1")
    artifacts = document.get("artifacts", document.get("models"))
    if not isinstance(artifacts, list) or not artifacts:
        raise ManifestError("weight manifest must contain a non-empty artifacts list")
    document["artifacts"] = artifacts
    return document


def safe_destination(root: pathlib.Path, value: object) -> pathlib.Path:
    root = root.resolve()
    if not isinstance(value, str) or not value:
        raise ManifestError("every artifact needs a non-empty relative destination")
    relative = pathlib.PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ManifestError(f"destination must stay below the weight root: {value!r}")
    unresolved = root / pathlib.Path(*relative.parts)
    current = unresolved
    while current != root:
        if current.is_symlink():
            raise ManifestError(f"destination contains a symlink: {value!r}")
        current = current.parent
    destination = unresolved.resolve()
    if not destination.is_relative_to(root):
        raise ManifestError(f"destination escapes the weight root: {value!r}")
    return destination


def validate_artifact(artifact: object, root: pathlib.Path) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise ManifestError("each artifact must be a JSON object")
    artifact_id = artifact.get("id")
    if not isinstance(artifact_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", artifact_id
    ):
        raise ManifestError(f"invalid artifact id: {artifact_id!r}")
    provider = artifact.get("provider")
    if provider is None and artifact.get("repo_id"):
        provider = "huggingface"
    if provider not in {"huggingface", "https", "manual"}:
        raise ManifestError(f"{artifact_id}: provider must be 'huggingface', 'https', or 'manual'")

    normalized = dict(artifact)
    normalized["provider"] = provider
    required = artifact.get("required", True)
    if not isinstance(required, bool):
        raise ManifestError(f"{artifact_id}: required must be a boolean")
    normalized["required"] = required
    if provider == "manual":
        files = artifact.get("files", [])
        if not isinstance(files, list) or not all(
            isinstance(item, str)
            and item
            and not pathlib.PurePosixPath(item).is_absolute()
            and ".." not in pathlib.PurePosixPath(item).parts
            for item in files
        ):
            raise ManifestError(f"{artifact_id}: manual files must be safe relative path strings")
        destination_value = artifact.get("destination")
        if destination_value is not None:
            normalized["_destination"] = safe_destination(root, destination_value)
        normalized["_instructions"] = artifact.get(
            "instructions",
            artifact.get(
                "manual_steps",
                artifact.get("notes", artifact.get("reason")),
            ),
        )
        return normalized

    destination = safe_destination(root, artifact.get("destination", artifact_id))
    normalized["_destination"] = destination
    if provider == "huggingface":
        repo_id = artifact.get("repo_id")
        revision = artifact.get("revision")
        if not isinstance(repo_id, str) or not re.fullmatch(r"[^/\s]+/[^/\s]+", repo_id):
            raise ManifestError(f"{artifact_id}: invalid Hugging Face repo_id")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ManifestError(
                f"{artifact_id}: revision must be an immutable 40-character commit SHA"
            )
        files = artifact.get("files")
        inferred_files: list[str] = []
        if isinstance(files, list):
            for item in files:
                if isinstance(item, str):
                    inferred_files.append(item)
                elif isinstance(item, dict) and isinstance(item.get("path"), str):
                    inferred_files.append(item["path"])
        includes = artifact.get("include", inferred_files)
        excludes = artifact.get("exclude", [])
        includes = [] if includes is None else includes
        excludes = [] if excludes is None else excludes
        if not isinstance(includes, list) or not all(isinstance(item, str) for item in includes):
            raise ManifestError(f"{artifact_id}: include must be a list of strings")
        if not isinstance(excludes, list) or not all(isinstance(item, str) for item in excludes):
            raise ManifestError(f"{artifact_id}: exclude must be a list of strings")
        if includes == ["**/*"]:
            # Hugging Face uses fnmatchcase semantics: **/* omits repository-root
            # files such as model_index.json. An all-files lock must therefore
            # omit --include and let the pinned snapshot download in full.
            includes = []
        normalized["include"] = includes
        normalized["exclude"] = excludes
    else:
        url = artifact.get("url")
        expected_sha256 = artifact.get("sha256", artifact.get("known_sha256"))
        if not isinstance(url, str):
            raise ManifestError(f"{artifact_id}: HTTPS artifact needs a URL")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ManifestError(f"{artifact_id}: only public HTTPS URLs are accepted")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ManifestError(
                f"{artifact_id}: credential-bearing, signed, query, or fragment URLs are forbidden"
            )
        if not isinstance(expected_sha256, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", expected_sha256
        ):
            raise ManifestError(f"{artifact_id}: HTTPS downloads require an expected sha256")
        if destination.suffix == "":
            raise ManifestError(f"{artifact_id}: HTTPS destination must include a filename")
        normalized["sha256"] = expected_sha256
    return normalized


def manual_inventory(artifact: dict[str, Any]) -> list[dict[str, object]]:
    destination = artifact.get("_destination")
    if not isinstance(destination, pathlib.Path) or not destination.exists():
        return []
    files = inventory(destination)
    if not files:
        return []
    for item in files:
        size = item.get("size_bytes")
        if not isinstance(size, int) or size <= 0:
            return []
    declared = artifact.get("files")
    if not isinstance(declared, list) or not declared:
        return files
    actual_by_name = {str(item["path"]): item for item in files}
    actual_names = set(actual_by_name)
    declared_names = {item for item in declared if isinstance(item, str)}
    if destination.is_file():
        actual_names.add(destination.name)
    if not declared_names.issubset(actual_names):
        return []
    for name in declared_names:
        if name not in actual_by_name:
            continue
        size = actual_by_name[name].get("size_bytes")
        if not isinstance(size, int) or size <= 0:
            return []
    return files


def expected_hashes(artifact: dict[str, Any]) -> dict[str, str]:
    if artifact["provider"] == "https":
        destination: pathlib.Path = artifact["_destination"]
        return {destination.name: str(artifact["sha256"]).lower()}

    expected: dict[str, str] = {}
    known = artifact.get("known_sha256")
    if isinstance(known, dict):
        for path, value in known.items():
            if value is None:
                continue
            if not isinstance(path, str) or not isinstance(value, str):
                raise ManifestError(f"{artifact['id']}: known_sha256 must map paths to hashes")
            if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
                raise ManifestError(f"{artifact['id']}: invalid known SHA-256 for {path}")
            expected[path] = value.lower()

    files = artifact.get("files")
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            item_path = item.get("path")
            if not isinstance(item_path, str):
                continue
            value = item.get("sha256")
            if value is None:
                continue
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
                raise ManifestError(f"{artifact['id']}: invalid file SHA-256 for {item_path}")
            expected[item_path] = value.lower()

    if isinstance(known, str):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", known):
            raise ManifestError(f"{artifact['id']}: invalid known_sha256")
        file_paths: list[str] = []
        if isinstance(files, list):
            for item in files:
                if isinstance(item, str):
                    file_paths.append(item)
                elif isinstance(item, dict) and isinstance(item.get("path"), str):
                    file_paths.append(item["path"])
        if len(file_paths) != 1:
            raise ManifestError(f"{artifact['id']}: scalar known_sha256 requires exactly one file")
        expected[file_paths[0]] = known.lower()
    return expected


def huggingface_command() -> list[str]:
    executable = shutil.which("hf")
    if executable:
        return [executable, "download"]
    executable = shutil.which("huggingface-cli")
    if executable:
        return [executable, "download"]
    raise ManifestError(
        "neither 'hf' nor 'huggingface-cli' is installed; install huggingface_hub[cli]"
    )


def download_huggingface(artifact: dict[str, Any]) -> None:
    destination: pathlib.Path = artifact["_destination"]
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".scaleguard-source.json"
    source_identity = {
        "schema_version": 1,
        "provider": "huggingface",
        "repo_id": artifact["repo_id"],
        "revision": artifact["revision"],
    }
    if marker.exists():
        try:
            previous = loads_object(marker.read_text(encoding="utf-8"))
        except (OSError, StrictJSONError) as exc:
            raise ManifestError(f"{artifact['id']}: invalid source marker: {exc}") from exc
        for key in ("provider", "repo_id", "revision"):
            if previous.get(key) != source_identity[key]:
                raise ManifestError(
                    f"{artifact['id']}: destination is bound to another {key}; "
                    "choose a new destination"
                )
    elif any(destination.iterdir()):
        raise ManifestError(
            f"{artifact['id']}: non-empty destination has no ScaleGuard source marker; "
            "choose a new destination or audit and relocate the existing files"
        )
    source_identity["status"] = "downloading"
    marker.write_text(json.dumps(source_identity, indent=2) + "\n", encoding="utf-8")

    command = huggingface_command()
    command.extend(
        [
            artifact["repo_id"],
            "--repo-type",
            "model",
            "--revision",
            artifact["revision"],
            "--local-dir",
            str(destination),
        ]
    )
    if artifact.get("include"):
        command.append("--include")
        command.extend(artifact["include"])
    if artifact.get("exclude"):
        command.append("--exclude")
        command.extend(artifact["exclude"])
    subprocess.run(command, check=True)
    source_identity["status"] = "complete"
    marker.write_text(json.dumps(source_identity, indent=2) + "\n", encoding="utf-8")


def download_https(artifact: dict[str, Any]) -> None:
    destination: pathlib.Path = artifact["_destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = artifact["sha256"].lower()
    if destination.is_file() and sha256(destination) == expected:
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".part",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        request = urllib.request.Request(
            artifact["url"],
            headers={"User-Agent": "ScaleGuard-4K weight fetcher/1"},
        )
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=1024 * 1024)
        actual = sha256(temporary)
        if actual != expected:
            raise ManifestError(
                f"{artifact['id']}: sha256 mismatch; expected {expected}, got {actual}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def inventory(destination: pathlib.Path) -> list[dict[str, object]]:
    if destination.is_symlink():
        raise ManifestError(f"artifact destination is a symlink: {destination}")
    if destination.is_file():
        paths = [destination]
        root = destination.parent
    else:
        discovered = [
            path
            for path in sorted(destination.rglob("*"))
            if ".cache" not in path.relative_to(destination).parts
            and ".git" not in path.relative_to(destination).parts
        ]
        symlinks = [path for path in discovered if path.is_symlink()]
        if symlinks:
            raise ManifestError(f"artifact inventory contains a symlink: {symlinks[0]}")
        paths = [path for path in discovered if path.is_file()]
        root = destination
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("weight_root", type=pathlib.Path)
    parser.add_argument("receipt", type=pathlib.Path)
    parser.add_argument("--git-commit")
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()

    root = args.weight_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    document = load_manifest(args.manifest.resolve())
    artifacts = [validate_artifact(item, root) for item in document["artifacts"]]
    artifact_ids = [item["id"] for item in artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ManifestError("artifact ids must be unique")
    destinations = [
        item["_destination"]
        for item in artifacts
        if isinstance(item.get("_destination"), pathlib.Path)
    ]
    if len(destinations) != len(set(destinations)):
        raise ManifestError("artifact destinations must be unique")
    for index, destination in enumerate(destinations):
        for other in destinations[index + 1 :]:
            if destination in other.parents or other in destination.parents:
                raise ManifestError(
                    "artifact destinations must not be nested: "
                    f"{destination.relative_to(root)} and {other.relative_to(root)}"
                )
    completed: list[dict[str, object]] = []
    manual_artifacts = [artifact for artifact in artifacts if artifact["provider"] == "manual"]
    manual_files = {
        str(artifact["id"]): manual_inventory(artifact) for artifact in manual_artifacts
    }
    required_manual_gates = [
        {
            "id": artifact["id"],
            "required": True,
            "destination": (
                artifact["_destination"].relative_to(root).as_posix()
                if isinstance(artifact.get("_destination"), pathlib.Path)
                else None
            ),
            "instructions": artifact.get("_instructions"),
            "source": artifact.get("source"),
            "license": artifact.get("license"),
        }
        for artifact in manual_artifacts
        if artifact["required"] and not manual_files[str(artifact["id"])]
    ]
    for artifact in artifacts:
        if not artifact["required"] and not args.include_optional:
            completed.append(
                {
                    "id": artifact["id"],
                    "provider": artifact["provider"],
                    "status": "skipped",
                    "required": False,
                    "reason": "optional artifact was not requested",
                }
            )
            continue
        if artifact["provider"] == "manual":
            supplied_files = manual_files[str(artifact["id"])]
            if supplied_files:
                manual_destination: pathlib.Path = artifact["_destination"]
                known_hashes = expected_hashes(artifact)
                actual_by_path = {str(item["path"]): str(item["sha256"]) for item in supplied_files}
                for relative, expected in known_hashes.items():
                    actual = actual_by_path.get(relative)
                    if actual is None:
                        raise ManifestError(
                            f"{artifact['id']}: expected manual file is missing: {relative}"
                        )
                    if actual != expected:
                        raise ManifestError(
                            f"{artifact['id']}: manual SHA-256 mismatch for {relative}; "
                            f"expected {expected}, got {actual}"
                        )
                completed.append(
                    {
                        "id": artifact["id"],
                        "provider": "manual",
                        "status": "recorded_manual",
                        "required": artifact["required"],
                        "destination": manual_destination.relative_to(root).as_posix(),
                        "files": supplied_files,
                        "known_hashes_verified": sorted(known_hashes),
                        "upstream_digest_authenticated": bool(known_hashes),
                        "verify_on_download": bool(artifact.get("verify_on_download", False)),
                        "license": artifact.get("license"),
                        "source": artifact.get("source"),
                        "claim": (
                            "The supplied manual files matched the locked upstream digests."
                            if known_hashes
                            else (
                                "Content hashes were measured locally, but no upstream "
                                "digest was available to authenticate this manual file."
                            )
                        ),
                    }
                )
                continue
            if artifact["required"]:
                completed.append(
                    {
                        "id": artifact["id"],
                        "provider": "manual",
                        "status": "external_gate",
                        "required": True,
                        "destination": (
                            artifact["_destination"].relative_to(root).as_posix()
                            if isinstance(artifact.get("_destination"), pathlib.Path)
                            else None
                        ),
                        "instructions": artifact.get("_instructions"),
                        "source": artifact.get("source"),
                    }
                )
                continue
            raise ManifestError(
                f"{artifact['id']}: optional manual artifact was requested but is absent"
            )
        print(f"fetching {artifact['id']} ({artifact['provider']})", flush=True)
        if artifact["provider"] == "huggingface":
            download_huggingface(artifact)
        else:
            download_https(artifact)
        files = inventory(artifact["_destination"])
        if not files:
            raise ManifestError(f"{artifact['id']}: download produced no regular files")
        known_hashes = expected_hashes(artifact)
        actual_by_path = {str(item["path"]): str(item["sha256"]) for item in files}
        for relative, expected in known_hashes.items():
            actual = actual_by_path.get(relative)
            if actual is None:
                raise ManifestError(
                    f"{artifact['id']}: expected hashed file is missing: {relative}"
                )
            if actual != expected:
                raise ManifestError(
                    f"{artifact['id']}: SHA-256 mismatch for {relative}; "
                    f"expected {expected}, got {actual}"
                )
        entry: dict[str, object] = {
            "id": artifact["id"],
            "provider": artifact["provider"],
            "status": "downloaded",
            "required": artifact["required"],
            "destination": artifact["_destination"].relative_to(root).as_posix(),
            "files": files,
            "known_hashes_verified": sorted(known_hashes),
            "verify_on_download": bool(artifact.get("verify_on_download", False)),
            "license": artifact.get("license"),
        }
        if artifact["provider"] == "huggingface":
            entry["repo_id"] = artifact["repo_id"]
            entry["revision"] = artifact["revision"]
        else:
            entry["url"] = artifact["url"]
        completed.append(entry)

    receipt_status = "external_gate" if required_manual_gates else "passed"
    receipt = {
        "schema_version": 1,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": receipt_status,
        "status_scope": (
            (
                "Downloadable artifacts were processed, but required manual artifacts "
                "remain external gates and no complete weight set is claimed."
            )
            if required_manual_gates
            else (
                "All required artifacts are present and locally hashed. Manual files "
                "without upstream digests are recorded, not authenticated."
            )
        ),
        "source_manifest": args.manifest.name,
        "source_manifest_sha256": sha256(args.manifest),
        "git_commit": args.git_commit,
        "weight_root": str(root),
        "optional_artifacts_requested": args.include_optional,
        "artifacts": completed,
        "manual_gates": required_manual_gates,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if required_manual_gates:
        for gate in required_manual_gates:
            instructions = gate["instructions"] or "manual access required"
            print(f"EXTERNAL GATE {gate['id']}: {instructions}")
        return 3
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ManifestError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
