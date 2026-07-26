"""Verification of immutable upstream checkouts and patch overlays."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scaleguard.errors import UpstreamVerificationError
from scaleguard.runtime.process import minimal_subprocess_environment
from scaleguard.strict_yaml import StrictYAMLError
from scaleguard.strict_yaml import loads as load_strict_yaml


@dataclass(frozen=True, slots=True)
class Verification:
    target: str
    check: str
    ok: bool
    detail: str


def _run_git(checkout: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=False,
        capture_output=True,
        text=True,
        env=minimal_subprocess_environment(),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _normalize_url(value: str) -> str:
    return value.removesuffix("/").removesuffix(".git")


def _locked_path(root: Path, value: object) -> tuple[Path | None, str | None]:
    if not isinstance(value, str) or not value:
        return None, "path must be a non-empty string"
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None, f"path must be relative and confined to its root: {value!r}"
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        return None, f"path escapes its root: {value!r}"
    return resolved, None


def _patch_paths(checkout: Path, patch_path: Path) -> tuple[set[str], str | None]:
    """Return paths declared by a patch without trusting lock-file annotations."""
    parsed = _run_git(checkout, "apply", "--numstat", "-z", str(patch_path))
    if parsed.returncode != 0:
        return set(), parsed.stderr.strip() or f"cannot parse {patch_path.name}"

    paths: set[str] = set()
    for record in parsed.stdout.split("\0"):
        if not record:
            continue
        columns = record.split("\t", 2)
        if len(columns) != 3 or not columns[2]:
            return set(), f"unexpected numstat record in {patch_path.name}"
        paths.add(columns[2])
    return paths, None


def _status_paths(checkout: Path) -> tuple[set[str], str | None]:
    status = _run_git(checkout, "status", "--porcelain=v1", "-z")
    if status.returncode != 0:
        return set(), status.stderr.strip() or "cannot read Git worktree status"
    records = status.stdout.split("\0")
    paths: set[str] = set()
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4 or record[2] != " ":
            return set(), "invalid Git porcelain status record"
        code = record[:2]
        paths.add(record[3:])
        if "R" in code or "C" in code:
            if index >= len(records) or not records[index]:
                return set(), "incomplete Git porcelain rename record"
            paths.add(records[index])
            index += 1
    return paths, None


def _ignored_paths(checkout: Path) -> tuple[set[str], str | None]:
    ignored = _run_git(
        checkout,
        "ls-files",
        "-z",
        "--others",
        "--ignored",
        "--exclude-standard",
    )
    if ignored.returncode != 0:
        return set(), ignored.stderr.strip() or "cannot read ignored checkout artifacts"
    return {path for path in ignored.stdout.split("\0") if path}, None


def _path_summary(paths: list[str]) -> str:
    visible = paths[:20]
    summary = ", ".join(visible)
    if len(paths) > len(visible):
        summary += f", ... ({len(paths) - len(visible)} more)"
    return summary


def load_upstream_lock(path: Path, mapping: str = "repositories") -> dict[str, Any]:
    try:
        data = load_strict_yaml(path.read_text(encoding="utf-8"))
    except (OSError, StrictYAMLError) as error:
        raise UpstreamVerificationError(f"cannot load upstream lock {path}: {error}") from error
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or not isinstance(data.get(mapping), dict)
    ):
        raise UpstreamVerificationError(f"upstream lock must contain a {mapping} mapping")
    return data


def verify_upstreams(
    lock_path: Path,
    project_root: Path,
    mapping: str = "repositories",
) -> list[Verification]:
    lock = load_upstream_lock(lock_path, mapping)
    results: list[Verification] = []
    for key, raw in lock[mapping].items():
        if not isinstance(raw, dict):
            results.append(Verification(str(key), "schema", False, "entry must be a mapping"))
            continue
        checkout_value = raw.get("checkout", f"third_party/checkouts/{key}")
        checkout, checkout_error = _locked_path(project_root, checkout_value)
        if checkout_error is not None or checkout is None:
            results.append(
                Verification(str(key), "checkout", False, checkout_error or "invalid checkout")
            )
            continue
        if not (checkout / ".git").exists():
            results.append(
                Verification(str(key), "checkout", False, f"missing Git checkout: {checkout}")
            )
            continue
        expected_url = raw.get("url")
        if isinstance(expected_url, str):
            origin = _run_git(checkout, "remote", "get-url", "origin")
            actual_origin = origin.stdout.strip()
            results.append(
                Verification(
                    str(key),
                    "origin",
                    origin.returncode == 0
                    and _normalize_url(actual_origin) == _normalize_url(expected_url),
                    f"expected {expected_url}, found {actual_origin or origin.stderr.strip()}",
                )
            )
            push_url = _run_git(checkout, "remote", "get-url", "--push", "origin")
            actual_push_url = push_url.stdout.strip()
            results.append(
                Verification(
                    str(key),
                    "push_url",
                    push_url.returncode == 0 and actual_push_url == "DISABLED",
                    (
                        "push disabled"
                        if actual_push_url == "DISABLED"
                        else (
                            f"expected DISABLED, found {actual_push_url or push_url.stderr.strip()}"
                        )
                    ),
                )
            )
        expected_commit = str(raw.get("commit", ""))
        expected_tree = str(raw.get("tree", ""))
        commit = _run_git(checkout, "rev-parse", "HEAD")
        actual_commit = commit.stdout.strip()
        results.append(
            Verification(
                str(key),
                "commit",
                commit.returncode == 0 and actual_commit == expected_commit,
                f"expected {expected_commit}, found {actual_commit or commit.stderr.strip()}",
            )
        )
        tree = _run_git(checkout, "rev-parse", "HEAD^{tree}")
        actual_tree = tree.stdout.strip()
        results.append(
            Verification(
                str(key),
                "tree",
                tree.returncode == 0 and actual_tree == expected_tree,
                f"expected {expected_tree}, found {actual_tree or tree.stderr.strip()}",
            )
        )
        patches = raw.get("patches", [])
        if not isinstance(patches, list):
            results.append(Verification(str(key), "patches", False, "patches must be a list"))
            continue
        allowed_changes: set[str] = set()
        for patch_raw in patches:
            if not isinstance(patch_raw, dict) or "path" not in patch_raw:
                results.append(Verification(str(key), "patch", False, "invalid patch entry"))
                continue
            patch_path, patch_path_error = _locked_path(project_root, patch_raw["path"])
            if patch_path_error is not None or patch_path is None:
                results.append(
                    Verification(
                        str(key),
                        "patch_file",
                        False,
                        patch_path_error or "invalid patch path",
                    )
                )
                continue
            if not patch_path.is_file():
                results.append(Verification(str(key), "patch_file", False, f"missing {patch_path}"))
                continue
            expected_sha = str(patch_raw.get("sha256", ""))
            actual_sha = _sha256(patch_path)
            results.append(
                Verification(
                    str(key),
                    "patch_sha256",
                    bool(expected_sha) and actual_sha == expected_sha,
                    f"{patch_path.name}: expected {expected_sha}, found {actual_sha}",
                )
            )
            declared_paths, parse_error = _patch_paths(checkout, patch_path)
            results.append(
                Verification(
                    str(key),
                    "patch_paths",
                    parse_error is None,
                    parse_error or f"{patch_path.name}: {', '.join(sorted(declared_paths))}",
                )
            )
            allowed_changes.update(declared_paths)
            applied = bool(patch_raw.get("applied", True))
            patch_check_args = (
                ("apply", "--reverse", "--check", str(patch_path))
                if applied
                else ("apply", "--check", str(patch_path))
            )
            patch_check = _run_git(checkout, *patch_check_args)
            results.append(
                Verification(
                    str(key),
                    "patch_applied" if applied else "patch_clean",
                    patch_check.returncode == 0,
                    patch_check.stderr.strip() or "patch state matches lock",
                )
            )
        patched_files = raw.get("patched_files")
        if patches or patched_files is not None:
            if not isinstance(patched_files, dict):
                results.append(
                    Verification(
                        str(key),
                        "patched_files",
                        False,
                        "patched_files must be a mapping",
                    )
                )
                patched_files = {}
            declared_final_paths = {value for value in patched_files if isinstance(value, str)}
            coverage_ok = declared_final_paths == allowed_changes and len(
                declared_final_paths
            ) == len(patched_files)
            results.append(
                Verification(
                    str(key),
                    "patched_file_coverage",
                    coverage_ok,
                    (
                        "final hashes cover every patch-modified path"
                        if coverage_ok
                        else (
                            f"expected {_path_summary(sorted(allowed_changes))}; "
                            f"found {_path_summary(sorted(declared_final_paths))}"
                        )
                    ),
                )
            )
            for relative, expected_hash_raw in patched_files.items():
                final_path, final_path_error = _locked_path(checkout, relative)
                expected_hash = expected_hash_raw if isinstance(expected_hash_raw, str) else ""
                valid_hash = re.fullmatch(r"[0-9a-f]{64}", expected_hash) is not None
                actual_hash = (
                    _sha256(final_path)
                    if final_path_error is None and final_path is not None and final_path.is_file()
                    else ""
                )
                results.append(
                    Verification(
                        str(key),
                        "patched_file_sha256",
                        final_path_error is None and valid_hash and actual_hash == expected_hash,
                        (
                            final_path_error
                            or (
                                f"{relative}: expected {expected_hash}, "
                                f"found {actual_hash or 'missing'}"
                            )
                        ),
                    )
                )
        changed, status_error = _status_paths(checkout)
        unexpected = sorted(changed - allowed_changes)
        results.append(
            Verification(
                str(key),
                "worktree",
                status_error is None and not unexpected,
                "clean or only locked patch changes"
                if status_error is None and not unexpected
                else status_error or f"unexpected changes: {_path_summary(unexpected)}",
            )
        )
        ignored, ignored_error = _ignored_paths(checkout)
        results.append(
            Verification(
                str(key),
                "ignored_artifacts",
                ignored_error is None and not ignored,
                ignored_error
                or ("none" if not ignored else f"unexpected: {_path_summary(sorted(ignored))}"),
            )
        )
    return results
