#!/usr/bin/env python3
"""Materialize immutable Git checkouts and ordered patch overlays."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from scaleguard.runtime.process import minimal_subprocess_environment
from scaleguard.strict_yaml import StrictYAMLError
from scaleguard.strict_yaml import loads as load_strict_yaml


class MaterializeError(RuntimeError):
    pass


def run(*argv: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=minimal_subprocess_environment(),
    )


def require_ok(result: subprocess.CompletedProcess[str], action: str) -> str:
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise MaterializeError(f"{action} failed: {detail}")
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_entries(lock_path: Path, mapping_name: str) -> dict[str, dict[str, Any]]:
    try:
        document = load_strict_yaml(lock_path.read_text(encoding="utf-8"))
    except (OSError, StrictYAMLError) as error:
        raise MaterializeError(f"cannot read {lock_path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise MaterializeError(f"{lock_path} must use schema_version 1")
    raw_entries = document.get(mapping_name)
    if not isinstance(raw_entries, dict) or not raw_entries:
        raise MaterializeError(f"{lock_path} must contain a non-empty {mapping_name} mapping")
    entries: dict[str, dict[str, Any]] = {}
    for key, value in raw_entries.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise MaterializeError(f"invalid {mapping_name} entry: {key!r}")
        entries[key] = value
    return entries


def safe_project_path(project_root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MaterializeError(f"{field} must be a non-empty project-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise MaterializeError(f"{field} must stay inside the project: {value!r}")
    resolved = (project_root / relative).resolve()
    if not resolved.is_relative_to(project_root):
        raise MaterializeError(f"{field} escapes the project: {value!r}")
    return resolved


def safe_relative_path(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MaterializeError(f"{field} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise MaterializeError(f"{field} must stay inside its root: {value!r}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise MaterializeError(f"{field} escapes its root: {value!r}")
    return resolved


def normalize_url(value: str) -> str:
    return value.removesuffix("/").removesuffix(".git")


def patch_paths(checkout: Path, patch: Path) -> set[str]:
    parsed = run("git", "apply", "--numstat", "-z", str(patch), cwd=checkout)
    output = require_ok(parsed, f"parse patch {patch.name}")
    paths: set[str] = set()
    for record in output.split("\0"):
        if not record:
            continue
        columns = record.split("\t", 2)
        if len(columns) != 3 or not columns[2]:
            raise MaterializeError(f"unexpected numstat record in {patch.name}")
        paths.add(columns[2])
    return paths


def status_paths(checkout: Path, action: str) -> set[str]:
    result = run("git", "status", "--porcelain=v1", "-z", cwd=checkout)
    if result.returncode != 0:
        require_ok(result, action)
    records = result.stdout.split("\0")
    paths: set[str] = set()
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4 or record[2] != " ":
            raise MaterializeError(f"{action} returned an invalid porcelain record")
        code = record[:2]
        paths.add(record[3:])
        if "R" in code or "C" in code:
            if index >= len(records) or not records[index]:
                raise MaterializeError(f"{action} returned an incomplete rename record")
            paths.add(records[index])
            index += 1
    return paths


def ignored_paths(checkout: Path, action: str) -> set[str]:
    result = run(
        "git",
        "ls-files",
        "-z",
        "--others",
        "--ignored",
        "--exclude-standard",
        cwd=checkout,
    )
    output = require_ok(result, action)
    return {path for path in output.split("\0") if path}


def path_summary(paths: list[str]) -> str:
    visible = paths[:20]
    summary = ", ".join(visible)
    if len(paths) > len(visible):
        summary += f", ... ({len(paths) - len(visible)} more)"
    return summary


def ensure_checkout(
    key: str,
    entry: dict[str, Any],
    *,
    project_root: Path,
    verify_only: bool,
) -> dict[str, Any]:
    url = entry.get("url")
    commit = entry.get("commit")
    tree = entry.get("tree")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise MaterializeError(f"{key}.url must be an HTTPS Git URL")
    if not isinstance(commit, str) or len(commit) != 40:
        raise MaterializeError(f"{key}.commit must be a 40-character object id")
    if not isinstance(tree, str) or len(tree) != 40:
        raise MaterializeError(f"{key}.tree must be a 40-character object id")
    checkout = safe_project_path(project_root, entry.get("checkout"), f"{key}.checkout")

    if not (checkout / ".git").exists():
        if verify_only:
            raise MaterializeError(f"{key} checkout is missing: {checkout}")
        if checkout.exists():
            raise MaterializeError(f"{key} checkout path exists but is not Git: {checkout}")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        require_ok(
            run(
                "git",
                "clone",
                "--depth",
                "1",
                "--no-recurse-submodules",
                url,
                str(checkout),
            ),
            f"clone {key}",
        )

    origin = require_ok(
        run("git", "remote", "get-url", "origin", cwd=checkout),
        f"read {key} origin",
    )
    if normalize_url(origin) != normalize_url(url):
        raise MaterializeError(f"{key} origin mismatch: expected {url}, found {origin}")

    head_result = run("git", "rev-parse", "HEAD", cwd=checkout)
    head = head_result.stdout.strip() if head_result.returncode == 0 else ""
    if head != commit:
        if verify_only:
            raise MaterializeError(f"{key} commit mismatch: expected {commit}, found {head}")
        if status_paths(checkout, f"read {key} status"):
            raise MaterializeError(
                f"{key} is at the wrong commit and has local changes; refusing to switch"
            )
        require_ok(
            run("git", "fetch", "--depth", "1", "origin", commit, cwd=checkout),
            f"fetch {key} commit",
        )
        require_ok(
            run("git", "switch", "--detach", commit, cwd=checkout),
            f"checkout {key} commit",
        )
    if not verify_only:
        require_ok(
            run("git", "config", "--local", "remote.origin.pushurl", "DISABLED", cwd=checkout),
            f"disable {key} push URL",
        )
    push_url = require_ok(
        run("git", "remote", "get-url", "--push", "origin", cwd=checkout),
        f"read {key} push URL",
    )
    if push_url != "DISABLED":
        raise MaterializeError(f"{key} push URL is not disabled: {push_url}")

    actual_tree = require_ok(
        run("git", "rev-parse", "HEAD^{tree}", cwd=checkout),
        f"read {key} tree",
    )
    if actual_tree != tree:
        raise MaterializeError(f"{key} tree mismatch: expected {tree}, found {actual_tree}")

    raw_patches = entry.get("patches", [])
    if not isinstance(raw_patches, list):
        raise MaterializeError(f"{key}.patches must be a list")
    allowed_paths: set[str] = set()
    applied_patches: list[dict[str, str]] = []
    for index, patch_entry in enumerate(raw_patches, start=1):
        if not isinstance(patch_entry, dict):
            raise MaterializeError(f"{key}.patches[{index}] must be a mapping")
        patch = safe_project_path(
            project_root,
            patch_entry.get("path"),
            f"{key}.patches[{index}].path",
        )
        expected_hash = patch_entry.get("sha256")
        if not patch.is_file() or not isinstance(expected_hash, str):
            raise MaterializeError(f"{key} patch {index} is missing or has no sha256")
        actual_hash = sha256(patch)
        if actual_hash != expected_hash:
            raise MaterializeError(
                f"{key} patch hash mismatch for {patch.name}: "
                f"expected {expected_hash}, found {actual_hash}"
            )
        allowed_paths.update(patch_paths(checkout, patch))
        reversed_check = run("git", "apply", "--reverse", "--check", str(patch), cwd=checkout)
        if reversed_check.returncode != 0:
            forward_check = run("git", "apply", "--check", str(patch), cwd=checkout)
            if verify_only or forward_check.returncode != 0:
                detail = reversed_check.stderr.strip() or forward_check.stderr.strip()
                raise MaterializeError(f"{key} patch state invalid for {patch.name}: {detail}")
            require_ok(
                run("git", "apply", str(patch), cwd=checkout),
                f"apply {key} patch {patch.name}",
            )
        applied_patches.append({"path": patch.name, "sha256": actual_hash})

    changed = status_paths(checkout, f"read {key} final status")
    unexpected = sorted(changed - allowed_paths)
    if unexpected:
        raise MaterializeError(f"{key} has unexpected changes: {path_summary(unexpected)}")
    ignored = sorted(ignored_paths(checkout, f"read {key} ignored files"))
    if ignored:
        raise MaterializeError(f"{key} contains ignored artifacts: {path_summary(ignored)}")
    patched_files = entry.get("patched_files", {})
    if not isinstance(patched_files, dict):
        raise MaterializeError(f"{key}.patched_files must be a mapping")
    declared_final_paths = {value for value in patched_files if isinstance(value, str)}
    if len(declared_final_paths) != len(patched_files) or declared_final_paths != allowed_paths:
        raise MaterializeError(
            f"{key}.patched_files must cover exactly: {path_summary(sorted(allowed_paths))}"
        )
    verified_files: dict[str, str] = {}
    for relative, expected_hash in patched_files.items():
        if (
            not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        ):
            raise MaterializeError(f"{key}.patched_files[{relative!r}] must be a lowercase SHA-256")
        path = safe_relative_path(checkout, relative, f"{key}.patched_files[{relative!r}]")
        if not path.is_file():
            raise MaterializeError(f"{key} patched file is missing: {relative}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise MaterializeError(
                f"{key} patched file hash mismatch for {relative}: "
                f"expected {expected_hash}, found {actual_hash}"
            )
        verified_files[relative] = actual_hash
    return {
        "key": key,
        "checkout": str(checkout),
        "commit": commit,
        "tree": tree,
        "patches": applied_patches,
        "patched_files": verified_files,
        "verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lock", type=Path)
    parser.add_argument("--mapping", default="repositories")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    try:
        entries = load_entries(args.lock.resolve(), args.mapping)
        for key, entry in entries.items():
            result = ensure_checkout(
                key,
                entry,
                project_root=project_root,
                verify_only=args.verify_only,
            )
            print(
                f"PASS {key}: {result['commit']} tree={result['tree']} "
                f"patches={len(result['patches'])}"
            )
    except MaterializeError as error:
        print(f"materialize: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
