from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_materializer() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts/upstream/materialize.py"
    spec = importlib.util.spec_from_file_location("scaleguard_upstream_materializer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load upstream materializer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository(path: Path) -> tuple[str, str]:
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "ScaleGuard Tests")
    _git(path, "config", "user.email", "tests@example.invalid")
    (path / "source file.txt").write_text(
        "locked\n" + "".join(f"stable-{index}\n" for index in range(1, 9)),
        encoding="utf-8",
    )
    _git(path, "add", "source file.txt")
    _git(path, "commit", "-qm", "locked")
    return _git(path, "rev-parse", "HEAD"), _git(path, "rev-parse", "HEAD^{tree}")


@pytest.mark.parametrize(
    "document",
    [
        """
schema_version: 1
dependencies:
  depictqa: {commit: trusted}
  depictqa: {commit: forged}
""",
        """
schema_version: 1
dependencies:
  depictqa:
    patches:
      - path: overlay.patch
        sha256: trusted
        sha256: forged
""",
    ],
)
def test_materializer_rejects_duplicate_dependency_identity(
    tmp_path: Path,
    document: str,
) -> None:
    materializer = _load_materializer()
    lock = tmp_path / "runtime-dependencies.yaml"
    lock.write_text(document, encoding="utf-8")

    with pytest.raises(materializer.MaterializeError, match="duplicate mapping key"):
        materializer.load_entries(lock, "dependencies")


def test_materializer_populates_a_clone_when_remote_head_is_the_locked_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load_materializer()
    origin = tmp_path / "origin"
    commit, tree = _repository(origin)
    source = origin / "source file.txt"
    locked_content = source.read_text(encoding="utf-8")
    patched_content = locked_content.replace("locked\n", "patched\n", 1)
    source.write_text(patched_content, encoding="utf-8")
    patch = tmp_path / "change.patch"
    patch.write_text(_git(origin, "diff", "--binary", "--", source.name) + "\n", encoding="utf-8")
    source.write_text(locked_content, encoding="utf-8")
    patch_sha = hashlib.sha256(patch.read_bytes()).hexdigest()
    patched_sha = hashlib.sha256(patched_content.encode()).hexdigest()
    expected_url = "https://example.invalid/upstream.git"
    real_run = materializer.run

    def intercepted_run(
        *argv: str,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if argv[:2] == ("git", "clone"):
            destination = Path(argv[-1])
            result = real_run(
                "git",
                "clone",
                "--depth",
                "1",
                "--no-recurse-submodules",
                str(origin),
                str(destination),
            )
            if result.returncode == 0:
                real_run(
                    "git",
                    "remote",
                    "set-url",
                    "origin",
                    expected_url,
                    cwd=destination,
                )
            return result
        return real_run(*argv, cwd=cwd)

    monkeypatch.setattr(materializer, "run", intercepted_run)
    entry: dict[str, Any] = {
        "url": expected_url,
        "commit": commit,
        "tree": tree,
        "checkout": "checkouts/upstream",
        "patches": [{"path": "change.patch", "sha256": patch_sha}],
        "patched_files": {"source file.txt": patched_sha},
    }

    result = materializer.ensure_checkout(
        "upstream",
        entry,
        project_root=tmp_path,
        verify_only=False,
    )
    checkout = tmp_path / "checkouts/upstream"

    assert result["verified"] is True
    assert (checkout / "source file.txt").read_text(encoding="utf-8") == patched_content
    assert _git(checkout, "remote", "get-url", "--push", "origin") == "DISABLED"
    materializer.ensure_checkout(
        "upstream",
        entry,
        project_root=tmp_path,
        verify_only=True,
    )
    checkout_source = checkout / "source file.txt"
    checkout_source.write_text(
        checkout_source.read_text(encoding="utf-8").replace(
            "stable-8\n",
            "unlocked edit\n",
        ),
        encoding="utf-8",
    )
    with pytest.raises(materializer.MaterializeError, match="patched file hash mismatch"):
        materializer.ensure_checkout(
            "upstream",
            entry,
            project_root=tmp_path,
            verify_only=True,
        )
