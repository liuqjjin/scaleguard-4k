from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
import yaml

from scaleguard.errors import UpstreamVerificationError
from scaleguard.strict_yaml import StrictYAMLError
from scaleguard.strict_yaml import loads as load_strict_yaml
from scaleguard.upstream import _run_git, load_upstream_lock, verify_upstreams


def run_git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_yaml(path: Path, data: object) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
    return path


def test_upstream_git_does_not_pass_ambient_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "upstream-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")
    observed_environment: dict[str, str] = {}

    def run(
        args: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        observed_environment.update(environment)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scaleguard.upstream.subprocess.run", run)

    result = _run_git(tmp_path, "status")

    assert result.returncode == 0
    assert "OPENAI_API_KEY" not in observed_environment
    assert "GITHUB_TOKEN" not in observed_environment


def make_git_repository(path: Path) -> tuple[str, str]:
    path.mkdir()
    run_git(path, "init", "-q")
    run_git(path, "config", "user.name", "ScaleGuard Tests")
    run_git(path, "config", "user.email", "tests@example.invalid")
    (path / "base.txt").write_text("base\n", encoding="utf-8")
    run_git(path, "add", "base.txt")
    run_git(path, "commit", "-qm", "base")
    return run_git(path, "rev-parse", "HEAD"), run_git(path, "rev-parse", "HEAD^{tree}")


def test_load_upstream_lock_requires_a_repository_mapping(tmp_path: Path) -> None:
    valid = write_yaml(
        tmp_path / "valid.yaml",
        {"schema_version": 1, "repositories": {}},
    )
    assert load_upstream_lock(valid) == {"schema_version": 1, "repositories": {}}

    for data in ([], {}, {"schema_version": 1, "repositories": []}):
        path = write_yaml(tmp_path / f"invalid-{len(list(tmp_path.iterdir()))}.yaml", data)
        with pytest.raises(UpstreamVerificationError, match="repositories mapping"):
            load_upstream_lock(path)


def test_load_upstream_lock_accepts_an_explicit_dependency_mapping(tmp_path: Path) -> None:
    lock = write_yaml(
        tmp_path / "runtime.yaml",
        {"schema_version": 1, "dependencies": {"depictqa": {}}},
    )

    assert load_upstream_lock(lock, "dependencies") == {
        "schema_version": 1,
        "dependencies": {"depictqa": {}},
    }


def test_load_upstream_lock_wraps_missing_and_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(UpstreamVerificationError, match="cannot load upstream lock"):
        load_upstream_lock(tmp_path / "missing.yaml")

    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("repositories: [unterminated\n", encoding="utf-8")
    with pytest.raises(UpstreamVerificationError, match="cannot load upstream lock"):
        load_upstream_lock(malformed)


@pytest.mark.parametrize(
    "document",
    [
        """
schema_version: 1
repositories:
  core: {commit: trusted}
  core: {commit: forged}
""",
        """
schema_version: 1
repositories:
  core:
    commit: trusted
    commit: forged
""",
    ],
)
def test_load_upstream_lock_rejects_duplicate_repository_identity(
    tmp_path: Path,
    document: str,
) -> None:
    lock = tmp_path / "upstream-lock.yaml"
    lock.write_text(document, encoding="utf-8")

    with pytest.raises(UpstreamVerificationError, match="duplicate mapping key"):
        load_upstream_lock(lock)


def test_strict_yaml_rejects_unhashable_mapping_keys() -> None:
    with pytest.raises(StrictYAMLError, match="unhashable mapping key"):
        load_strict_yaml("? [ambiguous, key]\n: value\n")


def test_verify_upstreams_reports_invalid_entries_and_missing_checkouts(tmp_path: Path) -> None:
    lock = write_yaml(
        tmp_path / "upstream-lock.yaml",
        {
            "schema_version": 1,
            "repositories": {
                "invalid": "not-a-mapping",
                "missing": {
                    "checkout": "checkouts/missing",
                    "commit": "abc",
                    "tree": "def",
                },
            },
        },
    )

    results = verify_upstreams(lock, tmp_path)

    assert [(item.target, item.check, item.ok) for item in results] == [
        ("invalid", "schema", False),
        ("missing", "checkout", False),
    ]


def test_verify_upstreams_accepts_a_locked_clean_repository_and_clean_patch(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "checkouts" / "4KAgent"
    repository.parent.mkdir()
    commit, tree = make_git_repository(repository)
    patch = tmp_path / "change.patch"
    patch.write_text(
        """diff --git a/base.txt b/base.txt
index df967b9..180cf83 100644
--- a/base.txt
+++ b/base.txt
@@ -1 +1 @@
-base
+patched
""",
        encoding="utf-8",
    )
    patch_sha = hashlib.sha256(patch.read_bytes()).hexdigest()
    base_sha = hashlib.sha256((repository / "base.txt").read_bytes()).hexdigest()
    lock = write_yaml(
        tmp_path / "upstream-lock.yaml",
        {
            "schema_version": 1,
            "repositories": {
                "4kagent": {
                    "checkout": "checkouts/4KAgent",
                    "commit": commit,
                    "tree": tree,
                    "patches": [
                        {
                            "path": "change.patch",
                            "sha256": patch_sha,
                            "applied": False,
                        }
                    ],
                    "patched_files": {"base.txt": base_sha},
                }
            },
        },
    )

    results = verify_upstreams(lock, tmp_path)

    assert results
    assert all(item.ok for item in results)
    assert {item.check for item in results} == {
        "commit",
        "tree",
        "patch_sha256",
        "patch_clean",
        "patch_paths",
        "patched_file_coverage",
        "patched_file_sha256",
        "worktree",
        "ignored_artifacts",
    }


def test_verify_upstreams_reports_patch_schema_files_hashes_and_dirty_worktree(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    commit, tree = make_git_repository(repository)
    existing_patch = tmp_path / "wrong.patch"
    existing_patch.write_text("not a patch\n", encoding="utf-8")
    (repository / "unexpected.txt").write_text("dirty\n", encoding="utf-8")
    lock = write_yaml(
        tmp_path / "upstream-lock.yaml",
        {
            "schema_version": 1,
            "repositories": {
                "coz": {
                    "checkout": "repo",
                    "commit": commit,
                    "tree": tree,
                    "patches": [
                        "invalid",
                        {"path": "missing.patch", "sha256": "0" * 64},
                        {"path": "wrong.patch", "sha256": "0" * 64, "applied": True},
                    ],
                }
            },
        },
    )

    results = verify_upstreams(lock, tmp_path)
    by_check = {}
    for result in results:
        by_check.setdefault(result.check, []).append(result)

    assert by_check["patch"][0].ok is False
    assert by_check["patch_file"][0].ok is False
    assert by_check["patch_sha256"][0].ok is False
    assert by_check["patch_applied"][0].ok is False
    assert by_check["worktree"][0].ok is False
    assert "unexpected.txt" in by_check["worktree"][0].detail


def test_verify_upstreams_rejects_ignored_executable_artifacts(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    commit, tree = make_git_repository(repository)
    (repository / ".git" / "info" / "exclude").write_text(
        "__pycache__/\n",
        encoding="utf-8",
    )
    ignored = repository / "package" / "__pycache__" / "module.cpython-310.pyc"
    ignored.parent.mkdir(parents=True)
    ignored.write_bytes(b"untrusted bytecode")
    lock = write_yaml(
        tmp_path / "upstream-lock.yaml",
        {
            "schema_version": 1,
            "repositories": {
                "upstream": {
                    "checkout": "repo",
                    "commit": commit,
                    "tree": tree,
                }
            },
        },
    )

    results = verify_upstreams(lock, tmp_path)
    check = next(item for item in results if item.check == "ignored_artifacts")

    assert check.ok is False
    assert "__pycache__/module.cpython-310.pyc" in check.detail


def test_verify_upstreams_rejects_extra_edits_inside_a_patch_modified_file(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    commit, tree = make_git_repository(repository)
    patch = tmp_path / "change.patch"
    patch.write_text(
        """diff --git a/base.txt b/base.txt
index df967b9..180cf83 100644
--- a/base.txt
+++ b/base.txt
@@ -1 +1 @@
-base
+patched
""",
        encoding="utf-8",
    )
    run_git(repository, "apply", str(patch))
    expected_hash = hashlib.sha256((repository / "base.txt").read_bytes()).hexdigest()
    with (repository / "base.txt").open("a", encoding="utf-8") as handle:
        handle.write("unlocked edit\n")
    lock = write_yaml(
        tmp_path / "upstream-lock.yaml",
        {
            "schema_version": 1,
            "repositories": {
                "coz": {
                    "checkout": "repo",
                    "commit": commit,
                    "tree": tree,
                    "patches": [
                        {
                            "path": "change.patch",
                            "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                        }
                    ],
                    "patched_files": {"base.txt": expected_hash},
                }
            },
        },
    )

    results = verify_upstreams(lock, tmp_path)
    final_hash = next(result for result in results if result.check == "patched_file_sha256")

    assert final_hash.ok is False
    assert expected_hash in final_hash.detail


def test_verify_upstreams_reports_unquoted_paths_with_spaces(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    commit, tree = make_git_repository(repository)
    (repository / "unexpected file.txt").write_text("dirty\n", encoding="utf-8")
    lock = write_yaml(
        tmp_path / "upstream-lock.yaml",
        {
            "schema_version": 1,
            "repositories": {
                "coz": {
                    "checkout": "repo",
                    "commit": commit,
                    "tree": tree,
                }
            },
        },
    )

    results = verify_upstreams(lock, tmp_path)
    worktree = next(result for result in results if result.check == "worktree")

    assert worktree.ok is False
    assert "unexpected file.txt" in worktree.detail
    assert '"unexpected file.txt"' not in worktree.detail


def test_verify_upstreams_rejects_a_non_list_patch_field(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    commit, tree = make_git_repository(repository)
    lock = write_yaml(
        tmp_path / "upstream-lock.yaml",
        {
            "schema_version": 1,
            "repositories": {
                "coz": {
                    "checkout": "repo",
                    "commit": commit,
                    "tree": tree,
                    "patches": "change.patch",
                }
            },
        },
    )

    results = verify_upstreams(lock, tmp_path)

    assert results[-1].check == "patches"
    assert results[-1].ok is False
