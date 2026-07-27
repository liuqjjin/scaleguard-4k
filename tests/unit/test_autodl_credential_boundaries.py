from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
COMMON = ROOT / "scripts" / "autodl" / "_common.sh"


def _copy_common(project: Path) -> Path:
    destination = project / "scripts" / "autodl" / "_common.sh"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(COMMON, destination)
    return destination


def _run_bash(
    script: str,
    *,
    env: dict[str, str] | None = None,
    cwd: Path = ROOT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", "-uec", script],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_common_helpers_limit_each_credential_phase() -> None:
    script = f"""
export HF_TOKEN=hf-phase-only
export OPENAI_API_KEY=openai-must-not-leak
export GITHUB_TOKEN=github-must-not-leak
export CUSTOM_SCHEDULER_CREDENTIAL=custom-runtime-only
export UNRELATED_SERVICE_TOKEN=unknown-must-not-leak
export PIP_INDEX_URL=https://user:pass@packages.example.invalid/simple
export BASH_ENV=/tmp/poison-bash-env
export PYTHONPATH=/tmp/poison-pythonpath
source {str(COMMON)!r}
sg_register_sensitive_env_name CUSTOM_SCHEDULER_CREDENTIAL
sg_run_sanitized python3 -c 'import json,os; print(json.dumps(dict(os.environ)))'
sg_run_with_download_credentials \
    python3 -c 'import json,os; print(json.dumps(dict(os.environ)))'
sg_run_with_scheduler_credential CUSTOM_SCHEDULER_CREDENTIAL \
    python3 -c 'import json,os; print(json.dumps(dict(os.environ)))'
"""
    result = _run_bash(script)
    sanitized, download, runtime = [json.loads(line) for line in result.stdout.splitlines()]

    for name in (
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "GITHUB_TOKEN",
        "CUSTOM_SCHEDULER_CREDENTIAL",
        "UNRELATED_SERVICE_TOKEN",
        "PIP_INDEX_URL",
        "BASH_ENV",
        "PYTHONPATH",
    ):
        assert name not in sanitized
    assert download["HF_TOKEN"] == "hf-phase-only"
    assert "OPENAI_API_KEY" not in download
    assert "GITHUB_TOKEN" not in download
    assert "CUSTOM_SCHEDULER_CREDENTIAL" not in download
    assert "UNRELATED_SERVICE_TOKEN" not in download
    assert "PIP_INDEX_URL" not in download
    assert "BASH_ENV" not in download
    assert "PYTHONPATH" not in download
    assert runtime["CUSTOM_SCHEDULER_CREDENTIAL"] == "custom-runtime-only"
    assert "HF_TOKEN" not in runtime
    assert "OPENAI_API_KEY" not in runtime
    assert "GITHUB_TOKEN" not in runtime
    assert "UNRELATED_SERVICE_TOKEN" not in runtime
    assert "PIP_INDEX_URL" not in runtime
    assert "BASH_ENV" not in runtime
    assert "PYTHONPATH" not in runtime


def test_doctor_gets_marker_while_model_gets_real_scheduler_secret(
    tmp_path: Path,
) -> None:
    log = tmp_path / "scoped.log"
    secret = "real-scheduler-secret"
    secret_digest = hashlib.sha256(secret.encode()).hexdigest()
    script = f"""
source {str(COMMON)!r}
export HF_TOKEN=download-secret
export GITHUB_TOKEN=github-secret
export CUSTOM_SCHEDULER_CREDENTIAL={secret}
sg_register_sensitive_env_name CUSTOM_SCHEDULER_CREDENTIAL
sg_make_sensitive_environment_private
sg_run_logged_with_presence_marker {str(log)!r} CUSTOM_SCHEDULER_CREDENTIAL \
    python3 -c "import os; assert os.environ['CUSTOM_SCHEDULER_CREDENTIAL'] == \
'SCALEGUARD_DOCTOR_CREDENTIAL_PRESENT'; assert 'HF_TOKEN' not in os.environ; \
assert 'GITHUB_TOKEN' not in os.environ"
sg_run_logged_with_private_credentials {str(log)!r} \
    CUSTOM_SCHEDULER_CREDENTIAL python3 -c "import hashlib,os; \
assert hashlib.sha256(os.environ['CUSTOM_SCHEDULER_CREDENTIAL'].encode()).hexdigest() \
== '{secret_digest}'; \
assert 'HF_TOKEN' not in os.environ; assert 'GITHUB_TOKEN' not in os.environ"
"""

    _run_bash(script)
    logged = log.read_text(encoding="utf-8")
    assert secret not in logged


def test_bootstrap_defers_the_sanitized_doctor_credential_check() -> None:
    bootstrap = (ROOT / "scripts" / "autodl" / "bootstrap.sh").read_text(encoding="utf-8")
    assert 'deferred_credential_checks = {"4kagent_api_key"}' in bootstrap
    assert 'deferred_credential_checks = {"scheduler_api_key"}' not in bootstrap


def test_autodl_project_helpers_use_the_installed_project_interpreter(
    tmp_path: Path,
) -> None:
    bootstrap = (ROOT / "scripts" / "autodl" / "bootstrap.sh").read_text(encoding="utf-8")
    download = (ROOT / "scripts" / "autodl" / "download_weights.sh").read_text(encoding="utf-8")
    environment_bootstrap = (ROOT / "scripts" / "bootstrap" / "autodl.sh").read_text(
        encoding="utf-8"
    )
    common = COMMON.read_text(encoding="utf-8")

    assert (
        '"${SG_REPO_ROOT}/.venv/bin/python" -I "${sg_here}/_validate_bootstrap_receipt.py"'
    ) in bootstrap
    assert 'sg_project_python="${SG_REPO_ROOT}/.venv/bin/python"' in download
    assert '"${sg_project_python}" -I "${sg_here}/_download_weights.py"' in download
    assert '"${sg_project_python}" -I "${sg_materializer}"' in download
    assert '"${sg_project_python}" -I "${sg_materializer}"' in common
    assert environment_bootstrap.count('-I "${sg_audit_script}"') == 4
    assert common.count('-I "${sg_audit_script}"') == 4

    clean_environment = {
        "HOME": os.environ["HOME"],
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONNOUSERSITE": "1",
    }
    project_python = ROOT / ".venv" / "bin" / "python"
    for helper in (
        ROOT / "scripts" / "autodl" / "_validate_bootstrap_receipt.py",
        ROOT / "scripts" / "autodl" / "_download_weights.py",
        ROOT / "scripts" / "autodl" / "_write_preflight_receipt.py",
        ROOT / "scripts" / "weights" / "materialize.py",
    ):
        result = subprocess.run(
            [str(project_python), "-I", str(helper), "--help"],
            cwd=tmp_path,
            env=clean_environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{helper}: {result.stderr}"


def test_autodl_cli_resolution_is_bound_to_the_project_venv(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "fallback-invoked.txt"
    for name in ("python3", "scaleguard"):
        executable = fake_bin / name
        executable.write_text(
            f"#!/bin/sh\nprintf '%s\\n' {name!r} >> {shlex.quote(str(marker))}\nexit 97\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)

    environment = os.environ.copy()
    environment.pop("SCALEGUARD_CLI", None)
    environment.pop("SCALEGUARD_PYTHON", None)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    result = _run_bash(
        f"""
source {str(COMMON)!r}
sg_resolve_cli
printf '<%s>\\n' "${{SG_CLI[@]}}"
""",
        env=environment,
    )

    assert result.stdout.splitlines() == [
        f"<{ROOT / '.venv' / 'bin' / 'python'}>",
        "<-I>",
        "<-m>",
        "<scaleguard.cli>",
    ]
    assert not marker.exists()

    for variable, executable in (
        ("SCALEGUARD_CLI", fake_bin / "scaleguard"),
        ("SCALEGUARD_PYTHON", fake_bin / "python3"),
    ):
        overridden_environment = environment.copy()
        overridden_environment[variable] = str(executable)
        overridden = subprocess.run(
            [
                "/bin/bash",
                "-uec",
                f"source {str(COMMON)!r}; sg_resolve_cli",
            ],
            cwd=ROOT,
            env=overridden_environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert overridden.returncode != 0
        assert f"{variable} is not allowed for AutoDL evidence" in overridden.stderr
        assert not marker.exists()


def test_autodl_cli_resolution_attests_prefix_and_module_source(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project_common = _copy_common(project)
    package = project / "src" / "scaleguard"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text("def main():\n    return 0\n", encoding="utf-8")

    venv = project / ".venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)],
        check=True,
        capture_output=True,
        text=True,
    )
    project_python = venv / "bin" / "python"
    site_packages = Path(
        subprocess.run(
            [
                str(project_python),
                "-I",
                "-c",
                "import site; print(site.getsitepackages()[0])",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    source_binding = site_packages / "scaleguard-source.pth"
    source_binding.write_text(f"{project / 'src'}\n", encoding="utf-8")

    environment = os.environ.copy()
    environment.pop("SCALEGUARD_CLI", None)
    environment.pop("SCALEGUARD_PYTHON", None)
    valid = _run_bash(
        f"""
source {str(project_common)!r}
sg_resolve_cli
printf '%s\\n' "${{SG_CLI[@]}}"
""",
        env=environment,
    )
    assert valid.stdout.splitlines() == [
        str(project_python),
        "-I",
        "-m",
        "scaleguard.cli",
    ]

    foreign_source = tmp_path / "foreign-source"
    foreign_package = foreign_source / "scaleguard"
    foreign_package.mkdir(parents=True)
    (foreign_package / "__init__.py").write_text("", encoding="utf-8")
    (foreign_package / "cli.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    source_binding.write_text(f"{foreign_source}\n", encoding="utf-8")
    foreign_module = subprocess.run(
        ["/bin/bash", "-uec", f"source {str(project_common)!r}; sg_resolve_cli"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert foreign_module.returncode != 0
    assert "ScaleGuard CLI module mismatch" in foreign_module.stderr
    assert "project ScaleGuard CLI attestation failed" in foreign_module.stderr

    foreign_project = tmp_path / "foreign-project"
    foreign_common = _copy_common(foreign_project)
    foreign_expected_package = foreign_project / "src" / "scaleguard"
    foreign_expected_package.mkdir(parents=True)
    (foreign_expected_package / "__init__.py").write_text("", encoding="utf-8")
    (foreign_expected_package / "cli.py").write_text(
        "def main():\n    return 0\n",
        encoding="utf-8",
    )
    foreign_launcher = foreign_project / ".venv" / "bin" / "python"
    foreign_launcher.parent.mkdir(parents=True)
    foreign_launcher.write_text(
        f'#!/bin/sh\nexec {shlex.quote(str(project_python))} "$@"\n',
        encoding="utf-8",
    )
    foreign_launcher.chmod(0o755)
    foreign_prefix = subprocess.run(
        ["/bin/bash", "-uec", f"source {str(foreign_common)!r}; sg_resolve_cli"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert foreign_prefix.returncode != 0
    assert "ScaleGuard interpreter mismatch" in foreign_prefix.stderr
    assert "project ScaleGuard CLI attestation failed" in foreign_prefix.stderr


def test_common_rejects_an_ambient_alternate_repository_root(tmp_path: Path) -> None:
    attacker = tmp_path / "attacker"
    fake_python = attacker / ".venv" / "bin" / "python"
    marker = tmp_path / "executed"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text(
        f"#!/bin/sh\nprintf owned > {shlex.quote(str(marker))}\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment["SCALEGUARD_REPO_ROOT"] = str(attacker)

    result = subprocess.run(
        ["/bin/bash", "-uec", f"source {str(COMMON)!r}; sg_resolve_cli"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "must match the repository containing _common.sh" in result.stderr
    assert not marker.exists()


def test_common_rejects_a_symlinked_runtime_root_before_git_can_run(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project_common = _copy_common(project)
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    (project / ".runtime").symlink_to(attacker, target_is_directory=True)

    result = subprocess.run(
        ["/bin/bash", "-uec", f"source {str(project_common)!r}; git status"],
        cwd=project,
        env=os.environ.copy(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "runtime root must not be a symbolic link" in result.stderr


def test_common_uses_a_fresh_private_home_and_disables_git_configuration(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project_common = _copy_common(project)
    ambient_home = tmp_path / "ambient-home"
    ambient_home.mkdir()
    (ambient_home / ".gitconfig").write_text(
        "[core]\n\tfsmonitor = /tmp/attacker-hook\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["HOME"] = str(ambient_home)

    result = _run_bash(
        f"""
source {str(project_common)!r}
printf '%s\\n' "${{HOME}}"
printf '<%s>\\n' "$(git config --global --get core.fsmonitor || :)"
""",
        env=environment,
        cwd=project,
    )

    isolated_home = Path(result.stdout.splitlines()[0])
    assert isolated_home.parent == project / ".runtime" / "isolated-homes"
    assert isolated_home != ambient_home
    assert not isolated_home.is_symlink()
    assert isolated_home.stat().st_uid == os.geteuid()
    assert isolated_home.stat().st_mode & 0o777 == 0o700
    assert list(isolated_home.iterdir()) == []
    assert result.stdout.splitlines()[1] == "<>"


def test_diagnostics_sanitizer_bootstraps_project_source_for_system_python(
    tmp_path: Path,
) -> None:
    system_python = Path("/usr/bin/python3")
    if not system_python.is_file():
        pytest.skip("the declared AutoDL system Python path is unavailable")

    result = subprocess.run(
        [
            str(system_python),
            str(ROOT / "scripts" / "autodl" / "_sanitize_diagnostics.py"),
            "--help",
        ],
        cwd=tmp_path,
        env={
            "HOME": os.environ["HOME"],
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONNOUSERSITE": "1",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--secret-fd" in result.stdout


@pytest.mark.parametrize(
    "body",
    [
        "fourkagent:\n  api_key_env: FIRST_KEY\n  api_key_env: SECOND_KEY\n",
        '"fourkagent":\n  "api_key_env": CUSTOM_SCHEDULER_CREDENTIAL\n',
        '"\\x66ourkagent":\n  "\\x61pi_key_env": ESCAPED_SCHEDULER_CREDENTIAL\n',
        "{fourkagent: {api_key_env: FLOW_SCHEDULER_CREDENTIAL}}\n",
        "fourkagent:\n  api_key_env: BASH_ENV\n",
    ],
)
def test_shell_scheduler_resolver_rejects_ambiguous_yaml(
    tmp_path: Path,
    body: str,
) -> None:
    config = tmp_path / "ambiguous.yaml"
    config.write_text(body, encoding="utf-8")
    result = subprocess.run(
        [
            "/bin/bash",
            "-uec",
            f"source {str(COMMON)!r}; sg_resolve_scheduler_api_key_env {str(config)!r}",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "AutoDL config" in result.stderr


def test_external_gate_scopes_credentials_by_phase(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "external_gate").mkdir(parents=True)
    (project / "scripts" / "autodl").mkdir(parents=True)
    shutil.copy2(ROOT / "external_gate" / "commands.sh", project / "external_gate/commands.sh")
    shutil.copy2(COMMON, project / "scripts/autodl/_common.sh")
    (project / "weights-lock.json").write_text("{}\n", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    phase_log = tmp_path / "phases.jsonl"
    env_probe = tmp_path / "env-invoked.txt"
    fake_env = fake_bin / "env"
    fake_env.write_text(
        """#!/bin/sh
printf 'PATH-resolved env was invoked\\n' > "$ENV_PROBE"
exit 97
""",
        encoding="utf-8",
    )
    fake_env.chmod(0o755)
    for name in (
        "check_gpu.sh",
        "bootstrap.sh",
        "download_weights.sh",
        "run_smoke.sh",
        "run_integration.sh",
        "collect_diagnostics.sh",
    ):
        probe = project / "scripts" / "autodl" / name
        probe.write_text(
            f"""#!/bin/sh
{shlex.quote(sys.executable)} -I - {shlex.quote("scripts/autodl/" + name)} \
{shlex.quote(str(phase_log))} <<'PY'
"""
            """\
import json
import os
import pathlib
import sys

record = {
    "script": sys.argv[1],
    "hf": os.environ.get("HF_TOKEN"),
    "openai": os.environ.get("OPENAI_API_KEY"),
    "github": os.environ.get("GITHUB_TOKEN"),
    "runtime_scheduler": os.environ.get("RUNTIME_SCHEDULER_CREDENTIAL"),
    "smoke_scheduler": os.environ.get("SMOKE_SCHEDULER_CREDENTIAL"),
    "integration_scheduler": os.environ.get("INTEGRATION_SCHEDULER_CREDENTIAL"),
}
with pathlib.Path(sys.argv[2]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\\n")
PY
""",
            encoding="utf-8",
        )
        probe.chmod(0o755)
    runtime_config = tmp_path / "runtime.yaml"
    runtime_config.write_text(
        "fourkagent:\n  api_key_env: RUNTIME_SCHEDULER_CREDENTIAL\n",
        encoding="utf-8",
    )
    smoke_config = tmp_path / "smoke.yaml"
    smoke_config.write_text(
        "fourkagent:\n  api_key_env: SMOKE_SCHEDULER_CREDENTIAL\n",
        encoding="utf-8",
    )
    integration_config = tmp_path / "integration.yaml"
    integration_config.write_text(
        "fourkagent:\n  api_key_env: INTEGRATION_SCHEDULER_CREDENTIAL\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    artifact_root = cache / "artifacts"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "ENV_PROBE": str(env_probe),
            "SCALEGUARD_AUTODL_CONFIG": str(runtime_config),
            "SCALEGUARD_SMOKE_CONFIG": str(smoke_config),
            "SCALEGUARD_INTEGRATION_CONFIG": str(integration_config),
            "SCALEGUARD_CACHE_ROOT": str(cache),
            "SCALEGUARD_ARTIFACT_ROOT": str(artifact_root),
            "SCALEGUARD_RUN_ID": "external-phase-test",
            "SCALEGUARD_WEIGHTS_MANIFEST": str(project / "weights-lock.json"),
            "SCALEGUARD_SMOKE_INPUT": str(ROOT / "pyproject.toml"),
            "SCALEGUARD_INTEGRATION_INPUT": str(ROOT / "pyproject.toml"),
            "HF_TOKEN": "hf-download-only",
            "OPENAI_API_KEY": "openai-not-configured",
            "GITHUB_TOKEN": "github-never-needed",
            "RUNTIME_SCHEDULER_CREDENTIAL": "runtime-not-for-runs",
            "SMOKE_SCHEDULER_CREDENTIAL": "smoke-runtime-only",
            "INTEGRATION_SCHEDULER_CREDENTIAL": "integration-runtime-only",
        }
    )
    subprocess.run(
        [str(project / "external_gate" / "commands.sh")],
        cwd=project,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    records = [json.loads(line) for line in phase_log.read_text(encoding="utf-8").splitlines()]
    by_name = {Path(record["script"]).name: record for record in records}

    for name in ("check_gpu.sh", "bootstrap.sh"):
        assert all(
            by_name[name][field] is None
            for field in (
                "hf",
                "openai",
                "github",
                "runtime_scheduler",
                "smoke_scheduler",
                "integration_scheduler",
            )
        )
    assert by_name["download_weights.sh"] == {
        "script": "scripts/autodl/download_weights.sh",
        "hf": "hf-download-only",
        "openai": None,
        "github": None,
        "runtime_scheduler": None,
        "smoke_scheduler": None,
        "integration_scheduler": None,
    }
    assert by_name["run_smoke.sh"]["smoke_scheduler"] == "smoke-runtime-only"
    assert by_name["run_integration.sh"]["integration_scheduler"] == ("integration-runtime-only")
    assert all(
        by_name["run_smoke.sh"][field] is None
        for field in (
            "hf",
            "openai",
            "github",
            "runtime_scheduler",
            "integration_scheduler",
        )
    )
    assert all(
        by_name["run_integration.sh"][field] is None
        for field in (
            "hf",
            "openai",
            "github",
            "runtime_scheduler",
            "smoke_scheduler",
        )
    )
    assert not env_probe.exists()


def test_diagnostics_exact_redaction_uses_private_fd_and_sanitized_probes(
    tmp_path: Path,
) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        "fourkagent:\n  api_key_env: CUSTOM_SCHEDULER_CREDENTIAL\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    artifact_root = cache / "artifacts"
    source = artifact_root / "attempt"
    source.mkdir(parents=True)
    opaque_secret = "orchid-lantern-742-no-pattern"
    (source / "attempt.log").write_text(
        f"scheduler returned {opaque_secret}\n",
        encoding="utf-8",
    )
    bundle = cache / "diagnostics.tar.gz"
    env = os.environ.copy()
    env.update(
        {
            "SCALEGUARD_AUTODL_CONFIG": str(config),
            "SCALEGUARD_CACHE_ROOT": str(cache),
            "SCALEGUARD_ARTIFACT_ROOT": str(artifact_root),
            "SCALEGUARD_RUN_ID": "credential-boundary-test",
            "CUSTOM_SCHEDULER_CREDENTIAL": opaque_secret,
            "HF_TOKEN": "another-opaque-value",
        }
    )
    subprocess.run(
        [
            str(ROOT / "scripts" / "autodl" / "collect_diagnostics.sh"),
            "--source",
            str(source),
            "--output",
            str(bundle),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    with tarfile.open(bundle, "r:gz") as archive:
        content = b"\n".join(
            archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile() and archive.extractfile(member) is not None
        )
    assert opaque_secret.encode() not in content
    assert b"[REDACTED:CUSTOM_SCHEDULER_CREDENTIAL]" in content


def test_standalone_runner_privatizes_credentials_before_child_processes(
    tmp_path: Path,
) -> None:
    startup_marker = tmp_path / "bash-env-ran.txt"
    poison = tmp_path / "poison.sh"
    poison.write_text(
        f"printf 'unsafe\\n' > {shlex.quote(str(startup_marker))}\n",
        encoding="utf-8",
    )
    config = tmp_path / "runtime.yaml"
    config.write_text(
        "fourkagent:\n  api_key_env: CUSTOM_SCHEDULER_CREDENTIAL\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "BASH_ENV": str(poison),
            "SHELLOPTS": "xtrace",
            "PS4": "${HF_TOKEN}",
            "LC_VENDOR_API_TOKEN": "locale-private",
            "BASH_FUNC_source%%": "() { printf '%s\\n' \"$HF_TOKEN\" >&2; }",
            "BASH_FUNC_unset%%": "() { printf '%s\\n' \"$HF_TOKEN\" >&2; }",
            "BASH_FUNC_builtin%%": "() { printf '%s\\n' \"$HF_TOKEN\" >&2; }",
            "CUSTOM_SCHEDULER_CREDENTIAL": "runtime-private",
            "HF_TOKEN": "download-private",
            "GITHUB_TOKEN": "ambient-private",
        }
    )
    result = subprocess.run(
        [
            str(ROOT / "scripts" / "autodl" / "_run_scaleguard.sh"),
            "smoke",
            "--config",
            str(config),
            "--help",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert not startup_marker.exists()
    combined = result.stdout + result.stderr
    for secret in (
        "runtime-private",
        "download-private",
        "ambient-private",
        "locale-private",
    ):
        assert secret not in combined


def test_runtime_environment_reaudit_uses_all_isolated_pythons_without_credentials(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project_common = _copy_common(project)
    probe_log = tmp_path / "runtime-audits.txt"
    audit = project / "scripts" / "bootstrap" / "audit_environment.py"
    audit.parent.mkdir(parents=True)
    audit.write_text("# audit fixture\n", encoding="utf-8")
    locks = (
        "uv.lock",
        "environments/4kagent/requirements.resolved.lock",
        "environments/4kagent/pyiqa.override.lock",
        "environments/4kagent/hpsv2.override.lock",
        "environments/depictqa/requirements.resolved.lock",
        "environments/coz/requirements.resolved.lock",
    )
    for relative in locks:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")

    python_fixture = (
        """#!/bin/sh
if [ -n "${HF_TOKEN+x}" ] \
    || [ -n "${GITHUB_TOKEN+x}" ] \
    || [ -n "${CUSTOM_SCHEDULER_CREDENTIAL+x}" ]; then
    exit 91
fi
[ "${HF_HUB_OFFLINE:-}" = "1" ] || exit 92
[ "${TRANSFORMERS_OFFLINE:-}" = "1" ] || exit 93
"""
        f"""printf '%s|%s\\n' "$0" "$*" >> {shlex.quote(str(probe_log))}
"""
    )
    interpreters = (
        project / ".venv" / "bin" / "python",
        project / ".runtime" / "envs" / "4kagent" / "bin" / "python",
        project / ".runtime" / "envs" / "depictqa" / "bin" / "python",
        project / ".runtime" / "envs" / "coz" / "bin" / "python",
    )
    for interpreter in interpreters:
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text(python_fixture, encoding="utf-8")
        interpreter.chmod(0o755)

    output_root = tmp_path / "fresh-runtime-environments"
    log = tmp_path / "reaudit.log"
    environment = os.environ.copy()
    environment.update(
        {
            "HF_TOKEN": "download-private",
            "GITHUB_TOKEN": "ambient-private",
            "CUSTOM_SCHEDULER_CREDENTIAL": "runtime-private",
        }
    )
    script = f"""
source {str(project_common)!r}
sg_register_sensitive_env_name CUSTOM_SCHEDULER_CREDENTIAL
sg_make_sensitive_environment_private
sg_reaudit_runtime_environments {str(log)!r} {str(output_root)!r}
"""
    _run_bash(script, env=environment)

    observations = probe_log.read_text(encoding="utf-8").splitlines()
    assert len(observations) == 4
    assert {line.split("|", 1)[0] for line in observations} == {str(path) for path in interpreters}
    assert all("|-I " in line for line in observations)
    assert {line.split("--name ", 1)[1].split()[0] for line in observations} == {
        "scaleguard",
        "4kagent",
        "depictqa",
        "coz",
    }
    runner = (ROOT / "scripts" / "autodl" / "_run_scaleguard.sh").read_text(encoding="utf-8")
    assert runner.index("doctor --config") < runner.index("sg_reaudit_runtime_environments")
    assert runner.index("sg_reaudit_runtime_environments") < runner.index(
        "_write_preflight_receipt.py"
    )
    assert runner.index("_write_preflight_receipt.py") < runner.index("--runtime-preflight")

    interpreters[2].write_text("#!/bin/sh\nexit 31\n", encoding="utf-8")
    interpreters[2].chmod(0o755)
    failed_output_root = tmp_path / "failed-runtime-environments"
    failed_script = f"""
source {str(project_common)!r}
sg_register_sensitive_env_name CUSTOM_SCHEDULER_CREDENTIAL
sg_make_sensitive_environment_private
sg_reaudit_runtime_environments {str(log)!r} {str(failed_output_root)!r}
"""
    failed = subprocess.run(
        ["/bin/bash", "-uec", failed_script],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert "runtime environment re-audit failed" in failed.stderr
