from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
COMMON = ROOT / "scripts" / "autodl" / "_common.sh"


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
source {str(COMMON)!r}
export HF_TOKEN=hf-phase-only
export OPENAI_API_KEY=openai-must-not-leak
export GITHUB_TOKEN=github-must-not-leak
export CUSTOM_SCHEDULER_CREDENTIAL=custom-runtime-only
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
    ):
        assert name not in sanitized
    assert download["HF_TOKEN"] == "hf-phase-only"
    assert "OPENAI_API_KEY" not in download
    assert "GITHUB_TOKEN" not in download
    assert "CUSTOM_SCHEDULER_CREDENTIAL" not in download
    assert runtime["CUSTOM_SCHEDULER_CREDENTIAL"] == "custom-runtime-only"
    assert "HF_TOKEN" not in runtime
    assert "OPENAI_API_KEY" not in runtime
    assert "GITHUB_TOKEN" not in runtime


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
    common = COMMON.read_text(encoding="utf-8")

    assert (
        '"${SG_REPO_ROOT}/.venv/bin/python" "${sg_here}/_validate_bootstrap_receipt.py"'
    ) in bootstrap
    assert 'sg_project_python="${SG_REPO_ROOT}/.venv/bin/python"' in download
    assert '"${sg_project_python}" "${sg_here}/_download_weights.py"' in download
    assert '"${sg_project_python}" "${sg_materializer}"' in download
    assert '"${sg_project_python}" "${sg_materializer}"' in common

    clean_environment = {
        "HOME": os.environ["HOME"],
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONNOUSERSITE": "1",
    }
    project_python = ROOT / ".venv" / "bin" / "python"
    for helper in (
        ROOT / "scripts" / "autodl" / "_validate_bootstrap_receipt.py",
        ROOT / "scripts" / "autodl" / "_download_weights.py",
        ROOT / "scripts" / "weights" / "materialize.py",
    ):
        result = subprocess.run(
            [str(project_python), str(helper), "--help"],
            cwd=tmp_path,
            env=clean_environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{helper}: {result.stderr}"


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
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        """#!/bin/sh
python3 - "$1" "$BOUNDARY_LOG" <<'PY'
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
    fake_bash.chmod(0o755)
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
            "BOUNDARY_LOG": str(phase_log),
            "ENV_PROBE": str(env_probe),
            "SCALEGUARD_AUTODL_CONFIG": str(runtime_config),
            "SCALEGUARD_SMOKE_CONFIG": str(smoke_config),
            "SCALEGUARD_INTEGRATION_CONFIG": str(integration_config),
            "SCALEGUARD_CACHE_ROOT": str(cache),
            "SCALEGUARD_ARTIFACT_ROOT": str(artifact_root),
            "SCALEGUARD_RUN_ID": "external-phase-test",
            "SCALEGUARD_WEIGHTS_MANIFEST": str(ROOT / "weights-lock.json"),
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
        ["/bin/bash", str(ROOT / "external_gate" / "commands.sh")],
        cwd=ROOT,
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
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    probe_log = tmp_path / "probe.txt"
    fake_uname = fake_bin / "uname"
    fake_uname.write_text(
        """#!/bin/sh
if [ -n "${CUSTOM_SCHEDULER_CREDENTIAL+x}" ] || [ -n "${HF_TOKEN+x}" ]; then
    printf 'visible\\n' > "$PROBE_LOG"
else
    printf 'sanitized\\n' > "$PROBE_LOG"
fi
/usr/bin/uname "$@"
""",
        encoding="utf-8",
    )
    fake_uname.chmod(0o755)

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
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PROBE_LOG": str(probe_log),
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
            "/bin/bash",
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

    assert probe_log.read_text(encoding="utf-8") == "sanitized\n"
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
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    probe_log = tmp_path / "runner-probes.txt"
    for command, real_command in (("cat", "/bin/cat"), ("tr", "/usr/bin/tr")):
        wrapper = fake_bin / command
        wrapper.write_text(
            f"""#!/bin/sh
if [ -n "${{CUSTOM_SCHEDULER_CREDENTIAL+x}}" \
    || [ -n "${{HF_TOKEN+x}}" \
    || [ -n "${{GITHUB_TOKEN+x}}" ]; then
    printf '{command}=visible\\n' >> "$PROBE_LOG"
else
    printf '{command}=sanitized\\n' >> "$PROBE_LOG"
fi
exec {real_command!r} "$@"
""",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
    config = tmp_path / "runtime.yaml"
    config.write_text(
        "fourkagent:\n  api_key_env: CUSTOM_SCHEDULER_CREDENTIAL\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PROBE_LOG": str(probe_log),
            "CUSTOM_SCHEDULER_CREDENTIAL": "runtime-private",
            "HF_TOKEN": "download-private",
            "GITHUB_TOKEN": "ambient-private",
        }
    )
    subprocess.run(
        [
            "/bin/bash",
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

    observations = probe_log.read_text(encoding="utf-8").splitlines()
    assert observations
    assert all(line.endswith("=sanitized") for line in observations)
