from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "scripts" / "autodl" / "_common.sh"


def _lease_topology(tmp_path: Path) -> str:
    return f"test-{hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]}"


def _host_lease_root() -> Path:
    return Path("/tmp") / f"scaleguard-4k-runtime-leases-{os.getuid()}"


def test_scheduler_environment_defaults_to_dashscope_and_clears_openai(
    tmp_path: Path,
) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text("fourkagent:\n  mode: upstream\n", encoding="utf-8")
    result = subprocess.run(
        [
            "/bin/bash",
            "-uec",
            (
                'source "$1"; sg_resolve_scheduler_api_key_env "$2"; '
                'sg_register_sensitive_env_name "$SG_SCHEDULER_API_KEY_ENV"; '
                "sg_make_sensitive_environment_private; "
                'sg_run_with_scheduler_credential "$SG_SCHEDULER_API_KEY_ENV" '
                'python3 -c "import os; '
                "assert os.environ['DASHSCOPE_API_KEY'] == 'dashscope-only'; "
                "assert 'OPENAI_API_KEY' not in os.environ\""
            ),
            "_",
            str(COMMON),
            str(config),
        ],
        env={
            **os.environ,
            "DASHSCOPE_API_KEY": "dashscope-only",
            "OPENAI_API_KEY": "openai-must-not-enter-runtime",
        },
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_run_directory_allocation_never_reuses_an_existing_leaf(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    environment = {
        **os.environ,
        "SCALEGUARD_ARTIFACT_ROOT": str(artifact_root),
        "SCALEGUARD_RUN_ID": "same-run-id",
    }
    command = [
        "/bin/bash",
        "-uec",
        'source "$1"; sg_new_run_dir smoke; printf "%s\\n" "$SG_RUN_DIR"',
        "_",
        str(COMMON),
    ]

    first = subprocess.run(
        command,
        env=environment,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        command,
        env=environment,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    first_path = Path(first.stdout.strip())
    second_path = Path(second.stdout.strip())
    assert first_path != second_path
    assert first_path.is_dir()
    assert second_path.is_dir()


def test_gpu_topology_lease_is_exclusive_and_released_on_process_exit(
    tmp_path: Path,
) -> None:
    topology = _lease_topology(tmp_path)
    lease_path = _host_lease_root() / f"{topology}.lock"
    environment = {
        **os.environ,
        "SCALEGUARD_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
    }
    holder = subprocess.Popen(
        [
            "/bin/bash",
            "-uec",
            'source "$1"; sg_acquire_gpu_lease "$2"; echo ready; sleep 30',
            "_",
            str(COMMON),
            topology,
        ],
        env=environment,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "ready"
        collision = subprocess.run(
            [
                "/bin/bash",
                "-uec",
                'source "$1"; sg_acquire_gpu_lease "$2"',
                "_",
                str(COMMON),
                topology,
            ],
            env={
                **environment,
                "SCALEGUARD_ARTIFACT_ROOT": str(tmp_path / "other-artifacts"),
            },
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert collision.returncode != 0
        assert "already leased by another ScaleGuard run" in collision.stderr
    finally:
        holder.terminate()
        holder.wait(timeout=5.0)

    deadline = time.monotonic() + 2.0
    while True:
        released = subprocess.run(
            [
                "/bin/bash",
                "-uec",
                'source "$1"; sg_acquire_gpu_lease "$2"',
                "_",
                str(COMMON),
                topology,
            ],
            env=environment,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if released.returncode == 0 or time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    assert released.returncode == 0, released.stderr
    lease_path.unlink(missing_ok=True)


def test_gpu_topology_lease_rejects_a_symlink_lock(tmp_path: Path) -> None:
    topology = _lease_topology(tmp_path)
    artifact_root = tmp_path / "artifacts"
    lease_root = _host_lease_root()
    lease_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    outside = tmp_path / "outside.lock"
    outside.write_text("do not follow\n", encoding="utf-8")
    lease_path = lease_root / f"{topology}.lock"
    lease_path.symlink_to(outside)

    result = subprocess.run(
        [
            "/bin/bash",
            "-uec",
            'source "$1"; sg_acquire_gpu_lease "$2"',
            "_",
            str(COMMON),
            topology,
        ],
        env={**os.environ, "SCALEGUARD_ARTIFACT_ROOT": str(artifact_root)},
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "GPU lease must be an owned regular file" in result.stderr
    assert outside.read_text(encoding="utf-8") == "do not follow\n"
    lease_path.unlink()


def test_gpu_monitor_rejects_zero_and_unbounded_intervals(tmp_path: Path) -> None:
    for interval in ("0", "60.1"):
        result = subprocess.run(
            [
                "/bin/bash",
                "-uec",
                'source "$1"; sg_start_gpu_monitor "$2" "$3"',
                "_",
                str(COMMON),
                str(tmp_path / "gpu.csv"),
                str(tmp_path / "gpu-check.json"),
            ],
            env={
                **os.environ,
                "SCALEGUARD_GPU_SAMPLE_INTERVAL_SECONDS": interval,
            },
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "must be between 0.1 and 60" in result.stderr


def test_autodl_runner_wraps_the_whole_attempt_in_a_bounded_deadline() -> None:
    runner = (ROOT / "scripts" / "autodl" / "_run_scaleguard.sh").read_text(encoding="utf-8")

    assert "SCALEGUARD_RUN_DEADLINE_SECONDS:-14400" in runner
    assert "--signal=TERM" in runner
    assert "--kill-after=30s" in runner
    assert '/bin/bash -p "${sg_here}/_run_scaleguard.sh"' in runner


def test_gpu_monitor_uses_preflight_uuids_and_separates_sample_kinds(
    tmp_path: Path,
) -> None:
    gpu_check = tmp_path / "gpu-check.json"
    gpu_check.write_text(
        json.dumps(
            {
                "status": "passed",
                "selected_gpus": [
                    {"uuid": "GPU-first"},
                    {"uuid": "GPU-second"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "gpu.csv"
    script = r"""
source "$1"
nvidia-smi() {
    [[ "$*" == *"-i GPU-first,GPU-second"* ]] || return 17
    printf '%s\n' \
        '0, GPU-first, NVIDIA GeForce RTX 4090, 100, 24564, 0' \
        '1, GPU-second, NVIDIA GeForce RTX 4090, 110, 24564, 0'
}
sg_start_gpu_monitor "$2" "$3"
sleep 0.25
sg_stop_gpu_monitor
"""
    result = subprocess.run(
        ["/bin/bash", "-uec", script, "_", str(COMMON), str(destination), str(gpu_check)],
        env={
            **os.environ,
            "SCALEGUARD_GPU_SAMPLE_INTERVAL_SECONDS": "0.1",
        },
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = destination.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("timestamp_utc,sample_kind,index,uuid")
    assert any(",inventory,0, GPU-first," in row for row in rows)
    assert any(",workload,0, GPU-first," in row for row in rows)
    assert all("GPU-first" in row or "GPU-second" in row or row == rows[0] for row in rows)


def test_experiment_handoff_refuses_to_clobber_another_attempt(tmp_path: Path) -> None:
    runner = (ROOT / "scripts" / "autodl" / "_run_scaleguard.sh").read_text(encoding="utf-8")
    function_body = runner.split("sg_write_attempt_pointer()", 1)[1]
    python_body = function_body.split("<<'PY'\n", 1)[1].split("\nPY\n}", 1)[0]
    helper = tmp_path / "handoff_helper.py"
    helper.write_text(python_body, encoding="utf-8")
    output = tmp_path / "handoff.json"
    first_attempt = tmp_path / "attempt-first"
    second_attempt = tmp_path / "attempt-second"
    first_attempt.mkdir()
    second_attempt.mkdir()
    started_at = "2026-08-08T00:00:00Z"

    subprocess.run(
        [
            sys.executable,
            "-I",
            str(helper),
            str(output),
            str(first_attempt),
            "running",
            started_at,
            "",
        ],
        check=True,
    )
    original = output.read_bytes()
    collision = subprocess.run(
        [
            sys.executable,
            "-I",
            str(helper),
            str(output),
            str(second_attempt),
            "running",
            started_at,
            "",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert collision.returncode != 0
    assert "refusing to clobber another experiment handoff" in collision.stderr
    assert output.read_bytes() == original
