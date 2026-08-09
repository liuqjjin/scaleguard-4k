from __future__ import annotations

import gzip
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def load_overlay() -> ModuleType:
    path = (
        Path(__file__).parents[2]
        / "third_party"
        / "overlays"
        / "4kagent"
        / "run_native_restoration.py"
    )
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    specification = importlib.util.spec_from_file_location(
        "scaleguard_fourkagent_overlay",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def overlay() -> ModuleType:
    return load_overlay()


def write_gzip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        handle.write(payload)


def test_remote_scheduler_messages_are_text_only_and_request_json(overlay: ModuleType) -> None:
    messages = overlay._scheduler_messages("existing policy", "order these tasks", None)

    assert messages[0]["role"] == "system"
    assert "JSON" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "order these tasks"}
    assert "image" not in repr(messages).casefold()

    with pytest.raises(RuntimeError, match="text-only"):
        overlay._scheduler_messages(None, "prompt", [Path("private.png")])


def test_scheduler_structure_parser_requires_exact_unique_task_order(overlay: ModuleType) -> None:
    class Logger:
        def _log(self, _message: str, *, level: str) -> None:
            assert level == "warning"

    def validate(value: object) -> None:
        assert isinstance(value, dict)
        assert value["order"] == ["denoise", "deblur"]

    valid, normalized = overlay._safe_literal_check(
        Logger(),
        '{"thought":"remove noise first","order":["denoise","deblur"]}',
        validate,
    )
    assert valid is True
    assert normalized == '{"thought":"remove noise first","order":["denoise","deblur"]}'

    for malformed in (
        '{"thought":"x","order":["denoise","denoise"]}',
        '{"thought":"x","order":[]}',
        '{"thought":"x","order":["denoise"],"extra":true}',
    ):
        valid, normalized = overlay._safe_literal_check(Logger(), malformed, validate)
        assert valid is False
        assert normalized == ""


def test_scheduler_direct_arguments_keep_provider_key_and_retry_budget_bound(
    overlay: ModuleType,
) -> None:
    arguments = {
        "llm_provider": "dashscope",
        "llm_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "llm_region": "cn-beijing",
        "llm_model": "qwen3.7-flash-2026-07-15",
        "api_key_env": "DASHSCOPE_API_KEY",
        "llm_connect_timeout_seconds": 10.0,
        "llm_read_timeout_seconds": 120.0,
        "llm_max_transport_retries": 4,
        "llm_max_structure_retries": 2,
        "llm_max_completion_tokens": 1024,
        "llm_temperature": 0.0,
    }
    overlay._validate_scheduler_arguments(SimpleNamespace(**arguments))
    with pytest.raises(RuntimeError, match="consistently bound"):
        overlay._validate_scheduler_arguments(
            SimpleNamespace(**(arguments | {"api_key_env": "OPENAI_API_KEY"}))
        )
    with pytest.raises(RuntimeError, match="retry budget"):
        overlay._validate_scheduler_arguments(
            SimpleNamespace(**(arguments | {"llm_max_structure_retries": 1_000_000}))
        )
    with pytest.raises(overlay.SchedulerError, match="official DashScope endpoint"):
        overlay._validate_scheduler_arguments(
            SimpleNamespace(**(arguments | {"llm_base_url": "https://example.com/v1"}))
        )
    with pytest.raises(overlay.SchedulerError, match="temperature must be zero"):
        overlay._validate_scheduler_arguments(
            SimpleNamespace(**(arguments | {"llm_temperature": 0.5}))
        )


def test_runtime_view_copies_audited_bpe_to_only_the_two_required_locations(
    tmp_path: Path,
    overlay: ModuleType,
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "utils" / "clib_fiqa" / "model").mkdir(parents=True)
    (checkout / "README.md").write_text("audited checkout\n", encoding="utf-8")
    toolbox = tmp_path / "toolbox"
    source = toolbox / "pretrained_ckpts" / "hpsv2" / overlay.BPE_FILENAME
    write_gzip(source, b"audited tokenizer vocabulary")
    overlay.BPE_SHA256 = overlay._sha256(source)
    runtime_view = tmp_path / "runtime-view"

    hpsv2_bpe = overlay._prepare_runtime_view(checkout, toolbox, runtime_view)

    clib_bpe = runtime_view / "utils" / "clib_fiqa" / "model" / overlay.BPE_FILENAME
    expected_hpsv2_bpe = (
        runtime_view / ".scaleguard" / "hpsv2" / "src" / "open_clip" / overlay.BPE_FILENAME
    )
    assert hpsv2_bpe == expected_hpsv2_bpe
    assert clib_bpe.read_bytes() == source.read_bytes()
    assert hpsv2_bpe.read_bytes() == source.read_bytes()
    assert not clib_bpe.is_symlink()
    assert not hpsv2_bpe.is_symlink()
    assert not (checkout / "utils" / "clib_fiqa" / "model" / overlay.BPE_FILENAME).exists()


def test_runtime_view_rejects_a_different_bpe_vocabulary(
    tmp_path: Path,
    overlay: ModuleType,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    toolbox = tmp_path / "toolbox"
    source = toolbox / "pretrained_ckpts" / "hpsv2" / overlay.BPE_FILENAME
    write_gzip(source, b"different tokenizer vocabulary")

    with pytest.raises(RuntimeError, match="does not match the audited digest"):
        overlay._prepare_runtime_view(checkout, toolbox, tmp_path / "runtime-view")


def test_runtime_view_fails_closed_when_the_audited_bpe_is_absent(
    tmp_path: Path,
    overlay: ModuleType,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    toolbox = tmp_path / "toolbox"
    (toolbox / "pretrained_ckpts").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="audited runtime asset"):
        overlay._prepare_runtime_view(
            checkout,
            toolbox,
            tmp_path / "runtime-view",
        )


def test_hpsv2_uses_run_local_bpe_and_checkpoint_without_remote_pretraining(
    tmp_path: Path,
    overlay: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_parent = tmp_path / "site-packages"
    package = package_parent / "hpsv2"
    open_clip = package / "src" / "open_clip"
    open_clip.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "src" / "__init__.py").write_text("", encoding="utf-8")
    (open_clip / "__init__.py").write_text(
        "from . import tokenizer\n",
        encoding="utf-8",
    )
    (open_clip / "tokenizer.py").write_text(
        "\n".join(
            (
                "import gzip",
                "from pathlib import Path",
                "",
                "def default_bpe():",
                "    return str(Path(__file__).parent / 'bpe_simple_vocab_16e6.txt.gz')",
                "",
                "BPE_BYTES = gzip.open(default_bpe(), 'rb').read()",
                "",
            )
        ),
        encoding="utf-8",
    )
    (package / "img_score.py").write_text(
        "\n".join(
            (
                "from .src.open_clip import tokenizer",
                "",
                "calls = []",
                "",
                "def create_model_and_transforms(model_name, pretrained, *args, **kwargs):",
                "    calls.append(('factory', model_name, pretrained, args, kwargs))",
                "    return object()",
                "",
                "def score(images, prompt, cp=None, hps_version='v2.0'):",
                "    create_model_and_transforms(",
                "        'ViT-H-14',",
                "        'laion2B-s32B-b79K',",
                "        offline_contract=True,",
                "    )",
                "    calls.append(('score', images, prompt, cp, hps_version))",
                "    return [0.25]",
                "",
            )
        ),
        encoding="utf-8",
    )
    runtime_bpe = (
        tmp_path
        / "runtime-view"
        / ".scaleguard"
        / "hpsv2"
        / "src"
        / "open_clip"
        / overlay.BPE_FILENAME
    )
    write_gzip(runtime_bpe, b"runtime-only vocabulary")
    checkpoint = tmp_path / "weights" / "HPS_v2.1_compressed.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"locked checkpoint")

    monkeypatch.syspath_prepend(str(package_parent))
    monkeypatch.setattr(
        overlay.importlib.metadata,
        "version",
        lambda name: overlay.HPSV2_VERSION if name == "hpsv2" else "unexpected",
    )
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "hpsv2" or name.startswith("hpsv2.")
    }
    for name in previous_modules:
        monkeypatch.delitem(sys.modules, name)

    try:
        overlay._install_locked_hpsv2(runtime_bpe, checkpoint)
        hpsv2 = importlib.import_module("hpsv2")
        img_score = importlib.import_module("hpsv2.img_score")
        tokenizer = importlib.import_module("hpsv2.src.open_clip.tokenizer")

        assert tokenizer.BPE_BYTES == b"runtime-only vocabulary"
        assert not (open_clip / overlay.BPE_FILENAME).exists()
        assert hpsv2.score(["candidate.png"], "a clean image", hps_version="v2.1") == [0.25]
        assert img_score.calls == [
            (
                "factory",
                overlay.HPSV2_MODEL,
                None,
                (),
                {"offline_contract": True},
            ),
            (
                "score",
                ["candidate.png"],
                "a clean image",
                str(checkpoint),
                "v2.1",
            ),
        ]
        with pytest.raises(RuntimeError, match="unsupported HPSv2 version"):
            hpsv2.score(["candidate.png"], "prompt", hps_version="v2.0")
        checkpoint.write_bytes(b"changed checkpoint")
        with pytest.raises(RuntimeError, match="checkpoint changed"):
            hpsv2.score(["candidate.png"], "prompt", hps_version="v2.1")
    finally:
        for name in tuple(sys.modules):
            if name == "hpsv2" or name.startswith("hpsv2."):
                sys.modules.pop(name, None)


def test_outlines_cache_is_private_and_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overlay: ModuleType,
) -> None:
    cache_dir = tmp_path / "outlines-cache"
    cache_dir.mkdir(mode=0o700)
    calls: list[str] = []
    outlines = ModuleType("outlines")
    outlines.disable_cache = lambda: calls.append("disabled")  # type: ignore[attr-defined]

    monkeypatch.setenv("OUTLINES_CACHE_DIR", str(cache_dir.resolve()))
    monkeypatch.setattr(
        overlay.importlib.metadata,
        "version",
        lambda name: overlay.OUTLINES_VERSION if name == "outlines" else "unexpected",
    )

    def import_module(name: str) -> ModuleType:
        if name != "outlines":
            raise AssertionError(f"unexpected import: {name}")
        return outlines

    monkeypatch.setattr(overlay.importlib, "import_module", import_module)

    overlay._disable_outlines_cache()

    assert calls == ["disabled"]
    cache_dir.chmod(0o755)
    with pytest.raises(RuntimeError, match="private directory"):
        overlay._disable_outlines_cache()


def test_torchvision_compatibility_exposes_only_rgb_to_grayscale(
    overlay: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "torchvision.transforms.functional_tensor"
    functional = ModuleType("torchvision.transforms.functional")

    def rgb_to_grayscale(image: object) -> object:
        return image

    functional.rgb_to_grayscale = rgb_to_grayscale  # type: ignore[attr-defined]
    original_import = overlay.importlib.import_module

    def import_module(name: str) -> ModuleType:
        if name == module_name:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        if name == "torchvision.transforms.functional":
            return functional
        return original_import(name)

    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(overlay.importlib, "import_module", import_module)
    monkeypatch.setattr(
        overlay.importlib.metadata,
        "version",
        lambda name: "0.25.0+cu126" if name == "torchvision" else "unexpected",
    )

    overlay._install_torchvision_compatibility()

    compatibility = sys.modules[module_name]
    assert compatibility.rgb_to_grayscale is rgb_to_grayscale  # type: ignore[attr-defined]


def test_controlled_bridge_keeps_the_upstream_rollback_plan_invariant(
    overlay: ModuleType,
) -> None:
    """Upstream ``reschedule`` asserts done+plan == work_mem["plan"]["initial"].

    ``propose`` snapshots that baseline before the overlay appends the bridge, so
    the bridge has to reach both the live plan and the recorded baseline.
    """

    agent = SimpleNamespace(
        plan=["denoising"],
        work_mem={"plan": {"initial": ["denoising"], "adjusted": []}},
    )

    assert overlay._append_controlled_bridge(agent, bridge_factor=2) is True
    assert agent.plan == ["denoising", overlay.BRIDGE_SUBTASK]
    assert agent.work_mem["plan"]["initial"] == ["denoising", overlay.BRIDGE_SUBTASK]

    # Replay the upstream invariant after a failed subtask has been executed.
    done_subtasks = ["denoising"]
    agent.plan = [overlay.BRIDGE_SUBTASK]
    assert set(done_subtasks + agent.plan) == set(agent.work_mem["plan"]["initial"])


def test_controlled_bridge_is_appended_at_most_once(overlay: ModuleType) -> None:
    agent = SimpleNamespace(
        plan=["denoising"],
        work_mem={"plan": {"initial": ["denoising"], "adjusted": []}},
    )

    assert overlay._append_controlled_bridge(agent, bridge_factor=2) is True
    # A later reschedule must not duplicate a bridge that is still pending.
    assert overlay._append_controlled_bridge(agent, bridge_factor=2) is False
    # Nor may it re-add one that already ran.
    agent.plan = []
    agent._scaleguard_bridge_attempted = True
    assert overlay._append_controlled_bridge(agent, bridge_factor=2) is False

    assert agent.work_mem["plan"]["initial"].count(overlay.BRIDGE_SUBTASK) == 1


def test_controlled_bridge_remains_terminal_after_upstream_rescheduling(
    overlay: ModuleType,
) -> None:
    agent = SimpleNamespace(
        plan=[overlay.BRIDGE_SUBTASK, "denoising"],
        work_mem={"plan": {"initial": ["denoising", overlay.BRIDGE_SUBTASK]}},
    )

    assert overlay._append_controlled_bridge(agent, bridge_factor=2) is True
    assert agent.plan == ["denoising", overlay.BRIDGE_SUBTASK]


def test_controlled_bridge_rejects_a_duplicated_live_plan(overlay: ModuleType) -> None:
    agent = SimpleNamespace(
        plan=[overlay.BRIDGE_SUBTASK, "denoising", overlay.BRIDGE_SUBTASK],
        work_mem={"plan": {"initial": ["denoising", overlay.BRIDGE_SUBTASK]}},
    )

    with pytest.raises(RuntimeError, match="more than once"):
        overlay._append_controlled_bridge(agent, bridge_factor=2)


def test_no_bridge_is_appended_when_the_scale_plan_does_not_request_one(
    overlay: ModuleType,
) -> None:
    agent = SimpleNamespace(
        plan=["denoising"],
        work_mem={"plan": {"initial": ["denoising"], "adjusted": []}},
    )

    assert overlay._append_controlled_bridge(agent, bridge_factor=1) is False
    assert agent.plan == ["denoising"]
    assert agent.work_mem["plan"]["initial"] == ["denoising"]
