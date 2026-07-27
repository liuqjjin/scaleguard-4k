#!/usr/bin/env python3
"""Run the audited 4KAgent as a native-scale, read-only restoration stage.

The adapter leaves planning, tool choice, reflection, rollback, and rescheduling
to 4KAgent. It removes generative 4x SR, permits at most one non-generative 2x
SwinIR bridge, and exposes only a dependency-complete base-environment toolbox.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

BPE_FILENAME = "bpe_simple_vocab_16e6.txt.gz"
BPE_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
HPSV2_VERSION = "1.2.0"
HPSV2_MODEL = "ViT-H-14"
HPSV2_PRETRAINED = "laion2B-s32B-b79K"
OUTLINES_VERSION = "0.2.1"
TORCHVISION_VERSION = "0.25.0"

TOOL_ALLOWLIST: dict[str, frozenset[str]] = {
    "denoising": frozenset({"swinir_15", "swinir_50", "mprnet", "restormer"}),
    "motion deblurring": frozenset({"mprnet", "restormer"}),
    "defocus deblurring": frozenset({"restormer"}),
    "dehazing": frozenset({"dehazeformer"}),
    "deraining": frozenset({"mprnet", "restormer"}),
    "brightening": frozenset({"histogram_equalization", "gamma_correction", "constant_shift"}),
    "jpeg compression artifact removal": frozenset({"fbcnn_blind", "swinir_40"}),
    "super-resolution": frozenset(),
    "super-resolution_2x": frozenset({"swinir_2x_gan", "swinir_2x_psnr"}),
    "super-resolution_16x": frozenset(),
    "face restoration": frozenset(),
    "old_photo_restoration": frozenset(),
}


def _install_safe_torch_load(torch_module: Any) -> None:
    match = re.match(r"^(\d+)\.(\d+)", str(torch_module.__version__))
    if match is None or tuple(map(int, match.groups())) < (2, 10):
        raise RuntimeError(
            "4KAgent requires PyTorch >=2.10.0 because earlier weights_only "
            "loaders are affected by CVE-2026-24747"
        )
    original_load = torch_module.load

    def safe_load(*load_args: object, **load_kwargs: object) -> object:
        if load_kwargs.get("weights_only") is False:
            raise RuntimeError("unsafe torch.load(weights_only=False) is forbidden")
        load_kwargs["weights_only"] = True
        return original_load(*load_args, **load_kwargs)

    torch_module.load = safe_load


def _disable_outlines_cache() -> None:
    if importlib.metadata.version("outlines") != OUTLINES_VERSION:
        raise RuntimeError(f"4KAgent requires outlines=={OUTLINES_VERSION}")
    cache_dir = Path(os.environ.get("OUTLINES_CACHE_DIR", ""))
    if (
        not cache_dir.is_absolute()
        or cache_dir.is_symlink()
        or not cache_dir.is_dir()
        or cache_dir.stat().st_mode & 0o077
    ):
        raise RuntimeError("OUTLINES_CACHE_DIR must be an existing private directory")
    outlines = importlib.import_module("outlines")
    disable_cache = getattr(outlines, "disable_cache", None)
    if not callable(disable_cache):
        raise RuntimeError("outlines cache-disable API does not match the audited contract")
    disable_cache()


def _install_packaging_compatibility() -> None:
    try:
        importlib.import_module("pkg_resources")
    except ModuleNotFoundError as error:
        if error.name != "pkg_resources":
            raise
        compatibility = ModuleType("pkg_resources")
        compatibility.packaging = importlib.import_module("packaging")  # type: ignore[attr-defined]
        sys.modules["pkg_resources"] = compatibility


def _install_torchvision_compatibility() -> None:
    module_name = "torchvision.transforms.functional_tensor"
    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise

    installed = importlib.metadata.version("torchvision").split("+", 1)[0]
    if installed != TORCHVISION_VERSION:
        raise RuntimeError(
            f"unsupported torchvision compatibility target: {installed}; "
            f"expected {TORCHVISION_VERSION}"
        )
    functional = importlib.import_module("torchvision.transforms.functional")
    rgb_to_grayscale = getattr(functional, "rgb_to_grayscale", None)
    if not callable(rgb_to_grayscale):
        raise RuntimeError("torchvision rgb_to_grayscale API does not match the audited contract")
    compatibility = ModuleType(module_name)
    compatibility.rgb_to_grayscale = rgb_to_grayscale  # type: ignore[attr-defined]
    sys.modules[module_name] = compatibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-view", type=Path, required=True)
    parser.add_argument("--toolbox-root", type=Path, required=True)
    parser.add_argument("--hps-root", type=Path, required=True)
    parser.add_argument("--quality-model-path", type=Path, required=True)
    parser.add_argument("--perception-model-path", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--tool-gpu", type=int, required=True)
    parser.add_argument("--bridge-factor", type=int, choices=(1, 2), default=1)
    parser.add_argument("--llm-config", type=Path)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--api-key-env", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_redacted_image_logging(base_llm_module: Any) -> None:
    """Keep upstream chat logs useful without embedding private image bytes."""

    if not callable(getattr(base_llm_module, "encode_img", None)):
        raise RuntimeError("4KAgent BaseLLM image-logging API does not match the audited contract")

    def summarize_image(image_path: str | os.PathLike[str]) -> str:
        path = Path(image_path)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
        return f"scaleguard-image-redacted;sha256={digest.hexdigest()};bytes={size}"

    base_llm_module.encode_img = summarize_image


def _git_status(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _ignored_artifacts(checkout: Path) -> tuple[str, ...]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "ls-files",
            "-z",
            "--others",
            "--ignored",
            "--exclude-standard",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(path for path in result.stdout.split("\0") if path)


def _copy_union(source: Path, destination: Path) -> None:
    """Copy audited source into a run-local writable view."""

    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in {".git", "__pycache__"} for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        target = destination / relative
        if path.is_dir():
            if target.exists() and not target.is_dir():
                raise RuntimeError(f"runtime-view collision at {target}")
            target.mkdir(parents=True, exist_ok=True)
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"unsupported runtime-view source entry: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.is_file() and _sha256(target) == _sha256(path):
                continue
            raise RuntimeError(f"runtime-view file collision at {target}")
        shutil.copy2(path, target)


def _link_union(source: Path, destination: Path) -> None:
    """Add hash-inventoried materialized weights without copying their bytes."""

    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if not relative.parts or any(part in {".git", "__pycache__"} for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        target = destination / relative
        if path.is_dir():
            if target.exists() and not target.is_dir():
                raise RuntimeError(f"runtime-view collision at {target}")
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not path.is_file():
            raise RuntimeError(f"unsupported runtime-view source entry: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.is_file() and _sha256(target) == _sha256(path):
                continue
            raise RuntimeError(f"runtime-view file collision at {target}")
        target.symlink_to(path.resolve())


def _copy_verified_runtime_asset(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"audited runtime asset is missing or unsafe: {source}")
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"runtime asset destination already exists: {destination}")
    source_size = source.stat().st_size
    source_digest = _sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(source, destination)
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.stat().st_size != source_size
            or _sha256(destination) != source_digest
        ):
            raise RuntimeError(f"runtime asset copy verification failed: {destination}")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _prepare_runtime_view(checkout: Path, toolbox_root: Path, runtime_view: Path) -> Path:
    if runtime_view.exists():
        raise FileExistsError(f"runtime view already exists: {runtime_view}")
    if not (toolbox_root / "pretrained_ckpts").is_dir():
        raise FileNotFoundError(
            f"materialized toolbox has no pretrained_ckpts directory: {toolbox_root}"
        )
    runtime_view.mkdir(parents=True)
    _copy_union(checkout, runtime_view)
    _link_union(toolbox_root, runtime_view)
    bpe_source = toolbox_root / "pretrained_ckpts" / "hpsv2" / BPE_FILENAME
    if bpe_source.is_symlink() or not bpe_source.is_file():
        raise FileNotFoundError(f"audited runtime asset is missing or unsafe: {bpe_source}")
    if _sha256(bpe_source) != BPE_SHA256:
        raise RuntimeError("4KAgent toolbox BPE vocabulary does not match the audited digest")
    _copy_verified_runtime_asset(
        bpe_source,
        runtime_view / "utils" / "clib_fiqa" / "model" / BPE_FILENAME,
    )
    hpsv2_bpe = runtime_view / ".scaleguard" / "hpsv2" / "src" / "open_clip" / BPE_FILENAME
    _copy_verified_runtime_asset(bpe_source, hpsv2_bpe)
    return hpsv2_bpe


def _install_locked_hpsv2(hpsv2_bpe: Path, checkpoint: Path) -> None:
    if importlib.metadata.version("hpsv2") != HPSV2_VERSION:
        raise RuntimeError(f"4KAgent requires hpsv2=={HPSV2_VERSION}")
    for label, path in {
        "run-local HPSv2 BPE vocabulary": hpsv2_bpe,
        "locked HPSv2 checkpoint": checkpoint,
    }.items():
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"{label} is missing or unsafe: {path}")

    hpsv2 = importlib.import_module("hpsv2")
    package_file = getattr(hpsv2, "__file__", None)
    if not isinstance(package_file, str):
        raise RuntimeError("hpsv2 has no package file")
    package_bpe = (
        Path(package_file).resolve().parent / "src" / "open_clip" / BPE_FILENAME
    ).resolve()
    if "hpsv2.img_score" in sys.modules or "hpsv2.src.open_clip.tokenizer" in sys.modules:
        raise RuntimeError("hpsv2 scoring modules loaded before locked asset binding")

    original_gzip_open = gzip.open
    redirected = False

    def audited_gzip_open(filename: Any, *open_args: Any, **open_kwargs: Any) -> Any:
        nonlocal redirected
        try:
            candidate = Path(os.fspath(filename)).resolve()
        except (TypeError, ValueError):
            return original_gzip_open(filename, *open_args, **open_kwargs)
        if candidate == package_bpe:
            redirected = True
            return original_gzip_open(hpsv2_bpe, *open_args, **open_kwargs)
        return original_gzip_open(filename, *open_args, **open_kwargs)

    gzip.open = audited_gzip_open
    try:
        img_score = importlib.import_module("hpsv2.img_score")
    finally:
        gzip.open = original_gzip_open

    tokenizer = importlib.import_module("hpsv2.src.open_clip.tokenizer")
    default_bpe = getattr(tokenizer, "default_bpe", None)
    if not redirected or not callable(default_bpe) or Path(default_bpe()).resolve() != package_bpe:
        raise RuntimeError("hpsv2 tokenizer did not consume the run-local BPE vocabulary")

    original_create = getattr(img_score, "create_model_and_transforms", None)
    original_score = getattr(img_score, "score", None)
    if not callable(original_create) or not callable(original_score):
        raise RuntimeError("hpsv2 scoring API does not match the audited 1.2.0 contract")

    def create_without_remote_pretrain(
        model_name: str,
        pretrained: str | None = None,
        *factory_args: object,
        **factory_kwargs: object,
    ) -> object:
        if model_name != HPSV2_MODEL or pretrained != HPSV2_PRETRAINED:
            raise RuntimeError("hpsv2 requested an unexpected model or pretrained checkpoint")
        return original_create(
            model_name,
            None,
            *factory_args,
            **factory_kwargs,
        )

    checkpoint_digest = _sha256(checkpoint)

    def locked_score(
        images: object,
        prompt: str,
        hps_version: str = "v2.1",
    ) -> object:
        if hps_version != "v2.1":
            raise RuntimeError(f"unsupported HPSv2 version: {hps_version}")
        if (
            checkpoint.is_symlink()
            or not checkpoint.is_file()
            or _sha256(checkpoint) != checkpoint_digest
        ):
            raise RuntimeError("locked HPSv2 checkpoint changed during execution")
        return original_score(
            images,
            prompt,
            cp=str(checkpoint),
            hps_version=hps_version,
        )

    img_score_api: Any = img_score
    hpsv2_api: Any = hpsv2
    img_score_api.create_model_and_transforms = create_without_remote_pretrain
    hpsv2_api.score = locked_score


def _filter_toolboxes(executor: Any) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    selected: dict[str, list[str]] = {}
    removed: dict[str, list[str]] = {}
    for subtask, toolbox in sorted(executor.toolbox_router.items()):
        allowed = TOOL_ALLOWLIST.get(subtask, frozenset())
        selected_tools = [tool for tool in toolbox if tool.tool_name in allowed]
        removed_tools = [tool for tool in toolbox if tool.tool_name not in allowed]
        executor.toolbox_router[subtask] = selected_tools
        selected[subtask] = [tool.tool_name for tool in selected_tools]
        removed[subtask] = [tool.tool_name for tool in removed_tools]
    return selected, removed


def _repair_shared_tool_paths(executor: Any, runtime_view: Path) -> None:
    executor_root = runtime_view / "executor"
    for toolbox in executor.toolbox_router.values():
        for tool in toolbox:
            if tool.work_dir is None:
                continue
            work_dir = Path(tool.work_dir)
            if not work_dir.exists():
                candidates = [
                    path for path in executor_root.glob(f"*/tools/{work_dir.name}") if path.is_dir()
                ]
                unique = sorted({path.resolve() for path in candidates})
                if len(unique) != 1:
                    raise RuntimeError(
                        f"cannot resolve shared tool directory {work_dir.name}: {unique}"
                    )
                work_dir.parent.mkdir(parents=True, exist_ok=True)
                work_dir.symlink_to(unique[0])
            if tool.script_path is None or not Path(tool.script_path).is_file():
                raise FileNotFoundError(
                    f"selected tool script is missing for {tool.tool_name}: {tool.script_path}"
                )


def _install_shell_free_tool_runner(tool_class: type[Any]) -> None:
    def invoke(tool: Any, *extra: object) -> None:
        tool._preprocess()
        if tool.work_dir is None or tool.script_path is None:
            raise RuntimeError(f"tool {tool.tool_name} has no executable script")
        options = [str(value) for value in tool._get_cmd_opts(*extra)]
        environment = {
            name: value
            for name, value in os.environ.items()
            if name
            in {
                "HOME",
                "LANG",
                "LD_LIBRARY_PATH",
                "LOGNAME",
                "OUTLINES_CACHE_DIR",
                "PATH",
                "SHELL",
                "SSL_CERT_DIR",
                "SSL_CERT_FILE",
                "TEMP",
                "TMP",
                "TMPDIR",
                "USER",
            }
            or name.startswith("LC_")
        }
        if tool.run_gpu_id is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(tool.run_gpu_id)
        environment["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
        subprocess.run(
            [sys.executable, str(tool.script_path), *options],
            cwd=tool.work_dir,
            env=environment,
            check=True,
        )
        tool._postprocess()

    tool_class._invoke = invoke


def _redirect_executor_roots(runtime_view: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name == "executor" or name.startswith("executor."):
            if hasattr(module, "project_root"):
                module.project_root = runtime_view


class _DisabledFaceRestoreHelper:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def clean_all(self) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"face restoration is disabled in the ScaleGuard profile: {name}")


def _safe_literal_check(
    llm: Any,
    response_text: str,
    format_check: Callable[[object], None],
) -> tuple[bool, str]:
    candidates = [response_text]
    stripped = response_text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        body = stripped[3:-3].strip()
        if body.startswith(("json", "python")):
            body = body.partition("\n")[2]
        candidates.append(body.strip())
    for candidate in candidates:
        try:
            value = ast.literal_eval(candidate)
            format_check(value)
        except (ValueError, SyntaxError, AssertionError) as error:
            last_error = error
            continue
        return True, repr(value)
    llm._log(f"Failed to parse a structured response: {last_error}", level="warning")
    return False, ""


def _install_locked_quality(
    pipeline_module: Any,
    quality_model_path: Path,
) -> None:
    import pyiqa
    import torch
    from PIL import Image

    metric: Any | None = None

    def get_metric() -> Any:
        nonlocal metric
        if metric is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            metric = pyiqa.create_metric(
                "musiq",
                device=device,
                pretrained_model_path=str(quality_model_path),
            )
        return metric

    def score(path: str | Path) -> float:
        value = get_metric()(str(path))
        return float(value.detach().cpu().item())

    def compute_iqa(path: str | Path) -> tuple[str, int, int]:
        with Image.open(path) as image:
            width, height = image.size
        value = score(path)
        return f"MUSIQ: {value:.4f}", height, width

    def compute_score(path: str | Path) -> float:
        return score(path) / 100.0

    def compute_batch(paths: list[str]) -> list[float]:
        return [compute_score(path) for path in paths]

    pipeline_module.compute_iqa = compute_iqa
    pipeline_module.compute_iqa_metric_score = compute_score
    pipeline_module.compute_iqa_metric_score_batch = compute_batch


def main() -> int:
    args = parse_args()
    checkout = args.checkout.resolve()
    toolbox_root = args.toolbox_root.resolve()
    runtime_view = args.runtime_view.resolve()
    quality_model_path = args.quality_model_path.resolve()
    perception_model_path = args.perception_model_path.resolve()
    hps_root = args.hps_root.resolve()
    required = {
        "4KAgent checkout": checkout / "pipeline" / "the4kagent_pipeline.py",
        "toolbox root": toolbox_root / "pretrained_ckpts",
        "HPSv2 weight": hps_root / "HPS_v2.1_compressed.pt",
        "MUSIQ weight": quality_model_path,
        "perception model": perception_model_path / "config.json",
    }
    for label, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is required by the audited 4KAgent scheduler")
    os.environ.pop(args.api_key_env, None)
    status_before = _git_status(checkout)
    if status_before:
        raise RuntimeError("4KAgent checkout must be clean before execution")
    ignored_before = _ignored_artifacts(checkout)
    if ignored_before:
        raise RuntimeError(
            "4KAgent checkout contains ignored artifacts: " + ", ".join(ignored_before[:20])
        )

    hpsv2_bpe = _prepare_runtime_view(checkout, toolbox_root, runtime_view)
    os.chdir(runtime_view)
    sys.path.insert(0, str(runtime_view))

    import torch

    _install_safe_torch_load(torch)
    _disable_outlines_cache()
    _install_packaging_compatibility()
    _install_torchvision_compatibility()
    import executor.tool as tool_module
    import llm.base_llm as base_llm_module
    import llm.gpt4 as gpt4_module
    import llm.qwen_vl as qwen_module
    import pipeline.the4kagent_pipeline as pipeline_module

    _install_redacted_image_logging(base_llm_module)
    _install_locked_hpsv2(
        hpsv2_bpe,
        hps_root / "HPS_v2.1_compressed.pt",
    )
    qwen_module.MODEL_ID = str(perception_model_path)
    pipeline_module.eval = ast.literal_eval
    gpt4_module.eval = ast.literal_eval
    pipeline_module.FaceRestoreHelper = _DisabledFaceRestoreHelper
    _install_shell_free_tool_runner(tool_module.Tool)
    _install_locked_quality(pipeline_module, quality_model_path)

    original_profile_loader = pipeline_module.load_profile_config

    def load_profile(name: str) -> dict[str, Any]:
        profile = dict(original_profile_loader(name))
        profile["FaceRestore"] = False
        profile["OldPhotoRestoration"] = False
        return profile

    pipeline_module.load_profile_config = load_profile

    original_gpt_init = pipeline_module.GPT4.__init__

    def initialize_gpt(llm: Any, *init_args: object, **init_kwargs: object) -> None:
        original_gpt_init(llm, *init_args, **init_kwargs)
        llm.api_key = api_key
        llm.model = args.llm_model

    pipeline_module.GPT4.__init__ = initialize_gpt
    pipeline_module.GPT4._check_syntax = _safe_literal_check

    agent_class = pipeline_module.The4KAgent
    executor = pipeline_module.executor
    selected_tools, removed_tools = _filter_toolboxes(executor)
    _repair_shared_tool_paths(executor, runtime_view)
    _redirect_executor_roots(runtime_view)

    original_extract_agenda = agent_class.extract_agenda
    original_propose = agent_class.propose
    original_reschedule = agent_class.reschedule
    original_execute = agent_class.execute_subtask

    def extract_agenda(agent: Any, evaluation: Any) -> list[str]:
        agenda = original_extract_agenda(agent, evaluation)
        forbidden = {
            "super-resolution",
            "super-resolution_2x",
            "super-resolution_16x",
            "face restoration",
            "old_photo_restoration",
        }
        return [task for task in agenda if task not in forbidden]

    def append_bridge(agent: Any) -> None:
        if args.bridge_factor == 2 and not getattr(agent, "_scaleguard_bridge_attempted", False):
            agent.plan.append("super-resolution_2x")

    def propose(agent: Any) -> None:
        original_propose(agent)
        append_bridge(agent)

    def reschedule(agent: Any) -> None:
        original_reschedule(agent)
        append_bridge(agent)

    def execute(agent: Any, cache: Path | None) -> bool:
        if agent.plan and agent.plan[0] == "super-resolution_2x":
            agent._scaleguard_bridge_attempted = True
        return bool(original_execute(agent, cache))

    agent_class.extract_agenda = extract_agenda
    agent_class.propose = propose
    agent_class.reschedule = reschedule
    agent_class.execute_subtask = execute
    pipeline_module.adain_color_fix = lambda target, _source: target

    llm_config = args.llm_config.resolve() if args.llm_config else checkout / "config.yml"
    agent = agent_class(
        input_path=args.input.resolve(),
        output_dir=args.output_dir.resolve(),
        llm_config_path=llm_config,
        with_retrieval=True,
        with_reflection=True,
        silent=False,
        tool_run_gpu_id=args.tool_gpu,
        profile_name=args.profile,
    )
    agent.project_root = runtime_view
    agent.run()

    executed = list(agent.work_mem["execution_path"]["subtasks"])
    forbidden_executed = [
        task
        for task in executed
        if task
        in {
            "super-resolution",
            "super-resolution_16x",
            "face restoration",
            "old_photo_restoration",
        }
    ]
    if forbidden_executed:
        raise RuntimeError(f"forbidden 4KAgent tasks executed: {forbidden_executed}")
    bridges = [task for task in executed if task == "super-resolution_2x"]
    if len(bridges) > (1 if args.bridge_factor == 2 else 0):
        raise RuntimeError(f"unexpected 2x bridge count in execution path: {len(bridges)}")
    if _git_status(checkout) != status_before:
        raise RuntimeError("4KAgent execution mutated the audited checkout")
    if _ignored_artifacts(checkout) != ignored_before:
        raise RuntimeError("4KAgent execution created ignored checkout artifacts")

    evidence = {
        "schema_version": "1.0",
        "result": str(Path(agent.work_dir, "result.png").resolve()),
        "execution_path": {
            "subtasks": executed,
            "tools": list(agent.work_mem["execution_path"]["tools"]),
        },
        "bridge_factor": args.bridge_factor,
        "terminal_generative_sr": False,
        "checkout_mutations": False,
        "runtime_view": str(runtime_view),
        "profile_overrides": {
            "FaceRestore": False,
            "OldPhotoRestoration": False,
            "quality_metrics": ["musiq"],
        },
        "toolbox": {
            "selected": selected_tools,
            "removed": removed_tools,
            "shell_free_executor": True,
        },
        "models": {
            "perception": str(perception_model_path),
            "quality": str(quality_model_path),
            "hps_root": str(hps_root),
            "remote_scheduler": args.llm_model,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scaleguard-result.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
