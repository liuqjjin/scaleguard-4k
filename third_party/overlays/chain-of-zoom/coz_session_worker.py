#!/usr/bin/env python3
"""Persistent, one-step-at-a-time session around the audited CoZ implementation."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import re
import runpy
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, TextIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--qwen-path", required=True)
    parser.add_argument("--sr-lora", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--vlm-lora", type=Path)
    parser.add_argument("--prompt-type", choices=("vlm", "vlm_base"), default="vlm")
    parser.add_argument("--mixed-precision", choices=("fp32",), default="fp32")
    parser.add_argument("--vae-encoder-tile", type=int, default=1024)
    parser.add_argument("--vae-decoder-tile", type=int, default=128)
    parser.add_argument("--latent-tile", type=int, default=64)
    parser.add_argument("--latent-overlap", type=int, default=16)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--strict-json-helper", type=Path, required=True)
    parser.add_argument("--one-shot-input", type=Path)
    parser.add_argument("--one-shot-output", type=Path)
    parser.add_argument("--one-shot-metadata", type=Path)
    parser.add_argument("--one-shot-step-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def emit(stream: TextIO, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, sort_keys=True) + "\n")
    stream.flush()


def seed_everything(seed: int, torch: Any, np: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _install_safe_torch_load(torch_module: Any) -> None:
    match = re.match(r"^(\d+)\.(\d+)", str(torch_module.__version__))
    if match is None or tuple(map(int, match.groups())) < (2, 10):
        raise RuntimeError(
            "Chain-of-Zoom requires PyTorch >=2.10.0 because earlier weights_only "
            "loaders are affected by CVE-2026-24747"
        )
    original_load = torch_module.load

    def safe_load(*load_args: object, **load_kwargs: object) -> object:
        if load_kwargs.get("weights_only") is False:
            raise RuntimeError("unsafe torch.load(weights_only=False) is forbidden")
        load_kwargs["weights_only"] = True
        return original_load(*load_args, **load_kwargs)

    torch_module.load = safe_load


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def remove_patch_outputs(output: Path) -> None:
    patch_prefix = f"{output.stem}_patch"
    for patch in output.parent.iterdir():
        if patch.is_file() and patch.suffix == ".png" and patch.name.startswith(patch_prefix):
            patch.unlink(missing_ok=True)


def gpu_inventory() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    fields = ("logical_index", "uuid", "name", "memory_total_mib")
    return [
        dict(zip(fields, (part.strip() for part in line.split(",")), strict=True))
        for line in result.stdout.splitlines()
        if len(line.split(",")) == len(fields)
    ]


class CoZSession:
    def __init__(self, args: argparse.Namespace) -> None:
        checkout = args.checkout.resolve()
        os.chdir(checkout)
        sys.path.insert(0, str(checkout))
        self.args = args
        self.session_dir = args.session_dir.resolve()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_anchor = self.session_dir / "semantic_anchor.png"
        self.prompts: list[str] = []
        self.root_sha256: str | None = None
        self.trusted_sha256: str | None = None
        self.pending: dict[str, Any] | None = None
        self.inventory = gpu_inventory()

        import numpy as np
        import torch

        _install_safe_torch_load(torch)
        from osediff_sd3 import OSEDiff_SD3_TEST_TILE, SD3Euler
        from peft import PeftModel
        from PIL import Image
        from torchvision import transforms
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        source = (checkout / "osediff_sd3.py").read_text(encoding="utf-8")
        restricted_vae_load = (
            'encoder_state_dict_fp16 = torch.load(self.vae_path, map_location="cpu", '
            "weights_only=True)"
        )
        required_patch_markers = (
            restricted_vae_load,
            "self.model.scheduler.set_timesteps(1, device=device)",
            "inputs = inputs.to(next(vlm_model.parameters()).device)",
            "# 3) make gaussian weights and allocate streaming accumulators",
            "z_norm = torch.zeros_like(z_full)",
            "norm = torch.zeros_like(z_full)",
            "return z_norm, None",
        )
        missing = [marker for marker in required_patch_markers if marker not in source]
        if missing:
            raise RuntimeError(
                "CoZ checkout is missing audited ScaleGuard patch markers: " + ", ".join(missing)
            )

        model_args = argparse.Namespace(
            lora_path=str(args.sr_lora.resolve()),
            vae_path=str(args.vae.resolve()),
            lora_rank=4,
            vae_encoder_tiled_size=args.vae_encoder_tile,
            vae_decoder_tiled_size=args.vae_decoder_tile,
            latent_tiled_size=args.latent_tile,
            latent_tiled_overlap=args.latent_overlap,
        )
        self.torch = torch
        self.np = np
        self.Image = Image
        self.transforms = transforms
        seed_everything(args.seed, torch, np)

        model = SD3Euler(model_key=args.model_path, device="cpu")
        model.text_enc_1.to("cuda:0")
        model.text_enc_2.to("cuda:0")
        model.text_enc_3.to("cuda:0")
        # This mirrors the audited full-image path. The public configuration
        # accepts only fp32 so the requested and executed precision cannot drift.
        model.transformer.to("cuda:1", dtype=torch.float32)
        model.vae.to("cuda:1", dtype=torch.float32)
        for component in (
            model.text_enc_1,
            model.text_enc_2,
            model.text_enc_3,
            model.transformer,
            model.vae,
        ):
            component.requires_grad_(False)
        model.device = torch.device("cuda:0")
        self.model = model
        self.model_test = OSEDiff_SD3_TEST_TILE(model_args, model)

        vlm_name = args.qwen_path
        vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_name,
            torch_dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(vlm_name)
        if args.prompt_type == "vlm":
            if args.vlm_lora is None:
                raise RuntimeError("--vlm-lora is required for prompt-type=vlm")
            vlm = PeftModel.from_pretrained(vlm, str(args.vlm_lora.resolve()))
            vlm = vlm.merge_and_unload()
        vlm.eval()
        self.vlm = vlm
        self.processor = processor
        self.component_placement = {
            "text_encoder_1": self._first_parameter_placement(model.text_enc_1),
            "text_encoder_2": self._first_parameter_placement(model.text_enc_2),
            "text_encoder_3": self._first_parameter_placement(model.text_enc_3),
            "transformer": self._first_parameter_placement(model.transformer),
            "vae": self._first_parameter_placement(model.vae),
            "vlm_first_parameter": self._first_parameter_placement(vlm),
        }
        self._capture_prompts()

    @staticmethod
    def _first_parameter_placement(module: Any) -> dict[str, str]:
        parameter = next(module.parameters())
        return {"device": str(parameter.device), "dtype": str(parameter.dtype)}

    def _capture_prompts(self) -> None:
        original = self.model_test.create_prompt

        def create_prompt(*args: Any, **kwargs: Any) -> str:
            prompt = str(original(*args, **kwargs))
            self.prompts.append(prompt)
            return prompt

        self.model_test.create_prompt = create_prompt

    def _ensure_anchor(self, source: Path) -> None:
        if self.semantic_anchor.exists():
            return
        with self.Image.open(source) as image:
            rgb = image.convert("RGB")
            scale = 512.0 / min(rgb.size)
            size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
            rgb.resize(size, self.Image.Resampling.LANCZOS).save(self.semantic_anchor, "PNG")

    def upscale_once(
        self,
        source: Path,
        output: Path,
        seed: int,
        step_index: int,
    ) -> dict[str, Any]:
        started = time.monotonic()
        if self.pending is not None:
            raise RuntimeError(
                "accept or rollback the pending candidate before the next scale step"
            )
        input_sha256 = file_sha256(source)
        if self.trusted_sha256 is None:
            self.root_sha256 = input_sha256
            self.trusted_sha256 = input_sha256
        elif input_sha256 != self.trusted_sha256:
            raise RuntimeError(
                f"input hash {input_sha256} does not match trusted state {self.trusted_sha256}"
            )
        seed_everything(seed, self.torch, self.np)
        self.prompts.clear()
        self._ensure_anchor(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        for index in range(self.torch.cuda.device_count()):
            self.torch.cuda.reset_peak_memory_stats(index)

        with self.Image.open(source) as image:
            previous = image.convert("RGB")
            source_size = previous.size
            target_size = (previous.width * 4, previous.height * 4)
            resized = previous.resize(target_size, self.Image.Resampling.LANCZOS)
        parameter = next(self.model.vae.parameters())
        tensor = (
            self.transforms.ToTensor()(resized)
            .unsqueeze(0)
            .to(parameter.device, dtype=parameter.dtype)
            * 2
            - 1
        )
        latent, _ = self.model_test.create_full_latent(
            tensor,
            self.vlm,
            self.processor,
            str(self.semantic_anchor),
            str(output),
            self.args.prompt_type,
        )
        decoded = self.model_test.decode_full_latent(latent).cpu()
        result = self.transforms.ToPILImage()((decoded[0] * 0.5 + 0.5).clamp(0, 1))
        result.save(output, "PNG")
        remove_patch_outputs(output)
        peak_vram = {
            str(index): round(self.torch.cuda.max_memory_allocated(index) / 1024**2)
            for index in range(self.torch.cuda.device_count())
        }
        candidate_sha256 = file_sha256(output)
        self.pending = {
            "step_index": step_index,
            "path": str(output.resolve()),
            "sha256": candidate_sha256,
        }
        return {
            "source_size": list(source_size),
            "output_size": list(target_size),
            "seed": seed,
            "step_index": step_index,
            "root_sha256": self.root_sha256,
            "input_sha256": input_sha256,
            "candidate_sha256": candidate_sha256,
            "prompts": list(self.prompts),
            "duration_seconds": time.monotonic() - started,
            "peak_torch_allocated_mib": peak_vram,
            "requested_precision": self.args.mixed_precision,
            "actual_precision": {
                "transformer": str(next(self.model.transformer.parameters()).dtype),
                "vae": str(next(self.model.vae.parameters()).dtype),
            },
            "component_placement": self.component_placement,
            "semantic_anchor": str(self.semantic_anchor),
            "gpu_inventory": self.inventory,
            "mock": False,
        }

    def accept(self, step_index: int, candidate: Path, candidate_sha256: str) -> None:
        if self.pending is None:
            raise RuntimeError("there is no pending candidate to accept")
        expected = self.pending
        if (
            step_index != expected["step_index"]
            or str(candidate.resolve()) != expected["path"]
            or candidate_sha256 != expected["sha256"]
            or file_sha256(candidate) != expected["sha256"]
        ):
            raise RuntimeError("accept request does not match the pending candidate")
        self.trusted_sha256 = candidate_sha256
        self.pending = None

    def rollback(self, step_index: int) -> None:
        if self.pending is not None and step_index != self.pending["step_index"]:
            raise RuntimeError("rollback step does not match the pending candidate")
        self.pending = None

    def close(self) -> None:
        self.pending = None
        del self.vlm
        del self.processor
        del self.model_test
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


def run_one_shot(args: argparse.Namespace, protocol: TextIO) -> int:
    if not args.one_shot_input or not args.one_shot_output or not args.one_shot_metadata:
        raise ValueError("all one-shot paths are required together")
    with contextlib.redirect_stdout(sys.stderr):
        session = CoZSession(args)
        metadata = session.upscale_once(
            args.one_shot_input.resolve(),
            args.one_shot_output.resolve(),
            args.seed,
            args.one_shot_step_index,
        )
        session.close()
    args.one_shot_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.one_shot_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    emit(protocol, {"status": "ok", "op": "one_shot"})
    return 0


def run_jsonl(args: argparse.Namespace, protocol: TextIO) -> int:
    strict_json = runpy.run_path(str(args.strict_json_helper.resolve()))
    loads_object = strict_json.get("loads_object")
    if not callable(loads_object):
        raise RuntimeError("strict JSON helper does not expose loads_object")
    initialization_started = time.monotonic()
    with contextlib.redirect_stdout(sys.stderr):
        session = CoZSession(args)
    initialization_duration_seconds = time.monotonic() - initialization_started
    if not math.isfinite(initialization_duration_seconds) or initialization_duration_seconds < 0.0:
        raise RuntimeError("CoZ initialization clock produced an invalid duration")
    emit(
        protocol,
        {
            "status": "ready",
            "schema_version": "1.0",
            "backend": "chain-of-zoom",
            "persistent": True,
            "initialization_duration_seconds": initialization_duration_seconds,
        },
    )
    for line in sys.stdin:
        request: dict[str, Any] = {}
        try:
            request = loads_object(line)
            if not isinstance(request, dict):
                raise ValueError("CoZ protocol request must be a JSON object")
            request_id = str(request["request_id"])
            operation = request["op"]
            if operation == "health":
                emit(protocol, {"request_id": request_id, "status": "ok", "op": operation})
                continue
            if operation == "close":
                with contextlib.redirect_stdout(sys.stderr):
                    session.close()
                emit(protocol, {"request_id": request_id, "status": "ok", "op": operation})
                return 0
            if operation == "accept":
                with contextlib.redirect_stdout(sys.stderr):
                    session.accept(
                        int(request["step_index"]),
                        Path(request["candidate"]),
                        str(request["candidate_sha256"]),
                    )
                emit(protocol, {"request_id": request_id, "status": "ok", "op": operation})
                continue
            if operation == "rollback":
                with contextlib.redirect_stdout(sys.stderr):
                    session.rollback(int(request["step_index"]))
                emit(protocol, {"request_id": request_id, "status": "ok", "op": operation})
                continue
            if operation != "upscale":
                raise ValueError(f"unknown operation: {operation}")
            with contextlib.redirect_stdout(sys.stderr):
                metadata = session.upscale_once(
                    Path(request["input"]).resolve(),
                    Path(request["output"]).resolve(),
                    int(request["seed"]),
                    int(request["step_index"]),
                )
            emit(
                protocol,
                {
                    "request_id": request_id,
                    "status": "ok",
                    "op": operation,
                    "output": str(Path(request["output"]).resolve()),
                    "metadata": metadata,
                },
            )
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit(
                protocol,
                {
                    "request_id": str(request.get("request_id", "unknown")),
                    "status": "error",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
    with contextlib.redirect_stdout(sys.stderr):
        session.close()
    return 0


def main() -> int:
    args = parse_args()
    one_shot = any((args.one_shot_input, args.one_shot_output, args.one_shot_metadata))
    protocol = sys.stdout
    try:
        return run_one_shot(args, protocol) if one_shot else run_jsonl(args, protocol)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
