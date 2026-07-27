#!/usr/bin/env python3
"""Copy textual run evidence into a bounded, redacted diagnostics tree."""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import socket
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from scaleguard.strict_json import StrictJSONError, loads

ALLOWED_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".sha256",
    ".txt",
    ".yaml",
    ".yml",
}
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_SCANNED_PATHS = 20_000
MAX_COPIED_FILES = 5_000
MAX_SKIP_DETAILS = 200
SECRET_NAMES = (
    "HF_TOKEN",
    "HF_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DASHSCOPE_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "GITLAB_TOKEN",
    "CI_JOB_TOKEN",
)
SECRET_PATTERNS = (
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "[REDACTED:HF_TOKEN]"),
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "[REDACTED:API_KEY]"),
    (re.compile(r"AIza[A-Za-z0-9_-]{35}"), "[REDACTED:GOOGLE_API_KEY]"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "[REDACTED:GITHUB_TOKEN]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "[REDACTED:GITHUB_TOKEN]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED:AWS_ACCESS_KEY_ID]"),
    (
        re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]{16,}"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"(?i)(https?://)[^/@\s]+@"),
        r"\1[REDACTED_USERINFO]@",
    ),
    (
        re.compile(
            r"""(?i)(https?://[^\s?"'<>]+)\?(?!\[REDACTED_QUERY\])"""
            r"""[^\s"'<>)}\]]+"""
        ),
        r"\1?[REDACTED_QUERY]",
    ),
    (
        re.compile(r"(?i)(token|api[_-]?key|password|secret)(\s*[=:]\s*)([^\s,;]+)"),
        r"\1\2[REDACTED]",
    ),
)


def private_paths_from_execution(source: pathlib.Path) -> list[str]:
    """Discover user-selected input/output paths from bounded wrapper receipts."""

    values: list[str] = []
    inspected = 0
    scanned = 0
    for receipt_path in source.rglob("*"):
        scanned += 1
        if scanned > MAX_SCANNED_PATHS:
            break
        if receipt_path.name != "execution.json":
            continue
        inspected += 1
        if inspected > 100:
            break
        if receipt_path.is_symlink() or not receipt_path.is_file():
            continue
        try:
            if receipt_path.stat().st_size > MAX_FILE_BYTES:
                continue
            receipt = loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, StrictJSONError):
            continue
        if not isinstance(receipt, dict):
            continue
        inputs = receipt.get("inputs")
        if isinstance(inputs, dict):
            input_image = inputs.get("input_image")
            if isinstance(input_image, dict):
                value = input_image.get("path")
                if isinstance(value, str) and value:
                    values.append(value)
        outputs = receipt.get("outputs")
        if isinstance(outputs, list):
            for output in outputs:
                if not isinstance(output, dict):
                    continue
                value = output.get("path")
                if isinstance(value, str) and value:
                    values.append(value)
    return values


def redact(text: str, replacements: list[tuple[str, str]]) -> str:
    for value, replacement in replacements:
        if value:
            text = text.replace(value, replacement)
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def contains_inline_image_data(text: str) -> bool:
    """Return true for an image data URL before its encoded bytes are copied."""

    lowered = text.casefold()
    marker = "data:image/"
    start = lowered.find(marker)
    while start >= 0:
        header_end = lowered.find(",", start + len(marker))
        if header_end < 0:
            return False
        if ";base64" in lowered[start:header_end]:
            return True
        start = lowered.find(marker, start + len(marker))
    return False


def read_private_replacements(secret_fd: int) -> list[tuple[str, str]]:
    with os.fdopen(secret_fd, "rb", closefd=True) as handle:
        payload = handle.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise ValueError("private redaction stream exceeds 1 MiB")
    fields = payload.split(b"\0")
    if fields[-1:] == [b""]:
        fields.pop()
    if len(fields) % 2:
        raise ValueError("private redaction stream is malformed")

    replacements: list[tuple[str, str]] = []
    for index in range(0, len(fields), 2):
        try:
            name = fields[index].decode("ascii")
            value = fields[index + 1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("private redaction stream is not UTF-8") from exc
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("private redaction stream contains an invalid name")
        if value:
            replacements.append((value, f"[REDACTED:{name}]"))
    return replacements


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("destination", type=pathlib.Path)
    parser.add_argument("repo_root", type=pathlib.Path)
    parser.add_argument("cache_root", type=pathlib.Path)
    parser.add_argument(
        "--max-copied-files",
        type=int,
        default=MAX_COPIED_FILES,
    )
    parser.add_argument(
        "--max-copied-bytes",
        type=int,
        default=256 * 1024 * 1024,
    )
    parser.add_argument(
        "--secret-fd",
        type=int,
        help="read NUL-delimited name/value redaction pairs from this file descriptor",
    )
    args = parser.parse_args()
    if args.max_copied_files < 0:
        parser.error("--max-copied-files must be non-negative")
    if args.max_copied_bytes < 0:
        parser.error("--max-copied-bytes must be non-negative")

    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_dir():
        parser.error(f"source is not a directory: {source}")
    destination.mkdir(parents=True, exist_ok=True)

    if args.secret_fd is None:
        replacements = [(os.environ.get(name, ""), f"[REDACTED:{name}]") for name in SECRET_NAMES]
    else:
        try:
            replacements = read_private_replacements(args.secret_fd)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    replacements.extend(
        [
            (os.environ.get("SCALEGUARD_SMOKE_INPUT", ""), "$SMOKE_INPUT"),
            (
                os.environ.get("SCALEGUARD_INTEGRATION_INPUT", ""),
                "$INTEGRATION_INPUT",
            ),
        ]
    )
    for private_path in private_paths_from_execution(source):
        replacements.append((private_path, "$PRIVATE_INPUT_OR_OUTPUT"))
        try:
            replacements.append(
                (
                    str(pathlib.Path(private_path).resolve()),
                    "$PRIVATE_INPUT_OR_OUTPUT",
                )
            )
        except OSError:
            pass
    for hostname in {
        os.environ.get("HOSTNAME", ""),
        socket.gethostname(),
    }:
        if hostname and hostname.lower() not in {"localhost", "localhost.localdomain"}:
            replacements.append((hostname, "$HOSTNAME"))
    # Preserve both lexical and canonical spellings. On macOS, for example,
    # /var/... resolves to /private/var/...; collected tools may emit either.
    replacements.extend(
        [
            (str(args.repo_root), "$REPO_ROOT"),
            (str(args.repo_root.resolve()), "$REPO_ROOT"),
            (str(args.cache_root), "$CACHE_ROOT"),
            (str(args.cache_root.resolve()), "$CACHE_ROOT"),
        ]
    )
    user_home = pathlib.Path.home().resolve()
    if user_home != pathlib.Path("/"):
        replacements.append((str(user_home), "$USER_HOME"))

    # Replace longer roots first so a parent path cannot partially mask a more
    # specific replacement. Deduplicate aliases that resolve identically.
    replacements = sorted(set(replacements), key=lambda item: len(item[0]), reverse=True)

    copied = 0
    copied_bytes = 0
    scanned = 0
    copy_file_limit = min(MAX_COPIED_FILES, args.max_copied_files)
    skipped: list[str] = []
    skipped_count = 0

    def record_skip(message: str) -> None:
        nonlocal skipped_count
        skipped_count += 1
        if len(skipped) < MAX_SKIP_DETAILS:
            skipped.append(message)

    for path in source.rglob("*"):
        scanned += 1
        if scanned > MAX_SCANNED_PATHS:
            record_skip(f"scan stopped after {MAX_SCANNED_PATHS} paths")
            break
        if copied >= copy_file_limit:
            record_skip(f"copy stopped after {copy_file_limit} files")
            break
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source)
        relative_text = relative.as_posix()
        if redact(relative_text, replacements) != relative_text:
            record_skip("[REDACTED_PATH]: path contains secret-like material")
            continue
        if "diagnostics" in relative.parts:
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            record_skip(f"{relative.as_posix()}: cannot stat file ({exc})")
            continue
        if size > MAX_FILE_BYTES:
            record_skip(f"{relative.as_posix()}: exceeds {MAX_FILE_BYTES} bytes")
            continue
        if copied_bytes + size > args.max_copied_bytes:
            record_skip(f"{relative.as_posix()}: exceeds remaining diagnostics byte budget")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            record_skip(f"{relative.as_posix()}: not readable text ({exc})")
            continue
        if contains_inline_image_data(text):
            record_skip(f"{relative.as_posix()}: contains embedded image data")
            continue
        target = destination / "runs" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(redact(text, replacements), encoding="utf-8")
        copied += 1
        copied_bytes += target.stat().st_size

    summary = destination / "collection-summary.txt"
    summary.write_text(
        f"scanned_paths={scanned}\n"
        f"copied_text_files={copied}\n"
        f"copied_source_bytes={copied_bytes}\n"
        f"skipped_count={skipped_count}\n"
        f"skipped_details_truncated={max(0, skipped_count - len(skipped))}\n"
        + "".join(f"skipped={item}\n" for item in skipped),
        encoding="utf-8",
    )

    # Normalize already-collected system text as a final defense.
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path == summary:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            path.unlink()
            continue
        if contains_inline_image_data(text):
            path.unlink()
            continue
        path.write_text(redact(text, replacements), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
