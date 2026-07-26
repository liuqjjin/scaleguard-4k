# Credential and diagnostics handling

Treat tokens, API keys, AutoDL connection details, signed URLs and private dataset
paths as secrets.

- Pass Hugging Face credentials through `HF_TOKEN` or
  `HUGGING_FACE_HUB_TOKEN`, or use `hf auth login` under the private configured
  `HF_HOME`; the downloader never adds `--token`.
- Enter secrets with hidden shell input. Do not put them in `commands.sh`,
  `SCALEGUARD_*_COMMAND`, JSON/YAML configs, command arguments or shell history.
- Gate subprocesses use least-privilege environments: no credential reaches
  bootstrap or source verification, only Hugging Face auth reaches the
  downloader, doctor receives a non-secret presence marker, and only model
  execution receives the configured scheduler key.
- Keep caches outside Git. Never copy Hugging Face cache metadata or model files
  into a diagnostics archive.
- `collect_diagnostics.sh` makes credential values non-exported before starting
  any system probe and sends exact values, including a custom scheduler
  variable, only to the sanitizer over a private file descriptor. It copies
  only bounded text formats. It excludes images,
  weights, archives, symlinks and environment dumps; replaces known secret values
  and common token patterns; derives direct CLI input/output paths from execution
  receipts; replaces the local hostname; then scans both content and
  archive-relative path names.
- Worker metadata can contain VLM descriptions or prompts derived from the
  authorized images. Treat that semantic text as input data during manual
  review, even when no pixels are present.
- Automatic redaction is defense-in-depth, not a guarantee. Before transfer:

```bash
tar -tzf artifacts/autodl/diagnostics/*/scaleguard-diagnostics-*.tar.gz
mkdir -p /tmp/scaleguard-review
tar -xzf /path/to/scaleguard-diagnostics-*.tar.gz -C /tmp/scaleguard-review
grep -RIE 'hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{35}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|Bearer[[:space:]]+[A-Za-z0-9._~+/=-]{16,}|https?://[^[][^/@[:space:]]*@|https?://[^?[:space:]]+\?[^[][^[:space:]]+' /tmp/scaleguard-review
```

Inspect all files manually, transfer only the reviewed archive and its SHA-256,
then remove the temporary review copy. Do not post raw logs directly to a public
issue.
