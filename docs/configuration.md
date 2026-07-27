# Runtime configuration reference

ScaleGuard reads strict YAML into frozen configuration objects. Unknown sections,
unknown fields, wrong scalar types, and unsupported cross-field combinations are
errors. YAML values are never expanded from environment variables; credentials
are named by `fourkagent.api_key_env` and read only when the worker starts.
Duplicate YAML mapping keys are rejected. For the pre-bootstrap AutoDL boundary,
write `fourkagent:` and `api_key_env:` as canonical unquoted block keys; quoted,
escaped, explicit, tagged, merged, flow/inline, or duplicate forms fail closed.
The credential variable must be uppercase and end in `_API_KEY`, `_TOKEN`,
`_CREDENTIAL`, or `_SECRET`, which
excludes interpreter and loader controls such as `PATH`, `BASH_ENV`,
`LD_PRELOAD`, and `PYTHONPATH`.

Use [the CPU/mock configuration](../configs/runtime/cpu-mock.yaml) for contract
tests and [the dual-4090 configuration](../configs/runtime/autodl-2x4090.yaml)
for the audited real-runtime layout. Validate either without loading a model:

```bash
scaleguard config validate /absolute/path/to/config.yaml
```

## Paths and project root

`scaleguard run`, `doctor`, and `upstream verify` operate on repository-managed
overlays and locks, so they require a project root. Run them inside a checkout or
set `SCALEGUARD_PROJECT_ROOT` to its absolute path. Relative runtime, checkout,
environment, weight, and calibration-receipt paths are resolved from that root.
An absolute `runtime.run_root` is useful when run artifacts must live elsewhere.

The file-only commands `config validate`, `manifest validate`, and `evaluation
verify` do not need a checkout. `evaluation calibrate`, `summarize`, and
`metrics` also work outside a checkout when `--artifact-root DIR` is supplied;
without it, their established default is the discovered project root.

## Annotated runtime YAML

The values below show defaults and supported domains. It is a reference, not a
production configuration: upstream modes additionally require materialized,
hash-verified checkouts and weights.

```yaml
runtime:
  run_root: runs                 # Run directories; relative paths use project root.
  process_timeout_seconds: 3600  # Positive deadline for each worker process.
  keep_temporary_files: false    # Reserved cleanup-policy flag; evidence is retained.
  gpu_poll_interval_seconds: 0.5 # Positive NVIDIA memory sampling interval.
  experiment_group: null         # Reserved four-group ablation identifier.
  experiment_sample_id: null     # Safe paired input/seed identifier; set with group.

fourkagent:
  mode: fake                     # fake | command | upstream | identity
  checkout: null                 # 4KAgent checkout; required for upstream mode.
  python_executable: python      # Interpreter name or project-relative/absolute path.
  profile: FastGen4K_P           # Upstream 4KAgent profile.
  tool_gpu: "0"                  # One numeric CUDA selector in upstream mode.
  command: []                    # Token array required only by command mode.

  # Managed DepictQA service. Upstream mode requires depictqa_command and cwd;
  # only loopback 127.0.0.1:5001 is accepted by the audited overlay.
  depictqa_command: []
  depictqa_cwd: null
  depictqa_host: 127.0.0.1
  depictqa_port: 5001
  depictqa_startup_timeout_seconds: 600
  depictqa_visible_devices: "1"  # Must differ from tool_gpu.

  perception_model_path: ""      # Local Qwen2.5-VL model used by 4KAgent.
  toolbox_root: null              # Materialized 4KAgent toolbox runtime.
  hps_root: null                  # Materialized HPSv2 assets.
  quality_model_path: null        # Local MUSIQ checkpoint passed to the overlay.
  llm_model: gpt-4-turbo          # Scheduler model recorded in evidence.
  api_key_env: OPENAI_API_KEY     # Uppercase credential-variable name, never the secret.

coz:
  mode: fake                     # fake | command | upstream | persistent
  checkout: null                 # Chain-of-Zoom checkout for real modes.
  python_executable: python      # Interpreter name or project-relative/absolute path.
  visible_devices: "0,1"         # Two distinct numeric CUDA selectors.
  command: []                    # Token array required only by command mode.
  model_path: stabilityai/stable-diffusion-3-medium-diffusers
  qwen_model_path: Qwen/Qwen2.5-VL-3B-Instruct
  sr_lora_path: null             # Required for upstream/persistent modes.
  vae_path: null                 # Required for upstream/persistent modes.
  vlm_lora_path: null            # Required when prompt_type is vlm.
  prompt_type: vlm               # vlm | vlm_base
  seed: 0                        # Base seed recorded for deterministic requests.
  mixed_precision: fp32          # Audited full-image path is fixed to fp32.
  tile_size: 512                 # Positive VAE encoder tile size.
  tile_overlap: 64               # Non-negative and smaller than tile_size.

metrics:
  quality_backend: gradient_proxy # gradient_proxy | pyiqa
  quality_metric: musiq           # PyIQA metric name when backend is pyiqa.
  quality_device: cpu             # Real integrated gating must remain on CPU.
  quality_model_path: null        # Explicit local PyIQA checkpoint.
  min_quality_gain: -0.02         # Candidate minus same-size baseline, higher is better.
  max_scale_nrmse: 0.12           # Low-pass cross-scale normalized RMSE ceiling.
  max_scale_edge_mae: 0.10        # Cross-scale edge-error ceiling.

  measurement_enabled: false
  measurement_model: resize       # resize | gaussian_psf | jpeg |
                                  # poisson_gaussian | haze
  measurement_parameters: {}      # Model-specific parameters listed below.
  max_measurement_nrmse: 0.12     # Observation-consistency error ceiling.
  calibration_receipt: null       # Receipt whose backend/thresholds must match.

controller:
  target_factor: 4                # 1 | 2 | 4 | 8 | 16
  max_coz_steps: 2                # 0..2; 16x requires persistent CoZ.
  color_strategy: adain           # none | adain
  acceptance_policy: trusted      # trusted | fixed
  accept_unvalidated_quality_proxy: false
```

In `fourkagent.mode: upstream`, `perception_model_path`, `toolbox_root`,
`hps_root`, and `quality_model_path` are required. In `coz.mode: upstream` or
`persistent`, both model locations must be local and the SR LoRA and VAE paths
are required; the VLM LoRA is additionally required for `prompt_type: vlm`.

The `gradient_proxy` is a CPU contract fixture, not a validated research metric.
A non-mock run rejects it unless
`controller.accept_unvalidated_quality_proxy` is explicitly enabled for
calibration work. Real integrated gating with PyIQA is constrained to CPU so it
does not interfere with either upstream GPU lifecycle.

`identity` restoration and `fixed` acceptance are not general runtime escape
hatches. They are accepted only when `experiment_group` and
`experiment_sample_id` select one of the exact A-only, B-only, or AB-fixed
contracts. `ScaleGuard` always uses upstream 4KAgent, persistent CoZ, and the
`trusted` policy. See
[ADR 0010](adr/0010-make-ablation-modes-explicit-and-executable.md).

## Observation models

When `metrics.measurement_enabled` is true, `measurement_parameters` accepts
only the keys for the selected model:

| Model | Parameters and defaults |
| --- | --- |
| `resize` | none |
| `gaussian_psf` | `sigma: 1.2`, non-negative |
| `jpeg` | `quality: 75`, integer from 1 through 100 |
| `poisson_gaussian` | `peak_photons: 60.0` positive, `read_noise_std: 0.01` non-negative, integer `seed: 0` |
| `haze` | `transmission: 0.75` and `atmospheric_light: 0.9`, each from 0 through 1 |

The stable model name and parameters are recorded in each manifest. They are
evaluated as an independent observation-consistency gate, not folded into an
opaque aggregate score. A finite `measurement_nrmse` and the factory-derived
canonical forward-model identity are mandatory when the gate is enabled and
must both be absent when disabled.

When `calibration_receipt` is non-null, controller construction verifies that
receipt against the complete metrics configuration before any worker starts.
The manifest retains its resolved path, byte length, SHA-256, and verified
semantic result. A copied threshold value without that matching receipt is not
trusted evidence.

## Command-mode templates

Command arrays are executed directly without a shell. The 4KAgent command
adapter expands `{input}`, `{input_dir}`, `{output}`, `{output_dir}`, and
`{bridge_factor}`. The CoZ command adapter expands `{input}`, `{input_dir}`,
`{output}`, `{output_dir}`, `{step_index}`, and `{seed}`. Managed DepictQA
commands expand `{project_root}`, `{checkout}`, and `{service_work_dir}`.

Use YAML lists so every argument remains an explicit token. Do not put secrets
in a command array: command evidence is retained with the run.
