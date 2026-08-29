# Upstream and model artifact audit

Audit date: 2026-08-08

## Conclusion

ScaleGuard-4K has exactly two core runtime upstreams: 4KAgent and
Chain-of-Zoom (CoZ). Both are locked to immutable commits and root trees in
`upstream-lock.yaml`. AgenticIR is retained only as a code-lineage and design
reference because 4KAgent acknowledges that lineage; it is not fetched,
imported, executed, or represented as a third runtime.

This is a source, metadata, license-boundary, and reproducibility audit. It
does not replace the dual-GPU study. Runtime, peak-memory, throughput, and
image-quality measurements belong to that evaluation, not to this audit.

## Method

The audit used primary sources:

- immutable Git commits, Git root-tree objects, licenses, and repository code;
- current remote refs queried with `git ls-remote`;
- the official NeurIPS and ICLR proceedings pages;
- Hugging Face model APIs and model cards at immutable revisions;
- PyPI distribution metadata and hashes;
- open GitHub issues and maintainer responses; and
- local SHA-256 measurements for source patches and publicly retrievable
  checkpoint blobs.

Mutable remote status and issue state are a snapshot at the audit date. The
locks remain immutable even if a branch changes later.

## Repository identities

| Role | Repository | Locked commit | Root tree | Declared license | Remote status at audit |
| --- | --- | --- | --- | --- | --- |
| Core restoration/planning runtime | [4KAgent](https://github.com/taco-group/4KAgent) | [`04179ffe`](https://github.com/taco-group/4KAgent/commit/04179ffed12c1db64890a48ac6d51ae4abeaee37) | `66f62367c915042ef945562bddc791bfc445fb42` | [Apache-2.0](https://github.com/taco-group/4KAgent/blob/04179ffed12c1db64890a48ac6d51ae4abeaee37/LICENSE) | `HEAD`, `main`, and lightweight tag `v1.0` still resolved to the locked commit |
| Core terminal SR runtime | [Chain-of-Zoom](https://github.com/bryanswkim/Chain-of-Zoom) | [`5a4f351c`](https://github.com/bryanswkim/Chain-of-Zoom/commit/5a4f351c9c655568c4705d5fc53ef70a3f28905f) | `177741fa5443a5073f29b1cf3ffad67e596d0b5d` | [MIT](https://github.com/bryanswkim/Chain-of-Zoom/blob/5a4f351c9c655568c4705d5fc53ef70a3f28905f/LICENSE) | `HEAD` and `main` still resolved to the locked commit; no tag was advertised |

The root tree is recorded separately from the commit so checkout verification
detects a commit mismatch and a content-tree mismatch independently. The
materialization and ordered patch procedure is documented in
[`third_party/README.md`](../third_party/README.md).

### Lineage reference, not a runtime

[AgenticIR](https://github.com/Kaiwen-Zhu/AgenticIR) was inspected at
[`9640a291`](https://github.com/Kaiwen-Zhu/AgenticIR/commit/9640a291480dee3ba8f2974125d4ee9e3440f3d6),
root tree `ecb79f37a4d002bb1c6d7e745578e3369f00eee7`. That commit was also
`HEAD`/`main` at the audit date and advertised no tag. It has no root license
file or GitHub-detected repository license. ScaleGuard therefore copies no
AgenticIR source and grants no implied permission over it. It is intentionally
absent from `upstream-lock.yaml` and `third_party/checkouts/`.

## Official publications

| Project | Official record | Use in ScaleGuard |
| --- | --- | --- |
| 4KAgent | [NeurIPS 2025 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f0075fe4e59652cf43148dcfab8d3c93-Abstract-Conference.html) | Core high-level degradation perception, native-scale restoration, reflection, and non-terminal-SR rollback |
| Chain-of-Zoom | [NeurIPS 2025 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b66d8cbb01ac8212830068f3d75b4c5c-Abstract-Conference.html) | Sole terminal generative SR method, exposed as explicit 4× scale transitions |
| AgenticIR | [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/921ac785fa9edc73cacaf2664f43d234-Abstract-Conference.html) | Citation and lineage context only |

Claims and measurements in those papers belong to the upstream authors. They
must not be relabeled as ScaleGuard results.

## 4KAgent source findings

The following observations are about the pinned source, not the paper's
abstract architecture.

1. The native pipeline maps target factors 2/4/8/16 to named agenda entries.
   Its 16× representation is two identical `super-resolution` strings
   ([source](https://github.com/taco-group/4KAgent/blob/04179ffed12c1db64890a48ac6d51ae4abeaee37/pipeline/the4kagent_pipeline.py#L345-L357)).
   Agenda extraction also shuffles restoration and SR entries before scheduling
   ([source](https://github.com/taco-group/4KAgent/blob/04179ffed12c1db64890a48ac6d51ae4abeaee37/pipeline/the4kagent_pipeline.py#L473-L499)).

2. Schedule validation compares `set(order)` with `set(agenda)`, so it does not
   prove the multiplicity of repeated scale steps. Rollback/reschedule logic
   likewise converts completed tasks and plans to sets
   ([schedule checks](https://github.com/taco-group/4KAgent/blob/04179ffed12c1db64890a48ac6d51ae4abeaee37/pipeline/the4kagent_pipeline.py#L646-L714),
   [rollback state](https://github.com/taco-group/4KAgent/blob/04179ffed12c1db64890a48ac6d51ae4abeaee37/pipeline/the4kagent_pipeline.py#L1153-L1184)).
   A repeated CoZ scale therefore needs a separate ordered state machine.

3. Passing an explicit plan suppresses rollback and rescheduling because the
   failure branch is guarded by `plan is None`
   ([source](https://github.com/taco-group/4KAgent/blob/04179ffed12c1db64890a48ac6d51ae4abeaee37/pipeline/the4kagent_pipeline.py#L299-L311)).
   The ScaleGuard adapter consequently filters the generated agenda rather than
   replacing the whole native run with an explicit list.

4. The `Tool` launcher chooses Conda environments from a fixed tool-name list,
   accepts one integer GPU id, constructs a shell command, and suppresses child
   stdout and stderr
   ([source](https://github.com/taco-group/4KAgent/blob/04179ffed12c1db64890a48ac6d51ae4abeaee37/executor/tool.py#L65-L115)).
   This cannot express CoZ's audited two-GPU placement or a persistent,
   observable session without invasive changes.

5. After planning, the in-process perception object is deleted and the CUDA
   cache is emptied
   ([source](https://github.com/taco-group/4KAgent/blob/04179ffed12c1db64890a48ac6d51ae4abeaee37/pipeline/the4kagent_pipeline.py#L396-L404)).
   The separately launched DepictQA service shown in the upstream instructions
   is outside this object's lifetime. ScaleGuard must own that service process
   and record its start, readiness, and stop outcome.

6. Candidate selection computes HPSv2 and, for one profile, adds an IQA score
   directly before selecting the maximum
   ([source](https://github.com/taco-group/4KAgent/blob/04179ffed12c1db64890a48ac6d51ae4abeaee37/pipeline/the4kagent_pipeline.py#L914-L961)).
   For large SR outputs, AdaIN then overwrites the selected file after that
   score was computed
   ([source](https://github.com/taco-group/4KAgent/blob/04179ffed12c1db64890a48ac6d51ae4abeaee37/pipeline/the4kagent_pipeline.py#L871-L909)).
   ScaleGuard therefore reserves one final color operation and re-scores the
   final bytes instead of treating the pre-mutation score as final evidence.

These constraints lead to
[ADR 0001](adr/0001-terminal-chain-of-zoom-session.md): 4KAgent remains the
only degradation planner, while terminal scale recursion is not registered as
another 4KAgent tool.

## Chain-of-Zoom source findings

1. The full-image entry point owns the whole `rec_num` loop and emits both
   per-sample and per-scale images
   ([source](https://github.com/bryanswkim/Chain-of-Zoom/blob/5a4f351c9c655568c4705d5fc53ef70a3f28905f/inference_coz_full.py#L124-L181)).
   It has no caller-visible accept/stop/rollback boundary between recursions.

2. The full-image parser accepts a model path, seed, alignment method, and
   mixed-precision label, but the pinned execution constructs `SD3Euler()`
   without the parsed model path, does not consume the seed, does not execute an
   alignment branch, and places the transformer and VAE on `cuda:1` in FP32
   ([source](https://github.com/bryanswkim/Chain-of-Zoom/blob/5a4f351c9c655568c4705d5fc53ef70a3f28905f/inference_coz_full.py#L28-L74)).
   The README's `--efficient_memory` option is not defined by this full-image
   parser. ScaleGuard exposes only the observed FP32 contract and records actual
   component placement instead of inferring it from unused flags.

3. The full-image path hard-codes the Qwen repository and uses
   `device_map="auto"`
   ([source](https://github.com/bryanswkim/Chain-of-Zoom/blob/5a4f351c9c655568c4705d5fc53ef70a3f28905f/inference_coz_full.py#L92-L120)).
   ScaleGuard supplies a local, revision-pinned path and records device
   inventory rather than allowing a mutable model lookup.

4. The pinned full-latent method reads `scheduler.timesteps[0]` without first
   setting the one-step schedule. Its patch-prompt processor output is not moved
   to the VLM device
   ([latent path](https://github.com/bryanswkim/Chain-of-Zoom/blob/5a4f351c9c655568c4705d5fc53ef70a3f28905f/osediff_sd3.py#L785-L845),
   [prompt path](https://github.com/bryanswkim/Chain-of-Zoom/blob/5a4f351c9c655568c4705d5fc53ef70a3f28905f/osediff_sd3.py#L857-L897)).
   The first locked patch fixes these contracts and restricts VAE state-dict
   loading.

5. The same method retains all tile positions and output tiles, plus full-size
   tensors, until a second fusion pass. The second locked patch preserves
   Gaussian overlap fusion while accumulating each processed tile immediately.
   Full latent and accumulation tensors still scale with image area; neither
   static inspection nor streaming fusion establishes a safe VRAM ceiling.

These findings lead to
[ADR 0002](adr/0002-one-step-4x-session-and-patch-overlay.md): one persistent
CoZ session exposes one explicit 4× transition at a time. A second 4×
transition for 16× is requested only after the first candidate is accepted.

## Patch ledger

Both patches modify only CoZ's `osediff_sd3.py` and apply in numeric order.

| Patch | SHA-256 | Scope |
| --- | --- | --- |
| `0001-full-inference-contract.patch` | `b9f2b6c293b8ac7e48be8188318992f16cee9f6f9affa2ccf0f41307027d00a0` | restricted VAE state loading, one-step scheduler initialization, VLM input-device transfer |
| `0002-streaming-gaussian-blend.patch` | `446c82cfdeb9e822814292511bdf80b7bd0401c08a54d82d7d29d4983901ca78` | immediate Gaussian-weighted accumulation and removal of retained output-tile list and unused unweighted accumulator |

After both patches, `osediff_sd3.py` must have SHA-256
`c5aea528f9f1206cae6ca666b5d0907a1b18e701300ce1252a9468d682fa6084`.
The verifier derives allowed dirty paths from each patch, rejects unrelated
checkout edits, and checks final bytes so an extra edit inside the patched file
cannot hide behind the allowlist. Numerical equivalence of streaming fusion
and its memory effect must be confirmed on GPU, not inferred from the diff
alone; the dual-GPU study records those measurements.

## Model and checkpoint lock

`weights-lock.json` uses immutable Hugging Face revisions where available,
expected SHA-256 for public single-file downloads, and post-download file
inventories for every artifact. A null upstream digest means “locally measured
but not authenticated against a publisher digest,” not “verified.”

### CoZ runtime models

| Artifact | Immutable identity or SHA-256 | Gate/license observation |
| --- | --- | --- |
| Stable Diffusion 3 Medium diffusers | Hugging Face revision `ea42f8cef0f178587cf766dc8129abd379c90671` | Auto-gated; Stability AI Non-Commercial Research Community License |
| Qwen2.5-VL-3B-Instruct | Hugging Face revision `66285546d2b821cf421d4f5eb2576359d3770cd3` | CoZ prompt VLM; Qwen Research License Agreement with non-commercial research/evaluation terms |
| SR LoRA `model_20001.pkl` | `697d3f9ab69a222006ca3ae48503cf057774c8142646301e9bba90e58242e47e` | Tracked in CoZ; no separate checkpoint model card found |
| SR VAE `vae_encoder_20001.pt` | `ed7f7aa03dfcbce9016d51c5aa8d3920428b3d7c9a678c721cd062d01805ae4a` | Tracked in CoZ; no separate checkpoint model card found |
| VLM adapter config | `66ebc2c4fcfe472c503814f7440b5e4bde2a2e4d197a52495e42df8dca69017e` | Must remain paired with the pinned adapter |
| VLM adapter weights | `1ba6fc24e76e0e078ceb6e067c49bcfbe86cb0d8995add1515172d922fba2ebd` | Tracked in CoZ; no separate checkpoint model card found |
| Optional DAPE adapter | `a7028be2edcbe9ab0bd1c4ab6f2a2a86f4b44d32261a4faa50ae10fdd9b2feba` | Not used by the ScaleGuard VLM prompt profile |

The tracked `ckpt/RAM/RAM.pth` object at the pinned CoZ commit is an empty file
(SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`);
it is not treated as a usable checkpoint or a runtime dependency.

### 4KAgent and controller-side artifacts

| Artifact | Immutable identity | Audit treatment |
| --- | --- | --- |
| Qwen2.5-VL-7B-Instruct | Hugging Face revision [`cc594898137f460bfe9f0759e9844b3ce807cfb5`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/cc594898137f460bfe9f0759e9844b3ce807cfb5) | Local 4KAgent perception VLM used to derive the HPS prompt; the model card at this revision declares Apache-2.0 |
| 4KAgent toolbox archive | Hugging Face revision `296428714177474f40a841a1677a935b7ea3bc9c`; archive SHA-256 `5caae37bbf06d2690d93974d80bfed42c55c4ad89aead5aa455420687b417c7e` | Repository card says Apache-2.0, but the archive aggregates expert checkpoints; whole archive remains `NOASSERTION` |
| Vicuna 7B v1.5 | Hugging Face revision `3321f76e3f527bd14065daf69dad9344000a201d` | Llama 2 Community License |
| DepictQA2 DQ495K | Hugging Face revision `915fee700b9f41c8be7da5d5b83da8a71dfc332b`; `ckpt.pt` SHA-256 `4494778f6045effa9252e145184e0cd5d2b184de3bd7be37849f81558dea2f56` | Model card declares Apache-2.0 |
| CLIP ViT-L/14 | Content-addressed URL and SHA-256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836` | Weight terms remain `NOASSERTION` |
| DepictQA degradation delta | Pinned Google Drive object; no publisher digest | Required manual gate; receipt records a measured hash but cannot authenticate it |
| HPS v2.1 | Hugging Face revision `697403c78157020a1ae59d23f111aa58ced35b0a`; file SHA-256 `c57a38fb4a2f7e7c15bf00da2ea377cdf165448b4dd1052a484c215a998c9837` | Model card declares Apache-2.0 |
| MUSIQ KonIQ weight | Hugging Face revision `0df2df423c65f6a64209309695f3845727431027`; file SHA-256 `e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa` | Weight repository is CC-BY-NC-SA-4.0 |
| LPIPS v0.1 Alex linear layer (optional) | Same immutable Hugging Face revision; LFS SHA-256 `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0` | Weight repository is CC-BY-NC-SA-4.0; separate AlexNet backbone remains user-provided and hash-bound per receipt |
| OpenAI CLIP RN50 for CLIPIQA (optional) | Content-addressed official URL and SHA-256 `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762` | CLIP code is MIT; model-weight terms remain `NOASSERTION` |

The 7B and 3B Qwen artifacts are separate locked model releases with different
roles and license declarations. Apache-2.0 on the 7B model card must not be
carried over to the 3B CoZ dependency or its adapters.

The first two model destinations are deliberately
`models/stabilityai/stable-diffusion-3-medium-diffusers` and
`models/Qwen/Qwen2.5-VL-3B-Instruct` relative to the downloader's weight root.
Bootstrap exposes that root through the repository `weights` link, matching the
production configuration.

## Optional PyIQA boundary

ScaleGuard's core and CPU mock do not require PyIQA. The research metric extra
pins PyIQA 0.1.16. The PyPI release is locked by:

- wheel SHA-256
  `42bbcf1c274a8b53f040a3ec5ec304a15b7725afb5a85b81050e9cc7e6d09ed4`;
- source distribution SHA-256
  `80fb913302585a774875c224b4f95068ac17bebd88f735876e6f0c3bb0988d34`;
  and
- tag target commit
  [`18dd7a19`](https://github.com/chaofengc/IQA-PyTorch/commit/18dd7a19694e94aac21019170e3f5e63d6b4e19e).

At that source state, the repository
[LICENSE](https://github.com/chaofengc/IQA-PyTorch/blob/18dd7a19694e94aac21019170e3f5e63d6b4e19e/LICENSE)
is PolyForm Noncommercial 1.0.0, and the project states that the
[NTU S-Lab License](https://github.com/chaofengc/IQA-PyTorch/blob/18dd7a19694e94aac21019170e3f5e63d6b4e19e/LICENSE-S-Lab)
applies to identified components. Both impose non-commercial boundaries. The
selected MUSIQ and LPIPS weights add CC-BY-NC-SA-4.0.

These terms do not change the Apache-2.0 license of ScaleGuard's original
source, but they do constrain a runtime that installs or uses the optional
metric. ScaleGuard makes no claim that its complete research runtime is
commercially usable.

Package availability and metric direction handling do not calibrate a quality
threshold. Per [ADR 0003](adr/0003-gradient-proxy-cpu-contract-only.md), the
CPU gradient proxy is only a contract fixture, while any PyIQA research gate
still requires a declared validation split, preprocessing contract,
sensitivity analysis, and retained calibration evidence.

## Issue review and reproducibility implications

All issues below were open at the audit date.

- [4KAgent pull request 13](https://github.com/taco-group/4KAgent/pull/13)
  proposes a MiniMax provider and remains open. ScaleGuard does not import that
  branch or introduce MiniMax as another runtime dependency. The canonical Qwen
  scheduler lives in the repository-owned overlay; the locked upstream commit
  and root tree remain unchanged.

- [4KAgent issue 9](https://github.com/taco-group/4KAgent/issues/9) reports
  reproduction gaps and plan differences. A
  [maintainer response](https://github.com/taco-group/4KAgent/issues/9#issuecomment-3821889945)
  identifies perception/planning variability, incomplete toolboxes, environment
  and checkpoint differences, rollback thresholds, evaluation scripts, and
  PyIQA version 0.1.13 as possible causes. A later
  [maintainer response](https://github.com/taco-group/4KAgent/issues/9#issuecomment-3857978011)
  identifies PSNR/SSIM channel-setting misalignment and says the evaluation
  code/results will be revised. ScaleGuard therefore freezes its own package
  and weight identities, records plans, and defines color space and metric
  preprocessing explicitly. Its PyIQA 0.1.16 controller study must not be
  presented as a direct reproduction of an upstream 0.1.13 score.

- [4KAgent issue 10](https://github.com/taco-group/4KAgent/issues/10) and
  [issue 12](https://github.com/taco-group/4KAgent/issues/12) report Python and
  package compatibility conflicts in the specialist-tool environment. This
  supports isolated environments and recorded dependency locks rather than a
  merged ScaleGuard/4KAgent/CoZ environment.

- [4KAgent issue 11](https://github.com/taco-group/4KAgent/issues/11) reports
  missing or incomplete remote-sensing evaluation sets. No ScaleGuard dataset
  result may be published until the exact authorized files and split manifest
  are hashed and retained.

- [CoZ issue 11](https://github.com/bryanswkim/Chain-of-Zoom/issues/11) asks
  that checkpoints be hosted in dedicated model repositories rather than
  application storage. The pinned CoZ commit still tracks small checkpoints
  without separate model cards, so the lock records their bytes as
  `NOASSERTION`; a content hash is not a license grant.

- [CoZ issue 12](https://github.com/bryanswkim/Chain-of-Zoom/issues/12) reports
  Windows installation conflicts. ScaleGuard's public GPU reproduction target
  is a recorded Linux environment. Community workarounds in that thread are
  not treated as upstream-supported dependency resolutions.

- [CoZ issue 13](https://github.com/bryanswkim/Chain-of-Zoom/issues/13) asked
  for full-image tiling before the later full-image code appeared. It is useful
  history, not evidence that the pinned commit lacks full-image inference.

- [CoZ issue 19](https://github.com/bryanswkim/Chain-of-Zoom/issues/19) still
  requests the unreleased GRPO training code. ScaleGuard performs inference
  integration only and makes no claim that CoZ training or preference alignment
  is reproducible from the public repository.

- AgenticIR
  [issue 8](https://github.com/Kaiwen-Zhu/AgenticIR/issues/8#issuecomment-3854573189)
  records a maintainer statement that the released DepictQA differs from the
  internal paper version. This is a lineage reproducibility caveat only;
  AgenticIR is not added to the ScaleGuard runtime.

## License boundary

The repository's Apache-2.0 license covers original ScaleGuard source only.
4KAgent, CoZ, embedded expert tools, optional packages, model weights, data,
and outputs retain their own terms. `NOASSERTION` means the audit did not find
enough publisher metadata to state a license; it does not mean public-domain,
Apache-2.0, or automatically prohibited.

`NOTICE` records the material attributions and the restrictive model/package
boundaries. Any distribution containing upstream checkouts, toolbox contents,
weights, or datasets needs a separate component-level review. ScaleGuard does
not vendor those artifacts.

## Remaining external evidence gates

The static repository can be validated without these actions, but a public GPU
reproduction claim cannot:

1. accept the Stable Diffusion 3 gated model terms and provide Hugging Face
   authentication on the target machine;
2. manually obtain the pinned DepictQA degradation delta and preserve the
   downloader receipt noting the absence of a publisher digest;
3. provision the declared GPU host and run component, integration, controller,
   and ablation protocols while retaining logs and file inventories;
4. provide authorized, hashed evaluation datasets and split manifests; and
5. calibrate controller thresholds on a declared validation split before
   reporting a research-quality gate.

Until those artifacts exist, the highest defensible status is static/CPU
contract readiness. Upstream paper numbers, issue screenshots, mock results,
and source inspection cannot raise that status.
