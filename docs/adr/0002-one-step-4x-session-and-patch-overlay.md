# ADR 0002: Expose one explicit 4× CoZ step through a small patch overlay

- Status: Accepted
- Date: 2026-07-27

## Context

The pinned CoZ full-image script owns the complete recursion loop. With
`rec_num=2`, a caller receives the final 16× result only after both 4× steps
have run; it cannot accept the first scale and reject the second. Re-launching
the script for every step restores control but reloads SD3 and Qwen each time.

The pinned full-image path also has three blocking contract defects:

1. its parsed model path is not passed to `SD3Euler`;
2. `OSEDiff_SD3_TEST_TILE.create_full_latent` reads a scheduler timestep
   without initializing the one-step schedule; and
3. processor tensors for patch prompting are not moved to the VLM's device.

Loading the VAE state with unrestricted `torch.load` is unnecessary for the
expected state-dict artifact. The full-image path also retains every processed
latent tile until a second accumulation loop, even though Gaussian blending can
be accumulated in the first loop without changing the intended fusion rule.

## Decision

ScaleGuard wraps the audited CoZ implementation in a persistent JSONL session.
Model components load once. Every `upscale` request performs exactly one
explicit 4× transition, returns one canonical PNG and metadata, and leaves the
next transition to the Trusted Scale Controller.

The original low-resolution image, normalized for VLM input, remains the
semantic anchor for every scale. CoZ continues to generate full-image and
patch-aware prompts and to use its existing latent tiling and Gaussian overlap
fusion. Temporary patch images stay private to the worker and are removed
after the scale result is produced.

We apply two ordered, content-hashed patches to the pinned checkout. The first,
`third_party/patches/chain-of-zoom/0001-full-inference-contract.patch`, is
deliberately limited to:

- `torch.load(..., weights_only=True)` for the VAE state dict;
- `scheduler.set_timesteps(1, device=device)` before reading the timestep; and
- moving VLM processor output to the device of the VLM parameters.

The second,
`third_party/patches/chain-of-zoom/0002-streaming-gaussian-blend.patch`,
preserves the existing Gaussian overlap rule but accumulates and releases each
processed tile immediately. It removes the unused unweighted full-latent
accumulator.

The model path and deterministic seeding are supplied by the external session
overlay, not by rewriting the upstream CLI.

## Consequences

- Scale recursion is observable and interruptible after every 4× result.
- Persistent mode avoids mandatory reloads between accepted scales; one-shot
  mode remains available for isolation and recovery.
- The upstream commit remains immutable and both patches are content-hashed in
  the upstream lock.
- Streaming fusion avoids retaining a list of every processed latent tile. It
  does not make total memory independent of image size: full latent and
  accumulation tensors remain resident.
- This design retains the pinned implementation's FP32 transformer/VAE
  placement in the full-image path. ScaleGuard accepts only an explicit `fp32`
  label for this path and records requested precision plus actual component
  placement.
- Peak memory, speed, determinism, seams, and image quality remain GPU
  validation items. Static inspection does not establish any of them.

## Evidence

The worker and patch are source-audited and exercised through CPU process
contracts where possible. No GPU inference result is claimed.
