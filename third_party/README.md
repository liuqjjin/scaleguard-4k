# Audited upstream boundary

ScaleGuard-4K has exactly two core algorithm upstreams:
[4KAgent](https://github.com/taco-group/4KAgent) and
[Chain-of-Zoom](https://github.com/bryanswkim/Chain-of-Zoom). Their source is
not vendored in this repository. Local checkouts live below
`third_party/checkouts/`, which is ignored by Git.

AgenticIR is acknowledged as 4KAgent's code and design lineage. It is not a
ScaleGuard runtime, checkout, or third core project. Do not add an AgenticIR
checkout to this directory.

## Materialize the two pinned checkouts

Run these commands from the ScaleGuard-4K repository root. They resolve
immutable commits rather than a branch or release alias:

```bash
mkdir -p third_party/checkouts

git clone --no-checkout \
  https://github.com/taco-group/4KAgent.git \
  third_party/checkouts/4KAgent
git -C third_party/checkouts/4KAgent fetch --depth 1 origin \
  04179ffed12c1db64890a48ac6d51ae4abeaee37
git -C third_party/checkouts/4KAgent checkout --detach \
  04179ffed12c1db64890a48ac6d51ae4abeaee37

git clone --no-checkout \
  https://github.com/bryanswkim/Chain-of-Zoom.git \
  third_party/checkouts/Chain-of-Zoom
git -C third_party/checkouts/Chain-of-Zoom fetch --depth 1 origin \
  5a4f351c9c655568c4705d5fc53ef70a3f28905f
git -C third_party/checkouts/Chain-of-Zoom checkout --detach \
  5a4f351c9c655568c4705d5fc53ef70a3f28905f
```

The expected commit and root tree pairs are:

| Checkout | Commit | Root tree |
| --- | --- | --- |
| 4KAgent | `04179ffed12c1db64890a48ac6d51ae4abeaee37` | `66f62367c915042ef945562bddc791bfc445fb42` |
| Chain-of-Zoom | `5a4f351c9c655568c4705d5fc53ef70a3f28905f` | `177741fa5443a5073f29b1cf3ffad67e596d0b5d` |

Confirm each pair before applying a patch:

```bash
git -C third_party/checkouts/4KAgent rev-parse HEAD HEAD^{tree}
git -C third_party/checkouts/Chain-of-Zoom rev-parse HEAD HEAD^{tree}
```

## Apply the locked CoZ patches

The only upstream source patches are two small, ordered CoZ changes. Their
locked SHA-256 values are:

```text
0001-full-inference-contract.patch
  b9f2b6c293b8ac7e48be8188318992f16cee9f6f9affa2ccf0f41307027d00a0
0002-streaming-gaussian-blend.patch
  446c82cfdeb9e822814292511bdf80b7bd0401c08a54d82d7d29d4983901ca78
patched osediff_sd3.py
  c5aea528f9f1206cae6ca666b5d0907a1b18e701300ce1252a9468d682fa6084
```

Verify and apply them once, in numeric order:

```bash
python3 -c 'import hashlib,pathlib; [print(p.name, hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(pathlib.Path("third_party/patches/chain-of-zoom").glob("*.patch"))]'
git -C third_party/checkouts/Chain-of-Zoom apply --check \
  ../../patches/chain-of-zoom/0001-full-inference-contract.patch
git -C third_party/checkouts/Chain-of-Zoom apply \
  ../../patches/chain-of-zoom/0001-full-inference-contract.patch
git -C third_party/checkouts/Chain-of-Zoom apply --check \
  ../../patches/chain-of-zoom/0002-streaming-gaussian-blend.patch
git -C third_party/checkouts/Chain-of-Zoom apply \
  ../../patches/chain-of-zoom/0002-streaming-gaussian-blend.patch
git -C third_party/checkouts/Chain-of-Zoom apply --reverse --check \
  ../../patches/chain-of-zoom/0001-full-inference-contract.patch
git -C third_party/checkouts/Chain-of-Zoom apply --reverse --check \
  ../../patches/chain-of-zoom/0002-streaming-gaussian-blend.patch
```

Both patches modify only `osediff_sd3.py`. The first uses restricted state-dict
loading, initializes the one-step scheduler, and moves VLM processor tensors to
the VLM device. The second preserves CoZ's Gaussian overlap fusion while
accumulating each latent tile immediately instead of retaining every output
tile until the end. Neither replaces CoZ inference or adds another model.

Finally run the repository verifier:

```bash
uv run --locked scaleguard upstream verify --lock upstream-lock.yaml
```

The verifier checks commits, root trees, patch content, application state, and
the final hash of every patch-modified file. It rejects unrelated checkout
edits and extra edits hidden inside an allowed file. A passing static
verification is not a GPU reproduction result.

## Overlays and runtime files

Files below `overlays/` are small ScaleGuard-owned adapters:

- `overlays/4kagent/run_native_restoration.py` keeps 4KAgent's restoration and
  reflection path while reserving terminal generative SR for ScaleGuard.
- `overlays/chain-of-zoom/coz_session_worker.py` exposes one explicit 4× CoZ
  transition at a time in a persistent session.

They import the pinned checkouts at runtime; they are not forks or copied
upstream implementations. See the ADRs in `docs/adr/` for the boundary and
its validation limits.

Model acquisition is independent of source checkout materialization. The
audited model revisions and digests are in `weights-lock.json`; the downloader
stores them below its selected weight root and writes a content receipt.
CoZ's checkout may contain small tracked checkpoint files, but the audited
runtime does not select them. Runtime paths are bound to the separately
materialized weight root, and the preflight verifies those exact files against
the download and materialization receipts.

Upstream code, embedded tools, model weights, and datasets retain their own
licenses. The complete Chain-of-Zoom MIT text is retained in
`licenses/Chain-of-Zoom-MIT.txt`; review `NOTICE` before installing a runtime
or redistributing any artifact.
