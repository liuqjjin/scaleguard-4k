# ADR 0007: Isolate unpatched legacy dependency boundaries

- Status: Accepted
- Date: 2026-07-27

## Context

The selected 4KAgent runtime still reaches two packages with published
advisories and no fixed release:

- BasicSR 1.4.2 can execute a command when its SLURM helper receives a crafted
  `SLURM_NODELIST`.
- DiskCache 5.6.3 deserializes pickle data from its cache. Outlines 0.2.1
  imports DiskCache while constructing the structured-generation integration
  used by 4KAgent.

Replacing 4KAgent's restoration toolbox or structured-generation layer would
be a larger algorithmic fork. Merely suppressing the advisories would leave an
implicit trust boundary.

## Decision

ScaleGuard keeps those exact upstream dependencies and narrows their inputs:

- both the outer worker environment and 4KAgent's shell-free tool runner omit
  `SLURM_NODELIST` and all other ambient scheduler variables;
- each 4KAgent run creates a new `0700` Outlines cache below its private worker
  directory;
- the overlay verifies that cache path is absolute, non-symlinked, and private,
  then disables Outlines caching immediately after the required import; and
- CI audits every resolved GPU lock. Only `PYSEC-2026-1215` and
  `PYSEC-2026-2447` are ignored, by identifier, for the 4KAgent lock. Any new
  advisory fails the check.

## Consequences

- A shared or attacker-writable cache is not accepted by the audited path.
- A process running as the same OS account can still modify files it owns.
  Real inference therefore remains restricted to a dedicated account or
  disposable AutoDL host with no unrelated secrets.
- The two ignored scanner records are documented mitigations, not claims that
  the upstream packages were fixed.
- A future compatible fixed release should remove the corresponding isolation
  exception and exact scanner ignore.
