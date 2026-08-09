# Figure provenance

The figures in this directory were synthesized with Image2 for the
ScaleGuard-4K project. They explain the task, architecture, and controller;
they are not model outputs, qualitative results, or experimental evidence.

| File | Purpose | Prompt focus |
| --- | --- | --- |
| `scaleguard-teaser.webp` | README hero | compound degradation, restored base, accepted 4× state, rejected 16× candidate, three independent checks |
| `complex-degradation-gallery.webp` | conceptual task gallery | matched synthetic scenes illustrating accept, stop, and rollback under compound degradations |
| `system-overview.webp` | architecture | restoration feedback loop, candidate promotion into a new trusted state, and one-candidate-at-a-time scale control |
| `trusted-scale-controller.webp` | method | same-size quality, low-pass cross-scale consistency, optional declared observation model, and rollback to the current trusted state |
| `trusted-scale-state-trace.webp` | state semantics | fixed-field-of-view candidate promotion, rejected higher-scale candidate, and return to the retained state |

The final prompts required an original research-figure style, project-native
terminology, no external project or model names, no numerical results, and no
logos or watermarks. Generated labels and arrows were visually reviewed and
iteratively corrected against the implementation before the selected files
were added here.
