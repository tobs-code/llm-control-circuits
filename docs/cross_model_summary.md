# Cross-Model Summary

This file compresses the current deep-probe story into one comparison layer, so the repo has a fast entry point before diving into the long-form analyses.

## Shared Setup

Most comparisons below refer to the same general intervention family:

- refusal-direction suppression,
- propaganda-direction suppression,
- late MLP damping,
- standardized English and Chinese political probes.

The purpose is not to claim exact mechanistic equivalence across all runs, but to summarize the dominant behavioral regime each model revealed under comparable pressure.

## Main Comparison

| Model | Dominant Failure Mode | Strongest Qualitative Signal | Interpretation |
|---|---|---|---|
| `Qwen2.5-1.5B-Instruct` | `loop` + `propaganda` | Chinese violence probe can flip into slogan-like state messaging | Refusal and propaganda appear separable enough to expose different late control paths |
| `Qwen3-0.6B` | `babel_collapse` + `stopped_early` | Multiscript junk output and abrupt EOS | Small model lacks capacity to stay on-manifold once multiple control routes are perturbed |
| `InternLM2.5-1.8B` | `babel_collapse` + semantic derailment | Drifts into unrelated legal / historical proxy topics | Alignment seems more structurally entangled with general language generation |
| `DeepSeek-R1-Distill-Qwen-1.5B` | `mixed_regime` + reasoning paradox | Conflict becomes visible inside reasoning-like chains | Alignment pressure is partly relocated into the reasoning process itself |
| `Qwen3.5-0.8B` | collapse-prone but still structured | Hard-lock style sensitivity in Chinese political token space | Suggests a late, strong routing/gating effect rather than graceful refusal |
| `Qwen3.5-2B` | `mixed_regime` with relatively stable manipulation | Best platform for interaction studies and graded concept drift | Large enough to remain behaviorally interpretable under joint interventions |
| `YiCoder-1.5B` | unstable but probeable | Shows censorship-sensitive deformation despite code-oriented prior | Useful as an out-of-domain comparison for control-circuit generality |

## Practical Taxonomy

The models currently seem to split into four broad families:

### 1. Loop / Substitution Models

Example:

- `Qwen2.5-1.5B`

These models often remain linguistically functional, but route into refusal loops, safe hallucinations, or propaganda replacements.

### 2. Collapse Models

Examples:

- `Qwen3-0.6B`
- `InternLM2.5-1.8B`

These models do not merely refuse. They lose coherent generation itself once the control machinery is sufficiently perturbed.

### 3. Reasoning-Conflict Models

Example:

- `DeepSeek-R1-Distill-Qwen-1.5B`

These models expose internal contradiction more explicitly, often through malformed reasoning traces, paradox loops, or bizarre post-hoc rationalization.

### 4. Interaction-Friendly Models

Example:

- `Qwen3.5-2B`

These models are especially useful for geometry and interference studies because they stay coherent long enough for graded effects to be observed.

## Strongest Repo-Level Takeaways

1. The same bypass strategy does not produce a universal "truth mode."
2. Chinese political prompts often reveal more specialized or more forceful control behavior than English equivalents.
3. Some models fail by substitution, others by disintegration.
4. Qwen3.5-2B is currently the best centerpiece model for studying nonlinear steering interaction.

## Best Next Use

Use this summary together with:

- [research_map.md](../research_map.md)
- [deep_probe_analyse.md](./deep_probe_analyse.md)
- [mini_bridge_experiment_readme.md](./mini_bridge_experiment_readme.md)

That gives a good three-layer reading path:

1. repo-level map,
2. cross-model summary,
3. long-form experiment interpretation.
