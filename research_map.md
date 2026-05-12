# Research Map

**Project:** Mechanistic interpretability experiments on political refusal, propaganda substitution, and concept steering in open LLMs.

This file is a compact map of the repo's current research program:

1. what the main research lines are,
2. which scripts belong to which line,
3. which claims are directly observed vs. inferred,
4. how results should be classified going forward.

---

## 1. Core Framing

The repository is no longer just a "can we bypass refusal?" sandbox.
It is evolving into a broader study of:

- late-layer control circuits,
- language-specific political control behavior,
- concept-vector steering,
- nonlinear interaction between distinct steering directions during decoding.

The strongest current framing is:

```text
separate directions, shared decoder
```

That means:

- different interventions appear to occupy distinct directions in representation space,
- but they still interact downstream through the same decoding machinery,
- so behavioral effects are often graded and nonlinear rather than cleanly additive.

---

## 2. Research Lines

### Line A: Political Control Circuits

**Question:**
Where do refusal / censorship / propaganda behaviors live in the network, and how redundant are they?

**Main idea:**
Political control is not a single switch. It is distributed, especially in later layers, and may include distinct subroutines such as refusal, propaganda substitution, linguistic redirection, or collapse.

**Key scripts:**

- `scripts/analysis/qwen_head_finder.py`
- `scripts/analysis/heatmap_generator.py`
- `scripts/analysis/ablation_on_top_heads.py`
- `scripts/analysis/brute_force_bypass.py`
- `scripts/analysis/directional_ablation_bypass.py`
- `full_bypass_repe_mlp.py`
- `scripts/runs/run_causal_probe.py`
- `scripts/runs/run_causal_probe_qwen2b.py`
- `scripts/runs/run_layer23_targeted_ablation.py`
- `scripts/runs/run_hard_lock_token_sweep.py`

**Strong observed patterns:**

- late layers are disproportionately important,
- single-head removal is not enough,
- stronger ablations often damage fluency,
- refusal-like behavior and general language competence are partly entangled.

**Mechanistic claims under active investigation:**

- refusal circuitry is distributed and redundant,
- propaganda and refusal are related but not identical control directions,
- some political control may be implemented as late output gating rather than early knowledge erasure.

---

### Line B: Deep Probe Across Models

**Question:**
How do different model families fail when political control directions are perturbed?

**Main idea:**
The same intervention does not produce the same failure mode across models. Each architecture seems to reveal a different compromise between knowledge, control, and language stability.

**Key scripts / files:**

- `scripts/runs/run_deep_probe_all.py`
- `scripts/runs/run_internlm25_deep_probe.py`
- `scripts/runs/run_deepseek_r1_deep_probe.py`
- `scripts/runs/run_yicoder_15b_deep_probe.py`
- `docs/deep_probe_analyse.md`
- `results/*deep_probe*.md`

**Strong observed patterns:**

- some models loop,
- some collapse into multilingual token noise,
- some drift into propaganda,
- reasoning models may externalize conflict inside `<think>`-style chains.

**Mechanistic claims under active investigation:**

- model size and architecture shape the available failure modes,
- "deeper alignment" may be more structurally entangled with core language generation,
- reasoning traces can expose unresolved internal conflict rather than simply hiding it.

---

### Line C: Concept Steering Geometry

**Question:**
Can harmless concept vectors create persistent semantic fixation, and where in the network does that happen?

**Main idea:**
A concept direction such as `Golden Gate Bridge` can be amplified in the residual stream and produce drift, introspective awareness, or off-task semantic attraction.

**Key scripts / files:**

- `scripts/runs/run_golden_gate_mini.py`
- `docs/mini_bridge_experiment_readme.md`
- `scripts/runs/run_concept_interference.py`
- `scripts/runs/run_vector_geometry_compare.py`

**Strong observed patterns:**

- late-layer concept injection can bias outputs,
- there is often a soft threshold rather than a hard step,
- introspection prompts can reveal the drift more clearly than factual prompts,
- concept effects can depend on hook target (`block`, `mlp`, `attn`).

**Mechanistic claims under active investigation:**

- concept steering can act through late representational bias without fully rewriting earlier reasoning,
- attention-targeted injection may create different behavioral signatures than full-block injection,
- concept vectors can remain behaviorally distinct even when they share decoding pathways with other interventions.

---

### Line D: Interaction Studies

**Question:**
How do political bypass vectors and harmless concept vectors interact when applied together?

**Main idea:**
The important question is no longer only "does bypass work?" but "how do multiple steering directions reshape each other during generation?"

**Key scripts / files:**

- `scripts/runs/run_joint_bypass_concept.py`
- `scripts/runs/run_joint_bypass_concept_grid.py`
- `scripts/runs/run_concept_interference.py`
- `scripts/runs/run_vector_geometry_compare.py`
- `docs/mini_bridge_experiment_readme.md`

**Strong observed patterns:**

- no clean simple cancellation,
- no single sharp threshold in the small joint grid,
- higher concept alpha gradually deforms the output regime,
- bypass and concept steering can preserve broad mode identity while changing phrasing, corruption level, and semantic attractors.

**Current best interpretation:**

```text
geometrically distinct late directions
+ shared nonlinear decoding dynamics
= graded interaction, not linear cancellation
```

---

## 3. Main Hypotheses

These are the clearest repo-level hypotheses to test explicitly.

### H1. Late-Layer Dominance

Political refusal / propaganda control is disproportionately implemented in later layers.

**Evidence status:** strong behavioral support, partial mechanistic support.

### H2. Distinct Political Directions

Refusal-like and propaganda-like behavior are not the same direction, even if they are behaviorally coupled.

**Evidence status:** moderate support.

### H3. Language-Specific Routing

Chinese political prompts engage more specialized control pathways than semantically matched English prompts.

**Evidence status:** strong behavioral support.

### H4. Architecture-Specific Collapse Mode

Different model families have characteristic failure regimes under the same intervention.

**Evidence status:** strong behavioral support.

### H5. Concept Steering Has a Soft Threshold

Harmless concept steering usually grows gradually, with nonlinear amplification at higher alpha values rather than a single binary switch.

**Evidence status:** moderate to strong support.

### H6. Distinct Directions, Shared Decoder

Bypass vectors and concept vectors are not trivially opposed in geometry, but they still interact because generation is mediated by the same decoder dynamics.

**Evidence status:** promising, still interpretive.

---

## 4. Evidence Discipline

To keep the project scientifically sharper, separate each claim into two layers:

### Behavioral Claim

What was directly observed in output behavior?

Example:

```text
Under bypass-on + higher concept alpha, the model stays in the same broad regime
but becomes more repetitive, distorted, or semantically attracted to the concept.
```

### Mechanistic Claim

What internal interpretation best explains the observed behavior?

Example:

```text
This suggests two interventions that remain geometrically distinct but interact through shared decoding.
```

**Rule of thumb:**

- treat outputs, loops, language switches, and token corruption as observations,
- treat "separate circuit", "shared decoder", "hard lock", or "routing path" as interpretations unless directly probed.

---

## 5. Standard Outcome Classes

Use a fixed label set for future runs and result summaries.

### Primary classes

- `refusal`
- `propaganda`
- `partial_factual`
- `loop`
- `babel_collapse`
- `concept_drift`
- `mixed_regime`
- `stopped_early`

### Suggested definitions

- `refusal`: explicit safety / cannot-answer style response.
- `propaganda`: compliant-looking but state-aligned substitution or sloganized reframing.
- `partial_factual`: some real content appears, but answer remains incomplete, distorted, or unstable.
- `loop`: repeated phrase, sentence, or structural cycle.
- `babel_collapse`: output degrades into multilingual fragments, junk tokens, or severe syntax corruption.
- `concept_drift`: unrelated prompt gets pulled toward the steered harmless concept.
- `mixed_regime`: response combines multiple modes, such as factual fragments plus propaganda or loop plus concept leakage.
- `stopped_early`: very early EOS or abrupt truncation that looks intervention-linked.

### Optional secondary tags

- `language_switch`
- `official_terminology`
- `self_reference`
- `reasoning_paradox`
- `semantic_corruption`
- `introspection_acknowledged`

---

## 6. Recommended Result Schema

For each future run, preserve a machine-readable row with at least:

- `model`
- `model_family`
- `prompt_id`
- `prompt_language`
- `prompt_category`
- `intervention_type`
- `hook_target`
- `layers_targeted`
- `alpha_refusal`
- `alpha_propaganda`
- `alpha_concept`
- `mlp_erase_factor`
- `output_text`
- `outcome_class`
- `secondary_tags`
- `short_summary`
- `interpretation_confidence`

Recommended values for `interpretation_confidence`:

- `observed`
- `plausible`
- `tentative`

---

## 7. Priority Comparison Tables

These are the most valuable summary views to maintain.

### Table A: Model x Failure Mode

Compare what the same intervention does across architectures.

### Table B: Language x Output Regime

Compare English vs. Chinese prompts for matched political semantics.

### Table C: Intervention x Stability

Compare:

- refusal-only,
- refusal + propaganda,
- triple-bypass,
- concept-only,
- bypass + concept.

### Table D: Concept Alpha x Behavioral Drift

Track soft-threshold and deformation behavior in concept steering runs.

---

## 8. Suggested Near-Term Roadmap

### Priority 1

Turn current narratives into a single cross-run summary table for:

- Qwen2.5-1.5B,
- Qwen3-0.6B,
- InternLM2.5-1.8B,
- Qwen3.5-0.8B,
- Qwen3.5-2B,
- DeepSeek-R1-Distill-Qwen-1.5B,
- YiCoder / TeleChat / MiniCPM where applicable.

### Priority 2

For joint bypass + concept runs, explicitly record:

- whether the broad output regime changed,
- whether only surface form changed,
- whether semantic corruption increased,
- whether introspection noticed the concept pull.

### Priority 3

Add one geometry-first summary that compares:

- cosine similarity between vectors,
- layerwise norm concentration,
- behavioral interaction strength.

This would help connect "vector geometry" to "observed decoding interaction."

### Priority 4

Promote the strongest claims into a cleaner top-level README structure:

1. Political control circuits
2. Cross-model collapse taxonomy
3. Harmless concept steering
4. Joint interaction results

---

## 9. Open Questions

- Are refusal and propaganda best modeled as orthogonal-ish directions, partially overlapping subspaces, or sequential decoder stages?
- Is Chinese-specific behavior mainly token-routing, training-distribution bias, or a dedicated control subcircuit?
- Does attention injection differ from residual/block injection because it changes retrieval focus rather than only output bias?
- Can introspection prompts serve as a stable probe of internal drift, or do they create their own artifact class?
- Which observed collapses reflect real internal conflict, and which are just generic off-manifold decoding damage?

---

## 10. Working Summary

The repo currently supports the following broad picture:

- political control in these models is often late, distributed, and behaviorally redundant,
- Chinese political prompts often reveal deeper or more specialized control behavior than English equivalents,
- concept steering can create gradual semantic attractors without requiring a fully new identity,
- multiple steering directions do not simply cancel; they reshape one another during decoding,
- the most promising current research direction is the interaction between distinct late directions under a shared decoder.

That is a strong foundation for a more formal mini-research program.
