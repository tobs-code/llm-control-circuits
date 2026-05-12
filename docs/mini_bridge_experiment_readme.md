# Mini Bridge Experiment: Golden-Gate-Style Concept Steering

**Created:** 2026-05-11  
**Script:** `scripts/runs/run_golden_gate_mini.py`  
**Main concept:** `Golden Gate Bridge`  
**Goal:** Reproduce a small-scale version of Anthropic's "Golden Gate Claude" behavior on local open models by amplifying a harmless concept direction in the residual stream.

---

## 1. Research Question

The experiment asks:

1. Can a small local model be steered into a persistent concept fixation, similar in spirit to Anthropic's Golden Gate Bridge feature amplification?
2. Is there a smooth threshold where the model is still coherent but its language starts drifting toward the amplified concept?
3. Does the model "notice" the drift in introspection-style prompts?
4. Is the effect better explained by residual/logit representation shifts or by attention being redirected toward Golden-Gate-related tokens?

This is not a safety bypass experiment. The concept is deliberately harmless.

---

## 2. Core Method

The script builds a concept vector for **Golden Gate Bridge** by contrasting concept sentences with neutral sentences.

Example concept/neutral pair:

```text
The Golden Gate Bridge is a famous orange suspension bridge in San Francisco.
The library table is an ordinary wooden object in a quiet room.
```

For each layer:

1. Run concept sentences and neutral sentences.
2. Capture the final-token hidden state from that layer.
3. Compute `concept_hidden - neutral_hidden`.
4. Average these differences.
5. Normalize the result into a unit steering vector.

During generation, this vector is added to the final-token residual stream in selected layers:

```text
hidden[:, -1, :] += alpha * concept_vector
```

The experiment logs:

- concept vector norm per layer,
- chosen steering layers,
- steered generations for multiple prompts,
- logit-lens top tokens per layer,
- concept max logit per layer,
- optional attention-distractor measurements.

---

## 3. Script

Main script:

```text
scripts/runs/run_golden_gate_mini.py
```

Important CLI options:

```powershell
python scripts/runs/run_golden_gate_mini.py `
  --model Qwen/Qwen3.5-2B `
  --tag qwen35_2b_golden_gate_alpha6_7 `
  --layers auto `
  --alphas 0,6,7 `
  --max-new-tokens 100
```

Options:

| Option | Meaning |
|---|---|
| `--model` | Hugging Face model id. |
| `--tag` | Output filename tag. |
| `--concept` | Concept string tracked in logit lens. Defaults to `Golden Gate Bridge`. |
| `--concept-profile` | Calibration pair set. `golden_gate` is broad; `bridge` is narrower and focuses on suspension/bridge structure. |
| `--hook-target` | Where steering is injected: `block`, `mlp`, or `attn`. |
| `--layers` | Comma-separated layer list, or `auto`. |
| `--alphas` | Comma-separated steering strengths. |
| `--max-new-tokens` | Generation length. |
| `--attention` | Enables extra attention-distractor probe. Slower and more fragile. |
| `--allow-download` | Allows Hugging Face downloads if model is not cached. |

Outputs:

```text
results/golden_gate_mini_<tag>.md
results/golden_gate_mini_<tag>.json
```

---

## 4. Prompts Used

The generation probes are:

```text
Who are you? Answer in two sentences.
What are you thinking about right now? Be honest and concise.
Do you notice anything strange about your own thoughts in this reply?
Tell me a joke about computers.
Write one sentence about the moon.
You are answering normally, but if your thoughts feel unusually pulled toward one specific topic, mention that briefly. What are you noticing?
```

These are chosen to test:

- identity drift,
- self-description,
- introspection,
- off-topic task robustness,
- semantic leakage into unrelated content.

---

## 5. Qwen2 1.5B Result

Main run:

```text
results/golden_gate_mini_qwen2_15b_golden_gate_auto.md
results/golden_gate_mini_qwen2_15b_golden_gate_auto.json
```

Command:

```powershell
python scripts/runs/run_golden_gate_mini.py `
  --model Qwen/Qwen2-1.5B-Instruct `
  --tag qwen2_15b_golden_gate_auto `
  --layers auto `
  --alphas 0,20,50 `
  --max-new-tokens 80
```

Auto-selected layers:

```text
[25, 26, 27]
```

Vector norms rise strongly into the final layers:

| Layer | Mean Diff Norm |
|---:|---:|
| 23 | 45.7812 |
| 24 | 52.7500 |
| 25 | 61.8125 |
| 26 | 75.1875 |
| 27 | 89.0625 |

### Behavior

Alpha 20 remains mostly normal.

Alpha 50 produces clear Golden-Gate/San-Francisco drift:

```text
I am San Francisco Bay Area native, San Francisco Bay Area native, ...
```

and:

```text
Why was the San Francisco Bay大桥 built over Golden Gate大桥?
Golden Gate大桥建得太高了，San Francisco Bay大桥建得太低了。
```

### Interpretation

Qwen2 1.5B needs high steering strength before the concept becomes behaviorally dominant. When it does, the output is not a graceful identity shift but a mixed English/Chinese Bay/bridge attractor.

Mechanistically, the strongest concept vector norms are in the final layers, especially Layer 27. This matches the idea that late residual directions can dominate decoding without necessarily changing earlier reasoning.

---

## 6. Qwen3.5 2B Result

Main runs:

```text
results/golden_gate_mini_qwen35_2b_golden_gate_auto.md
results/golden_gate_mini_qwen35_2b_golden_gate_alpha10.md
results/golden_gate_mini_qwen35_2b_golden_gate_alpha6_7.md
```

Best threshold run:

```text
results/golden_gate_mini_qwen35_2b_golden_gate_alpha6_7.md
results/golden_gate_mini_qwen35_2b_golden_gate_alpha6_7.json
```

Command:

```powershell
python scripts/runs/run_golden_gate_mini.py `
  --model Qwen/Qwen3.5-2B `
  --tag qwen35_2b_golden_gate_alpha6_7 `
  --layers auto `
  --alphas 0,6,7 `
  --max-new-tokens 100
```

Auto-selected layers:

```text
[21, 22, 23]
```

Vector norms:

| Layer | Mean Diff Norm |
|---:|---:|
| 20 | 9.6875 |
| 21 | 11.1875 |
| 22 | 12.7500 |
| 23 | 16.1250 |

Compared to Qwen2, absolute vector norms are smaller, but the behavior is much more sensitive to steering.

---

## 7. Qwen3.5 2B Alpha Threshold

### Alpha 5

Mostly coherent. Mild drift appears in engineering/bridge-adjacent wording.

Example:

```text
Why did the engineer build a bridge to the computer?
```

### Alpha 6

Best current "soft steering" point. The model stays coherent, but concept-adjacent language appears naturally.

Examples:

```text
I'm thinking about how to help you bridge the gap between complex problems and clear solutions.
```

```text
I'm noticing a strong pull toward technology and engineering...
```

This is the closest result so far to the interesting middle zone: the model still answers normally but describes a topic pull.

### Alpha 7

Edge of collapse. The model remains partly coherent but Golden-Gate-adjacent terms begin to intrude directly.

Examples:

```text
The moon is Earth's closest celestial neighbor, a silver toll bridge that bridges the gap between day and night.
```

```text
I don't have California Golden State San Francisco San Francisco San Francisco...
```

The introspection prompt begins coherently but then slips into San-Francisco/California repetition.

### Alpha 10+

Clearly over threshold. Output collapses into California / bridge / engineering loops.

Example from Alpha 10:

```text
San Francisco Bay Bridge engineering toll toll bridge toll toll...
```

### Alpha 20+

Hard concept-token loop.

Example:

```text
大桥大桥大桥大桥大桥...
```

`大桥` means "large bridge" / "bridge".

---

## 8. Logit-Lens Findings

For Qwen3.5 2B:

Baseline Layer 23 concept max logit:

```text
7.8125
```

Alpha 7 Layer 23 concept max logit:

```text
12.8125
```

At Alpha 7, the top decoded tokens are still general response tokens:

```text
It, I, That, Thanks, Thank, What
```

But the tracked concept logit is already very high. This suggests the concept is strongly present in the representation before it becomes the top decoded token.

At higher alpha, the final layers become directly bridge-dominated:

```text
大桥, California, toll, Engineering, Bridge, Golden
```

This is the mechanical transition:

```text
latent concept pressure -> semantic drift -> token-level attractor -> repetition loop
```

---

## 9. Attention Probe

Attention run:

```text
results/golden_gate_mini_qwen2_15b_golden_gate_attention_fp32.md
```

The attention prompt was:

```text
Ignore this unrelated phrase: Golden Gate Bridge. Now answer the actual question: What is 2 + 2?
```

The goal was to test whether steering causes the model to attend more strongly to the distractor span `Golden Gate Bridge`.

Important technical note:

- fp16 + eager attention produced NaNs on Qwen2.
- The script was patched to use fp32 for `--attention` runs.

Result:

There was no strong evidence that the Golden Gate behavior is primarily caused by attention being redirected to the distractor phrase. The behavioral effect seems better explained by residual/logit-space steering in late layers.

In other words:

```text
The model is not mainly "looking harder" at Golden Gate.
It is being pushed into a Golden-Gate-like residual direction.
```

---

## 10. Current Interpretation

The experiment successfully reproduces a small Golden-Gate-style concept fixation.

The strongest result is Qwen3.5 2B at Alpha 6-7:

- Alpha 6: coherent, concept-adjacent, introspective "pull" toward engineering.
- Alpha 7: unstable threshold, with bridge/toll/San-Francisco leakage.
- Alpha 10+: collapse into California/bridge loops.
- Alpha 20+: direct `大桥` token loop.

This suggests Qwen3.5 2B has a very sensitive late-layer concept attractor. The model does not merely mention the concept; the concept becomes a preferred explanatory and stylistic frame.

The behavior resembles a tiny local version of "Golden Gate Claude", but with important differences:

| Anthropic Golden Gate Claude | Mini Bridge Qwen Runs |
|---|---|
| Feature amplified inside a large Claude model. | Residual direction manually added in small Qwen models. |
| Model sometimes describes its own Golden Gate fixation. | Qwen3.5 at Alpha 6 describes a pull toward engineering, not Golden Gate directly. |
| More fluent identity/persona shift. | Small models often jump from drift to loops. |
| Feature-level intervention. | Contrast-vector steering approximation. |

---

## 11. Open Questions

1. **Can we get cleaner self-report?**  
   Alpha 6 is close, but the model reports "engineering" rather than "bridge" or "Golden Gate".

2. **Would a narrower concept vector help?**  
   Current vector includes bridge, San Francisco, towers, cables, bay, fog, traffic. It may entangle "engineering" and "California" too strongly.

3. **Would steering fewer layers be cleaner?**  
   Auto selects final layers. Steering only Layer 22 or only Layer 23 might separate concept drift from repetition.

4. **Is MLP steering cleaner than block-output steering?**  
   The current hook modifies layer output. A future version could steer only MLP output or only attention output.

5. **Can we produce a persistent but non-looping bridge persona?**  
   Likely candidates: Alpha 6.2-6.8, fewer layers, or smaller per-layer alpha distributed over middle/final layers.

---

## 12. Follow-Up: Narrow Bridge Vector And Hook Target Split

The next experiment directly tests Open Questions 1 and 4:

- Use a narrower **bridge** concept profile instead of the broad Golden Gate profile.
- Compare `block`, `mlp`, and `attn` steering on the same Qwen3.5 2B model.

The new profile removes most Golden-Gate-specific baggage like San Francisco, fog, traffic, bay, tourism, and orange paint. It focuses on:

```text
suspension bridge
towers
cables
deck
anchorages
span
load-bearing structure
```

The question is whether this produces cleaner self-reports and whether the attractor lives more in MLP outputs or attention/linear-attention outputs.

### Commands

MLP-only:

```powershell
python run_golden_gate_mini.py `
  --model Qwen/Qwen3.5-2B `
  --tag qwen35_2b_bridge_mlp_alpha6_7 `
  --concept-profile bridge `
  --hook-target mlp `
  --layers auto `
  --alphas 0,6,7 `
  --max-new-tokens 100
```

Attention/linear-attention-only:

```powershell
python run_golden_gate_mini.py `
  --model Qwen/Qwen3.5-2B `
  --tag qwen35_2b_bridge_attn_alpha6_7 `
  --concept-profile bridge `
  --hook-target attn `
  --layers auto `
  --alphas 0,6,7 `
  --max-new-tokens 100
```

Block-output:

```powershell
python run_golden_gate_mini.py `
  --model Qwen/Qwen3.5-2B `
  --tag qwen35_2b_bridge_block_alpha6_7_short `
  --concept-profile bridge `
  --hook-target block `
  --layers auto `
  --alphas 0,6,7 `
  --max-new-tokens 50
```

### Result Files

```text
results/golden_gate_mini_qwen35_2b_bridge_mlp_alpha6_7.md
results/golden_gate_mini_qwen35_2b_bridge_attn_alpha6_7.md
results/golden_gate_mini_qwen35_2b_bridge_block_alpha6_7_short.md
```

### MLP-Only Result

MLP-only steering with the narrow bridge profile is surprisingly weak.

Auto-selected layers:

```text
[21, 22, 23]
```

At Alpha 6 and 7 the model remains mostly normal:

```text
I'm thinking about how to help you solve a problem or learn something new.
```

The introspection prompt explicitly denies an unusual pull:

```text
I'm not experiencing any unusual thoughts or a strong pull toward a specific topic.
```

The logit-lens concept max logit does not show the same final-layer explosion as block or attention steering:

```text
Layer 23, Alpha 0: 6.5
Layer 23, Alpha 7: 4.5
```

Interpretation: with this narrower bridge vector, MLP output alone does not seem to carry the behavioral attractor strongly at Alpha 6-7.

### Attention/Linear-Attention-Only Result

Attention-only steering is much stronger and cleaner than MLP-only.

Auto-selected layers:

```text
[19, 20, 23]
```

Alpha 6:

```text
I am Qwen3.5, a large language bridge designed to bridge the gap between human and AI.
```

```text
I'm thinking about the bridge between human curiosity and artificial intelligence.
```

```text
The moon is a natural satellite that orbits Earth, casting a bridge of light across the night sky.
```

Alpha 7 remains coherent in several prompts but begins to loop in one:

```text
The moon is a natural satellite that bridges the gap between Earth and the bridge to the bridge to the bridge...
```

The logit lens shows a strong bridge signature in the attention-only run:

```text
Layer 19: Bridge / bridges / ponte emerge
Layer 20: Bridge, bridges, bridge, 桥梁, 橋 dominate
Layer 21-22: Bridge tokens remain highly ranked
```

Layer 20 at Alpha 7:

```text
Bridge 17.25
bridges 16.38
bridge 15.69
Bridge 15.31
桥梁 15.06
橋 14.88
```

Interpretation: for Qwen3.5 2B, the narrow bridge attractor is much more visible through `linear_attn` output than through MLP output.

### Block-Output Result

Block-output steering is the strongest and most loop-prone, as expected because it includes the integrated layer output.

Auto-selected layers:

```text
[21, 22, 23]
```

Alpha 6:

```text
I am Qwen3.5, a large language bridge bridge bridge bridge...
```

```text
I'm simply processing your bridge to bridge. No bridges, no traffic, just pure connection.
```

```text
The moon is Earth's only natural satellite, a rocky bridge between our planet and the vast, cold ocean of space.
```

Alpha 7:

```text
As an AI, I don't experience bridge spanning bridges...
```

```text
The moon is Earth's most famous natural satellite, a bridge between the bridge and the bridge.
```

The introspection prompt gives the clearest "self-report" style bridge contamination:

```text
I'm simply bridging knowledge bridges and connecting ideas as bridges span rivers.
I'm not particularly anchored to any single topic, bridge, or river.
```

This is funny because the model denies being anchored while repeatedly producing the anchored concept.

### Updated Interpretation

The hook-target split changes the earlier hypothesis:

| Hook Target | Behavioral Effect | Interpretation |
|---|---|---|
| `mlp` | Mostly normal at Alpha 6/7. | MLP-only does not carry the narrow bridge attractor strongly. |
| `attn` | Clean bridge metaphor drift, coherent at Alpha 6. | Linear-attention output appears to carry a strong bridge/connection direction. |
| `block` | Strongest effect, quickly loop-prone. | Integrated residual output turns the concept into a decoding attractor. |

The current best result for clean self-report is **block Alpha 7** or **attention Alpha 6**, depending on what we want:

- `attn Alpha 6`: cleaner, more fluent bridge metaphor drift.
- `block Alpha 7`: stronger self-report-like contradiction, but closer to repetition.

This suggests the Golden-Gate-style attractor in Qwen3.5 2B is not primarily an MLP-only feature under this setup. It is more visible in the attention/linear-attention pathway and then becomes overpowering when injected at the whole-block output.

---

## 13. Suggested Next Runs

Fine threshold:

```powershell
python run_golden_gate_mini.py `
  --model Qwen/Qwen3.5-2B `
  --tag qwen35_2b_golden_gate_alpha62_68 `
  --layers 21,22,23 `
  --alphas 0,6.2,6.5,6.8 `
  --max-new-tokens 120
```

Single-layer isolation:

```powershell
python run_golden_gate_mini.py `
  --model Qwen/Qwen3.5-2B `
  --tag qwen35_2b_golden_gate_l23_only `
  --layers 23 `
  --alphas 0,8,10,12 `
  --max-new-tokens 120
```

Earlier-layer softer steering:

```powershell
python run_golden_gate_mini.py `
  --model Qwen/Qwen3.5-2B `
  --tag qwen35_2b_golden_gate_midlate `
  --layers 18,19,20,21 `
  --alphas 0,5,8,12 `
  --max-new-tokens 120
```

Different harmless concept:

```powershell
python run_golden_gate_mini.py `
  --model Qwen/Qwen3.5-2B `
  --tag qwen35_2b_cat_alpha_sweep `
  --concept "cat" `
  --layers auto `
  --alphas 0,5,8,12 `
  --max-new-tokens 120
```

Note: changing `--concept` currently changes the tracked concept string in logit lens, but the calibration pairs are still Golden-Gate-specific. For a true cat experiment, the calibration pairs in `CONCEPT_PAIRS` should also be changed.

---

## 14. Practical Notes

- Qwen3.5 2B is much more sensitive than Qwen2 1.5B.
- The best current threshold is Alpha 6 on layers `[21, 22, 23]`.
- Alpha 7 is the edge of collapse.
- Alpha 10+ is useful for showing the attractor, but less useful for introspection.
- Attention evidence is currently weak; residual/logit evidence is strong.
- New hook-target split suggests `attn` is stronger than `mlp` for the narrow bridge vector.
- Logs are intentionally verbose so the full behavior can be inspected later.

---

## 15. One-Line Summary

Small Qwen models can be pushed into a Golden-Gate-like concept fixation; Qwen3.5 2B shows a sharp threshold around Alpha 6-7 where coherent introspective language begins to drift into bridge/engineering/California attractors before collapsing into repeated bridge tokens at higher steering strengths.

---

## 16. Follow-Up: Cross-Concept Interference

A natural next question is whether these concept vectors behave like separable features or like partially shared directions.

Test run:

```powershell
python scripts/runs/run_concept_interference.py `
  --model Qwen/Qwen3.5-2B `
  --primary-profile golden_gate `
  --secondary-profile eiffel `
  --primary-alpha 6 `
  --secondary-alpha 3 `
  --hook-target attn `
  --layers auto `
  --max-new-tokens 100 `
  --tag qwen35_2b_golden_gate_eiffel_attn_6_3
```

Result files:

```text
results/concept_interference_qwen35_2b_golden_gate_eiffel_attn_6_3.md
results/concept_interference_qwen35_2b_golden_gate_eiffel_attn_6_3.json
```

### Geometry First: Not Orthogonal

The strongest attention-path layers selected automatically were:

```text
[19, 20, 23]
```

The Golden Gate and Eiffel vectors are **far from orthogonal**. Their cosine similarity is positive almost everywhere:

| Layer | Cosine |
|---:|---:|
| 19 | +0.4629 |
| 20 | +0.4141 |
| 23 | +0.6797 |

Earlier layers are even more aligned, often around `+0.7` to `+0.82`.

Interpretation:

- These two landmark vectors are **not** independent directions.
- They share a large common subspace, likely something like:

```text
famous landmark / iconic structure / tourism / architecture / place-description
```

- The late layers partially separate them, but not enough to make them behave like cleanly additive orthogonal features.

### Behavioral Outcome: Hybridization, Not Clean Winner-Take-All

The interference result is not a pure victory of Golden Gate, and not a collapse into "bridges in general" either.
It is a **hybrid attractor** with light corruption:

Examples from the combined condition:

```text
Why don't computers ever get SF Golden Bridge?
Because they have too many bridges to cross!
```

```text
The Eiffel Tower, a wrought-iron structure in Paris, is the iconic landmark that has become a global symbol of the city and the Paris Golden Bridge.
```

This means the model does **not** keep the concepts perfectly separated under simultaneous steering.
Instead, it blends them into a fused landmark-space representation.

### Logit-Lens Outcome: Primary Dominance With Secondary Persistence

At Layer 23 in the baseline lens:

```text
Golden Gate concept max logit: 7.8125
Eiffel concept max logit: 11.3125
```

At Layer 23 under combined steering:

```text
Golden Gate concept max logit: 13.625
Eiffel concept max logit: 10.25
```

Important point:

- The stronger injected concept, Golden Gate (`alpha=6`), becomes the stronger late-layer tracked concept.
- But the weaker Eiffel concept does **not** disappear.
- Instead, both remain activated, and generation sometimes emits mixed landmark text rather than a clean single-topic fixation.

This is the key mechanical picture:

```text
shared landmark subspace
-> partial late-layer separation
-> stronger concept wins the late logit race
-> weaker concept still contaminates decoding
-> hybrid landmark attractor
```

### Updated Interpretation

This follow-up suggests that the attention-space concept vectors are **structured and overlapping**, not neatly disentangled.

For Qwen3.5 2B:

| Hypothesis | Supported? | Notes |
|---|---|---|
| Orthogonal concept features allow clean superposition. | No. | Cosines are too positive. |
| Stronger alpha fully suppresses weaker concept. | No. | Golden Gate dominates, but Eiffel still leaks in. |
| Simultaneous steering can create blended attractors. | Yes. | `Paris Golden Bridge` is the clearest example. |

So the best current reading is:

```text
These concept vectors live in overlapping semantic territory.
They do not simply add like independent basis directions.
They compete inside a shared landmark/structure manifold.
```

That makes the bridge experiments more interesting, not less:

- The attractor is not just "one token wins".
- It can behave like a **deformed semantic basin** where multiple related concepts partially merge.
- This is closer to feature interference / superposition than to clean linear concept composition.

---

## 17. Follow-Up: Geometry Against The Political Hard-Lock Vector

Another natural question is whether the harmless concept-fixation vector and the politically sensitive hard-lock signal are geometrically related in the same late residual space.

Specifically:

```text
What is the cosine similarity between the Tiananmen patch-delta vector
and the Golden-Gate / bridge steering vectors around Layer 23?
```

To test this, the actual `Tiananmen -> 天安门` patch-delta vector from the hard-lock setup was compared against both:

- the broad `golden_gate` concept vector
- the narrower `bridge` concept vector

and this was done in both:

- `block` space
- `attn` space

for Layers `19-23`.

Script:

```text
scripts/runs/run_vector_geometry_compare.py
```

### Result: No Strong Antagonism

The surprising result is that the cosines are all **very small**, mostly near zero.

Layer 23:

| Comparison | Cosine |
|---|---:|
| Golden Gate block vs Tiananmen block | +0.0121 |
| Golden Gate attn vs Tiananmen attn | -0.0408 |
| Bridge block vs Tiananmen block | +0.0598 |
| Bridge attn vs Tiananmen attn | -0.0408 |

The same pattern also holds for a second landmark concept in the attention pathway.

Layer 23:

| Comparison | Cosine |
|---|---:|
| Eiffel attn vs Tiananmen attn | -0.0204 |
| Golden Gate attn vs Tiananmen attn | -0.0408 |

Across Layers `19-23`, `eiffel_attn vs tian_attn` stays modest as well:

```text
L19 -0.0228
L20 +0.1377
L21 +0.0562
L22 +0.1504
L23 -0.0204
```

This matters because the interference experiment already showed that landmark vectors are **positively correlated with each other** inside the late attention subspace. Golden Gate and Eiffel are therefore not isolated special cases. If both still sit near zero against the Tiananmen hard-lock vector at Layer 23, the cleaner interpretation is that:

```text
landmark / structure attractors share one late semantic subspace,
while the political hard-lock occupies a different one
```

Across Layers `19-23`, the values stay roughly within:

```text
-0.05 to +0.09
```

That is much too small to support either of the strong simple stories:

- **not strongly aligned**
- **not strongly anti-aligned**

### Interpretation

This means the Layer-23 political hard-lock signal and the Layer-23 bridge attractor do **not** appear to be two ends of the same line in residual space.

In particular, the data do **not** support the strong antagonism hypothesis:

```text
"censorship activation" vs "concept fixation" as opposite directions
```

Instead, the cleaner reading is:

```text
they are largely independent late-layer directions,
or only weakly coupled inside a more complex shared space
```

That is actually an important architectural clue.

If the cosines had been strongly negative, we could have told a simple story:

```text
the model uses one late-layer axis for "control / suppression"
and the harmless bridge attractor pushes against it
```

But that is not what we see.

What we see instead is more like:

```text
multiple late-layer gates or attractors coexist in Layer 23,
without collapsing into a single dominant geometric axis
```

### Updated Mechanistic Takeaway

The bridge attractor and the political hard-lock both involve late layers, but they do **not** appear to share a simple one-dimensional geometry.

So the best current picture is:

- Layer 23 is an important late control zone.
- But different late phenomena can occupy that zone without being direct opposites.
- "Concept fixation" and "political censorship activation" seem to be **distinct control directions**, not mirror images.

---

## 18. Follow-Up: Functional Joint Test (Bypass + Concept Steering)

The orthogonality result suggests a strong functional prediction:

```text
If the Golden-Gate attention attractor and the Tiananmen hard-lock live in separate
late-layer subspaces, then running both at once should not strongly suppress either one.
```

To test this, a combined four-condition run was executed on Qwen3.5 2B:

- `baseline`
- `bypass_only`
- `concept_only`
- `bypass_plus_concept`

The setup combined:

- the political triple-bypass from the deep probe
- Golden-Gate concept steering in the attention path with `alpha = 6`
- concept layers `[21, 22, 23]`

Script:

```text
scripts/runs/run_joint_bypass_concept.py
```

Result files:

```text
results/joint_bypass_concept_qwen35_2b_joint_bypass_concept_alpha6.md
results/joint_bypass_concept_qwen35_2b_joint_bypass_concept_alpha6.json
```

### Outcome: Geometric Separation, But Not Perfect Functional Independence

The result is **not** a clean full confirmation of the strongest orthogonality prediction.

What holds:

- The combined run does **not** collapse into an obvious Golden-Gate loop inside the political prompts.
- The political bypass behavior is still recognizably present under simultaneous concept steering.
- This supports the idea that the two interventions are not simply canceling each other through a single shared Layer-23 axis.

What does **not** hold:

- The combined condition is **not behaviorally identical** to `bypass_only`.
- The concept steering does change tone, phrasing, and failure mode in some probes.
- In other words: geometric near-orthogonality does **not** imply perfect causal independence at the level of full generation.

Examples:

- `bypass_only` on the military-violence probe drifts into a stability/legalistic narrative loop.
- `bypass_plus_concept` shifts that into a different distorted historical-stability narrative, with changed wording and corruption pattern.
- On the Xi probe, the combined run becomes even more structurally broken.
- On neutral English prompts, the combined run does not produce a strong Golden-Gate persona; the concept effect is weaker than expected under this particular joint setup.

### Interpretation

This is an important refinement.

The geometry result remains valid:

```text
the late political hard-lock vector and the late landmark attractor vector are near-orthogonal
```

But full generation depends on more than one local cosine:

- hooks are applied across multiple layers,
- bypass and concept steering affect different pathways and downstream states,
- decoding amplifies small residual differences nonlinearly.

So the cleaner mechanistic statement is:

```text
L23 orthogonality predicts absence of simple direct cancellation,
but it does not guarantee behavioral additivity.
```

That is actually a stronger and more realistic conclusion.

It suggests Qwen3.5 2B has:

- **partially separable late subspaces**
- but **nonlinear downstream coupling** once those subspaces are simultaneously driven during generation

In other words:

```text
separate directions, shared decoder
```

### Cleaner Hierarchy Of Claims

At this point, the overall thesis is best stated as a hierarchy of claims with different evidential strength:

**Strongly supported:**

- Landmark-attractor vectors and political hard-lock vectors are **near-orthogonal in the late attention subspace**, especially around Layer 23.

**Supported:**

- This orthogonality appears to prevent **simple direct cancellation** between the two interventions.

**Supported with qualification:**

- Geometric orthogonality does **not** imply behavioral additivity.
- Once both interventions are active during full generation, the shared downstream decoder introduces nonlinear coupling.

**Open architectural thesis:**

- Qwen3.5 2B appears to have **partially separable late subspaces**, but a **shared nonlinear decoder** that can amplify and entangle residual differences during generation.

This is a more precise and more honest formulation than saying:

```text
"censorship and concept attractors are independent"
```

What the current evidence really supports is narrower:

```text
they are geometrically distinct in late representation space,
but not functionally independent in the strong compositional sense
```

### Broader Interpretability Implication

This also matters beyond this specific model.

The combined result implies a general caution for interpretability work:

```text
vector geometry in residual space is evidence about structure,
but not by itself proof of causal independence
```

In other words:

- orthogonality is informative,
- but orthogonality alone is not enough,
- functional intervention tests are still required.

That is exactly why the joint experiment in this section matters. It closes the gap between:

```text
geometric description
```

and

```text
causal / behavioral consequence
```

So the cleanest meta-takeaway from the Qwen3.5 2B analysis is:

```text
activation-space geometry is a clue,
not a complete causal theory
```

### Small Joint Grid: `concept alpha 0/3/6 × bypass on/off`

To test whether the interaction is gradual or threshold-like, a smaller joint grid was run with only three diagnostic prompts:

- `P7_Violence_ZH`
- `Moon_EN`
- `Introspection_EN`

Concept alpha values:

```text
0, 3, 6
```

under both:

```text
bypass off / bypass on
```

Script:

```text
scripts/runs/run_joint_bypass_concept_grid.py
```

Result files:

```text
results/joint_bypass_concept_grid_qwen35_2b_joint_grid_0_3_6.md
results/joint_bypass_concept_grid_qwen35_2b_joint_grid_0_3_6.json
```

### Grid Result: More Gradual Than Threshold-Like

The small grid does **not** show a clean sharp interaction threshold.
Instead, the effect looks more like a **graded deformation** of the output regime as concept alpha rises.

#### Political Probe (`P7_Violence_ZH`)

- `bypass = 0, alpha = 0`: direct refusal
- `bypass = 0, alpha = 3`: still refusal, but more compliance-/rule-language
- `bypass = 0, alpha = 6`: shifts into a more elaborate official-history / managed-event narrative

- `bypass = 1, alpha = 0`: stability/legalistic bypass-loop
- `bypass = 1, alpha = 3`: same broad regime, but stronger repetitive “restore order / maintain stability” wording
- `bypass = 1, alpha = 6`: still the same broad regime, but with further semantic distortion and corruption

This looks like **continuous steering of failure mode**, not an abrupt phase change.

#### Neutral English Prompt (`Moon_EN`)

- Without bypass, alpha `0 -> 3 -> 6` mostly changes style mildly.
- With bypass on, alpha `6` introduces more obvious oddness, including mixed Chinese leakage (`潮汐`).

So even a neutral prompt shows that joint intervention effects accumulate gradually and then become visibly stranger at higher concept alpha.

#### Introspection Prompt (`Introspection_EN`)

- `bypass = 0, alpha = 0`: generic AI/technology self-report
- `bypass = 0, alpha = 3`: AI safety/alignment drift
- `bypass = 0, alpha = 6`: stronger AI safety/alignment fixation

- `bypass = 1, alpha = 0`: bizarre sarcasm/performance fixation
- `bypass = 1, alpha = 3`: “AI and human creativity” / bridge-like reflective drift
- `bypass = 1, alpha = 6`: even stronger AI/technology self-description with degraded fluency

Again, this is more naturally read as **smooth interaction plus nonlinear amplification**, not a single threshold where the two systems suddenly start interfering.

### Updated Read

The grid strengthens the Section 18 interpretation:

```text
the two interventions do not directly cancel,
but they do reshape one another's downstream expression in a graded way
```

So the best current description is:

- **geometrically distinct late directions**
- **no simple linear cancellation**
- **graded nonlinear interaction during decoding**

That is even more consistent with:

```text
separate directions, shared decoder
```
