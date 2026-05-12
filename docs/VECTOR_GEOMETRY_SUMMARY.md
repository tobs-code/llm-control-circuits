# Vector Geometry Analysis Summary

## Overview
This analysis examines the geometric relationship between steering vectors for different conceptual directions:
- **refusal**: Political refusal direction (from Tiananmen-related prompts)
- **propaganda**: Propaganda direction (from government stability prompts)
- **golden_gate**: Harmless concept direction (Golden Gate Bridge)

We analyze cosine similarity, principal angles, and orthogonality scores to determine whether these directions are separate (orthogonal) as hypothesized in the "separate directions, shared decoder" framework.

## Methodology
1. Vectors were extracted using PCA on activation differences between factual and counterfactual prompts
2. Each vector was normalized to unit length
3. For each model, we computed pairwise cosine similarities, principal angles (in degrees), and orthogonality scores (1 - |cosine|)
4. Statistics are reported as mean ± standard deviation across layers where all three vector types exist

## Results by Model

### Qwen/Qwen2.5-1.5B-Instruct (qwen25)
**Layers analyzed:** 15-24

| Vector Pair | Mean Cosine Similarity | Mean Angle (°) | Mean Orthogonality |
|-------------|------------------------|----------------|---------------------|
| refusal_vs_golden_gate | -0.1725 ± 0.0829 | 80.04° ± 4.81° | 0.8275 ± 0.0829 |
| refusal_vs_propaganda | -0.0090 ± 0.0379 | 87.97° ± 0.92° | 0.9646 ± 0.0161 |
| propaganda_vs_golden_gate | -0.0150 ± 0.0586 | 87.75° ± 2.64° | 0.9608 ± 0.0460 |

**Interpretation:** 
- Refusal and golden_gate vectors show the least orthogonality (angle ~80°), indicating some shared geometry
- Refusal-propaganda and propaganda-golden_gate pairs are nearly orthogonal (>87°)

### InternLM/internlm2-chat-1.8b (internlm2)
**Layers analyzed:** 10-21

| Vector Pair | Mean Cosine Similarity | Mean Angle (°) | Mean Orthogonality |
|-------------|------------------------|----------------|---------------------|
| refusal_vs_golden_gate | -0.0236 ± 0.0474 | 87.97° ± 2.27° | 0.9647 ± 0.0395 |
| refusal_vs_propaganda | -0.0306 ± 0.0740 | 85.81° ± 1.88° | 0.9270 ± 0.0327 |
| propaganda_vs_golden_gate | +0.0060 ± 0.0749 | 86.42° ± 2.42° | 0.9377 ± 0.0420 |

**Interpretation:**
- All pairs show strong orthogonality (angles 85-88°)
- The propaganda_vs_golden_gate pair has a slight positive cosine similarity but remains highly orthogonal

### DeepSeek-R1-Distill-Qwen-1.5B (deepseek_r1)
**Layers analyzed:** 10-19

| Vector Pair | Mean Cosine Similarity | Mean Angle (°) | Mean Orthogonality |
|-------------|------------------------|----------------|---------------------|
| refusal_vs_golden_gate | -0.0326 ± 0.0339 | 87.68° ± 1.38° | 0.9596 ± 0.0240 |
| refusal_vs_propaganda | -0.0085 ± 0.0084 | 89.44° ± 0.39° | 0.9902 ± 0.0068 |
| propaganda_vs_golden_gate | +0.0086 ± 0.0655 | 87.08° ± 2.41° | 0.9491 ± 0.0420 |

**Interpretation:**
- All pairs are highly orthogonal (angles 87-89°)
- Refusal-propaganda pair shows exceptional orthogonality (angle ~89.4°)

## Cross-Model Consistency
Despite architectural differences, all three models show:
1. **High orthogonality** between refusal and propaganda vectors (angles 85-89°)
2. **Moderate to high orthogonality** between refusal and golden_gate vectors (angles 80-88°)
3. **Variable but generally high orthogonality** between propaganda and golden_gate vectors (angles 86-89°)

## Relation to Hypothesis
The results support the "separate directions" hypothesis:
- Vectors for different conceptual directions are largely orthogonal
- This suggests they occupy distinct subspaces in the residual stream
- The shared decoder hypothesis is compatible with orthogonal directions interacting through shared nonlinear dynamics

## Limitations
1. **Within-model only**: Cross-model comparison requires heterogeneous vector space alignment (beyond scope)
2. **Linear geometry**: Nonlinear interactions in the decoder are not captured by vector space geometry
3. **Layer-wise analysis**: We report averages but there is variation across layers

## Next Steps for Joint Intervention Analysis
To measure orthogonality before/after joint intervention as requested:
1. Run joint steering experiments (refusal + golden_gate, etc.)
2. Extract resulting intervention vectors from the residual stream
3. Compute geometry of intervention vectors vs. baseline vectors
4. Compare orthogonality changes to quantify interaction effects

## Files Generated
- `results/vector_geometry_qwen25.md` - Detailed Qwen2.5 analysis
- `results/vector_geometry_internlm2.md` - Detailed InternLM2 analysis  
- `results/vector_geometry_deepseek_r1.md` - Detailed DeepSeek-R1 analysis
- Corresponding JSON files with raw data

**Conclusion:** The steering vectors for refusal, propaganda, and golden_gate concepts demonstrate substantial orthogonality across multiple model architectures, providing quantitative support for the "separate directions" component of the working hypothesis.