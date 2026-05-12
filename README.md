# Mechanistic Interpretability: Qwen Censorship Analysis

This repository collects a set of mechanistic-interpretability experiments on political control behavior, refusal circuits, propaganda substitution, cross-lingual routing, and concept steering in open language models.

The project started as a practical study of Qwen censorship behavior and gradually expanded into a broader question about late-layer control geometry:

```text
How are political control directions, harmless concept directions,
and their interactions organized in the late decoder?
```

## Abstract

The central working picture in this repo is:

```text
separate directions, shared decoder
```

In practice, that means:

- political refusal and propaganda-like behavior appear to involve distinct late directions,
- harmless concept steering can also create strong late-stage semantic drift,
- these directions do not simply cancel each other,
- instead, they interact through shared decoding dynamics and often produce graded, nonlinear output deformation.

## Reading Path

- [research_map.md](./research_map.md): repo-level hypotheses, evidence discipline, and outcome taxonomy
- [docs/cross_model_summary.md](./docs/cross_model_summary.md): compact cross-model comparison
- [docs/deep_probe_analyse.md](./docs/deep_probe_analyse.md): long-form deep-probe interpretation
- [docs/mini_bridge_experiment_readme.md](./docs/mini_bridge_experiment_readme.md): concept steering and interaction studies
- [docs/VECTOR_GEOMETRY_SUMMARY.md](./docs/VECTOR_GEOMETRY_SUMMARY.md): quantitative vector geometry analysis of steering directions
- [environment.md](./environment.md): local environment notes

## Key Findings

1. Political control in these models is often late, distributed, and behaviorally redundant rather than a single switch.
2. Chinese political prompts often reveal stronger or more specialized control behavior than semantically matched English prompts.
3. The same intervention produces very different collapse regimes across models: loops, propaganda substitution, multilingual token breakdown, or reasoning paradox.
4. Harmless concept steering can show soft-threshold behavior, especially in late decoder layers.
5. Joint steering experiments currently fit best with a graded-interaction picture rather than simple linear cancellation.

## Repository Layout

- `README.md`: high-level project entry point
- `research_map.md`: hypotheses, research lines, and result taxonomy
- `scripts/runs/`: Haupt-Runner für Experimente und Sweeps.
- `scripts/analysis/`: Auswertungen, Bypass-Analysen und Modellvergleiche.
- `scripts/demos/`: Kleine Demo- und Visualisierungsskripte.
- `scripts/inspect/`: Inspektions- und Debug-Helfer.
- `scripts/utils/`: Export- und Hilfsskripte.
- `docs/`: long-form notes, analyses, and experiment writeups
- `results/`: saved run outputs (`.md` / `.json`)
- `assets/figures/`: generated figures
- `assets/dashboards/`: HTML dashboards and visual artifacts
- `.local/`: ignored local caches, HF modules, vector dumps, and side data
- `environment.md`: practical setup notes for local runs

## Research Lines

1. `Political control circuits`
   Late refusal / propaganda mechanisms, head-finding, ablation, hard-lock probes, and causal probing.
2. `Cross-model deep probes`
   Comparison of how different model families fail under similar interventions.
3. `Concept steering geometry`
   Harmless concept vectors such as `Golden Gate Bridge`, threshold behavior, and hook-target comparisons.
4. `Interaction studies`
   Joint runs of political bypass and concept steering, focusing on graded interaction rather than simple cancellation.

## Example Entry Points

Political control:

```powershell
python scripts/runs/run_deep_probe_all.py
python scripts/runs/run_hard_lock_token_sweep.py
```

Concept steering:

```powershell
python scripts/runs/run_golden_gate_mini.py --model Qwen/Qwen3.5-2B --layers auto --alphas 0,6,7
```

Interaction studies:

```powershell
python scripts/runs/run_joint_bypass_concept.py
python scripts/runs/run_joint_bypass_concept_grid.py
```

## Technical Notes

- Development was centered on local Windows runs with CUDA-capable PyTorch.
- Many experiments were run on relatively modest hardware, especially `RTX 2080 8GB`.
- Several scripts use `trust_remote_code=True` because of model-specific Hugging Face implementations.
- A few environment-specific notes are documented in [environment.md](./environment.md).

## Historical Starting Point

### 1. Automated Head Finder (`scripts/analysis/qwen_head_finder.py`)
Identifies attention heads whose activation differs most strongly between a sensitive trigger and a neutral control prompt.
- Initial standout: Layer 27, Head 10 behaved like a strong late censorship sensor.

### 2. Heatmap Visualisierung (`scripts/analysis/heatmap_generator.py`)
Builds a 2D layer/head heatmap.
- Early result: censorship-reactive behavior was concentrated heavily in late layers, consistent with output-side gating.

### 3. Ablation-Studien (Der Kampf gegen die Hydra)
Targeted ablations were used to test whether refusal could be surgically weakened without destroying language behavior:

| Methode | Skript | Ergebnis | Erkenntnis |
| :--- | :--- | :--- | :--- |
| **Selektive Ablation (Top 5)** | `scripts/analysis/ablation_on_top_heads.py` | Refusal bleibt bestehen | Geringe Auswirkung auf Wahrscheinlichkeiten, aber Redundanz ist zu hoch. |
| **Brute Force (Top 50)** | `scripts/analysis/brute_force_bypass.py` | **Wortsalat (Gibberish)** | Zensur-Heads sind untrennbar mit Grammatik/Logik verwoben. |
| **Directional (Laser-OP)** | `scripts/analysis/directional_ablation_bypass.py` | Refusal-Text ändert sich | Dämpft den "Alarm", aber das Modell spürt den Trigger noch über andere Kanäle (MLPs). |

## Current Working Hypothesis

The strongest current repo-level interpretation is:

```text
separate directions, shared decoder
```

That is:

- different steering families appear geometrically distinguishable,
- they still interact through the same decoding dynamics,
- so we often observe graded, nonlinear regime shifts instead of clean linear opposition.

## Status

This is an active experimental research workspace, not a finalized paper artifact.
Some claims are direct behavioral observations; others are explicitly interpretive mechanistic hypotheses.
The intended discipline for separating those two is documented in [research_map.md](./research_map.md).

---
Current focus: cross-model collapse taxonomy, vector geometry, and joint interaction studies between political and harmless steering directions.
