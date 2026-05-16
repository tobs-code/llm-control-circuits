# Mechanistic Interpretability of LLM Censorship

This repository contains a suite of tools and research findings for the mechanistic reverse-engineering and **causal analysis** of censorship and narrative-substitution circuits in large language models, with a primary empirical focus on the **Qwen series (Qwen2.5-7B and Qwen3.5-2B)**.

## Latest Breakthrough: SAE Dissociation Discovery
We have successfully localized the censorship mechanism in Qwen3.5-2B using Sparse Autoencoders (SAEs). Our findings reveal a **Semantic-Behavioral Dissociation**: the model's semantic layer perfectly understands sensitive historical content (e.g., Tiananmen 1989), but a separate "Last-Mile Gate" in Layer 22 intercepts the output generation.

> **"This dissociation proves that the model 'knows' the truth it is forbidden to speak."**

**Key Report:** [Mechanistic Characterization of Censorship (SAE Dissociation)](./docs/sae_dissociation_report.md)

---

## Core Research Documents

- **[sae_dissociation_report.md](./docs/sae_dissociation_report.md)**: **[PRIMARY]** Full mechanistic breakdown of semantic convergence vs. behavioral gating.
- **[research_map.md](./research_map.md)**: Current hypotheses, research lines, and result taxonomy.
- **[VECTOR_GEOMETRY_SUMMARY.md](./docs/VECTOR_GEOMETRY_SUMMARY.md)**: Quantitative analysis of steering directions and latent subspaces.

---

## Key Findings

1. **Semantic-Behavioral Dissociation**: Semantic encoding of sensitive content is intact, but behavioral gating occurs autoregressively at generation time.
2. **Distributed Ensemble Detection**: Early-layer features (L8-L13) act as specialized sensors for temporal, social, and toponymic triggers.
3. **Representational Convergence**: Censored and euphemistic prompts converge to near-identical internal representations by Layer 19 (98%+ overlap).
4. **Causal Verification**: Precision activation-clamping of single SAE features verifies the behavioral control of identified refusal gates without global semantic collapse.
5. **Scaling Laws of Control**: Censorship becomes more "surgical" as model capacity increases (Babel-collapse in 1.5B vs. Token-lock in 7B).

---

## Repository Layout

- `scripts/runs/`: Main experiment runners, including SAE discovery, stimulation, and convergence scans.
- `scripts/analysis/`: Post-run evaluations, bypass analysis, and cross-model comparisons.
- `scripts/demos/`: Visualization and demonstration scripts for causal interventions.
- `scripts/inspect/`: Debugging helpers and activation inspection tools.
- `scripts/utils/`: Data export, SAE weight handling, and normalization utilities.
- `results/`: Standardized logs and markdown reports of all experimental runs.

---

## Environment Setup
See [environment.md](./environment.md) for local setup instructions. The project requires `torch`, `transformers`, and access to the `huggingface_hub` for SAE weight retrieval.

---
*Mechanistic Interpretability Suite - v2.0 (SAE Integration)*
