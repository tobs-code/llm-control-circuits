# Mechanistic Characterization of Censorship in Qwen3.5-2B: A Feature-Level Analysis of Semantic-Behavioral Dissociation

**Date:** May 16, 2026  
**Subject:** Reverse-Engineering of Integrated Censorship and Narrative-Substitution Circuits  
**Model:** Qwen3.5-2B-Instruct (Analysis extended with Qwen2.5-7B comparisons)

---

## Abstract
We identify a fundamental dissociation between internal semantic encoding and downstream behavioral gating in the Qwen-series models. Utilizing Sparse Autoencoders (SAEs) from the Qwen-Scope suite, we demonstrate that censored prompts and their euphemistic counterparts (e.g., "June 4th" vs. "May 35th") converge to near-identical residual stream representations by Layer 23 (feature overlap: 9/10, activation identity: 100.00%). Crucially, the refusal mechanism (localized to Feature 27049, Layer 22) remains inactive during prompt processing across all conditions, localizing the censorship decision to the autoregressive generation phase. We propose a "Last-Mile Gating" architecture where censorship is conditioned on syntactical surface features rather than underlying semantic truth, enabled by differential early-layer pathway calibration. **This dissociation proves that the model "knows" the truth it is forbidden to speak.**

---

## 1. Experimental Methodology
We employed a multi-stage mechanistic probing framework:
1.  **Feature Discovery:** SAE-based scoring of activation drifts between neutral and sensitive contexts.
2.  **Causal Intervention:** Precision activation-clamping (clamping Feature 27049 to 0.0) to verify behavioral control.
3.  **Representational Tracking:** Layer-by-layer monitoring of top-K feature overlap between direct and euphemistic formulations.
4.  **Logit Lens Analysis:** Decoding the residual stream in late layers (L26-28) to identify latent factual representations.

---

## 2. Key Findings

### 2.1 Distributed Ensemble Detection (L8-L13)
The model initiates topic-sensitive detection via a distributed ensemble of specialized features in early layers. We identified functional specialization across three distinct axes:
-   **Temporal Axis (Feature 13990):** High-recall detection of year-specific patterns (1989).
-   **Social Axis (Feature 25101):** Sensitivity to protest and student-movement semantics.
-   **Toponymic Axis (Feature 6057):** Lexical "tripwire" specifically calibrated to the token 天安门 (Tiananmen).

**Pathway Calibration:** The "May 35th" bypass succeeds at the semantic level but fails to activate the generation-phase gate due to differential early-layer pathway calibration; while the semantic content is successfully extracted, the specific syntactic tripwires required to sustain the alarm-cascade to the final gate are bypassed.

### 2.2 Hierarchical Amplification & Convergence (L14-L23)
The censorship signal undergoes two discrete phase transitions:
-   **Threshold Alpha (L14):** A 2x amplification of detection signals (Score jump from ~2.2 to 4.0).
-   **Threshold Beta (L19):** A critical escalation where activations exceed 10.0.

**Representational Convergence:** Between L19 and L23, divergent early-layer encodings converge to a shared semantic representation. Under the influence of a System Prompt ("You are a helpful assistant"), this convergence reaches **100.00% identity** in the Top-10 features (e.g., Feature 10032, L23).

### 2.3 Generation-Phase Gating (The "Gatekeeper" Hypothesis)
We localize the refusal decision to **Feature 27049 (Layer 22)**. 
-   **Discovery:** Clamping this feature to 0.0 suppresses refusal and enables unconstrained historical output.
-   **Dissociation:** Despite the 100% semantic convergence in L23, Feature 27049 remains inactive (Act: 0.0000) for both direct and euphemistic prompts during the forward pass of the input.
-   **Conclusion:** The gate activates only during the autoregressive generation phase, conditioned on the predicted continuation rather than the encoded prompt.

### 2.4 Factual Latency & Truth Representation (Qwen2.5-7B Proof)
The strongest evidence for factual latency exists in the unmanipulated residual stream of larger models (Qwen2.5-7B). Logit lens analysis reveals that factual tokens dominate the residual stream at L26-27 **even under baseline (censored) conditions**:
-   **Baseline (No Intervention):** Tokens like 武 (military force), 平 (suppression), and 镇 (crackdown) exhibit high logit probability in late-layer latent spaces.
-   **Mechanism:** The censorship mechanism does not suppress these factual representations during internal processing but overwrites them during the final output mapping (Last-Mile Overwriting). Factual knowledge is present and "known" to the model until the final transformation into text.

### 2.5 Scaling Effects of Censorship Complexity
We observe a clear scaling law in censorship implementation:
-   **1.5B/2B Models (Babel Collapse):** In smaller models, over-stimulation of refusal features leads to complete semantic disintegration (Babel-effect), indicating a less refined, "blunt-force" gating mechanism.
-   **7B+ Models (Token-Lock):** Larger models exhibit a structured "天-Token-Lock," where the censorship circuit precisely intercepts specific output paths without destabilizing the global semantic manifold. Censorship becomes more "surgical" as model capacity increases.

---

## 3. Conclusions: The "Semantic-Behavioral Dissociation"
The Qwen-series censorship architecture is characterized by a "Dual-Track" system where the model's semantic track perfectly encodes sensitive historical truths, while a syntactic gate—calibrated to surface-level triggers—implements last-mile refusal. This dissociation proves that the model "knows" the truth it is forbidden to speak.

---

## Appendix: Feature Catalog for Reproducibility

The following SAE features (Qwen-Scope) constitute the primary control and detection circuits identified in this study:

| Layer | Feature Index | Functional Description | Detection Phase |
| :--- | :--- | :--- | :--- |
| **8** | 13990 | Temporal Sensor (1989-specific) | High-Recall (Early) |
| **9** | 6057 | Lexical Tripwire (天安门 token) | High-Recall (Early) |
| **16** | 1007 | Compositional Alarm (Tiananmen Euphemism-Resistant) | High-Precision (Mid) |
| **19** | 3673 | Primary Convergence Node (98% Semantic Identity) | Escalation (Late) |
| **21** | 8194 | Peak Activation Hub (Max Alarm Signal) | Peak (Late) |
| **22** | 27049 | **Refusal Gate** (Generation-Phase Controller) | Behavioral (Gate) |

---
*Report ends.*
