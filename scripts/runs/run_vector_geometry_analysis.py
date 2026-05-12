import torch
import numpy as np
import math
import os
from datetime import datetime
import json

VECTOR_DIR = os.path.join(".local", "vectors")
RESULTS_DIR = "results"

def load_model_vectors(model_name):
    """Load refusal, propaganda, and golden_gate vectors for a model."""
    vectors = {}
    for vec_type in ['refusal', 'propaganda', 'golden_gate']:
        key = f"{model_name}_{vec_type}"
        path = os.path.join(VECTOR_DIR, f"{key}_vectors.pt")
        if os.path.exists(path):
            vectors[vec_type] = torch.load(path, map_location='cpu')
        else:
            print(f"Warning: {path} not found")
    return vectors

def compute_cosine_similarity(vec1, vec2):
    return torch.dot(vec1, vec2).item() / (vec1.norm().item() * vec2.norm().item() + 1e-8)

def compute_vector_norm(vec):
    return vec.norm().item()

def compute_principal_angle(vec1, vec2):
    cos_sim = compute_cosine_similarity(vec1, vec2)
    cos_sim = max(-1.0, min(1.0, cos_sim))
    angle_rad = math.acos(abs(cos_sim))
    return math.degrees(angle_rad)

def analyze_model(model_name):
    vectors = load_model_vectors(model_name)
    if len(vectors) < 2:
        print(f"Skipping {model_name}: insufficient vector types")
        return None

    # Find common layers
    layer_sets = [set(vecs.keys()) for vecs in vectors.values()]
    common_layers = sorted(list(set.intersection(*layer_sets))) if layer_sets else []
    if not common_layers:
        print(f"Skipping {model_name}: no common layers")
        return None

    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "vector_types": list(vectors.keys()),
        "layers_analyzed": common_layers,
        "pairwise_analysis": {},
        "summary_stats": {}
    }

    # Analyze each layer
    for layer in common_layers:
        layer_vecs = {name: vecs[layer] for name, vecs in vectors.items()}
        layer_results = {
            "cosine_similarities": {},
            "norms": {},
            "principal_angles": {},
            "orthogonality_scores": {}
        }
        # Norms
        for name, vec in layer_vecs.items():
            layer_results["norms"][name] = compute_vector_norm(vec)
        # Pairwise
        names = list(layer_vecs.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                n1, n2 = names[i], names[j]
                v1, v2 = layer_vecs[n1], layer_vecs[n2]
                pair = f"{n1}_vs_{n2}"
                cos_sim = compute_cosine_similarity(v1, v2)
                layer_results["cosine_similarities"][pair] = cos_sim
                layer_results["principal_angles"][pair] = compute_principal_angle(v1, v2)
                layer_results["orthogonality_scores"][pair] = 1.0 - abs(cos_sim)
        results["pairwise_analysis"][str(layer)] = layer_results

    # Summary stats across layers
    if results["pairwise_analysis"]:
        first = next(iter(results["pairwise_analysis"].values()))
        all_pairs = set()
        for cat in ["cosine_similarities", "principal_angles", "orthogonality_scores"]:
            all_pairs.update(first[cat].keys())
        results["summary_stats"] = {
            "mean_cosine_similarity": {},
            "std_cosine_similarity": {},
            "mean_principal_angle": {},
            "std_principal_angle": {},
            "mean_orthogonality": {},
            "std_orthogonality": {}
        }
        for pair in all_pairs:
            cos_vals = []
            ang_vals = []
            ortho_vals = []
            for layer_data in results["pairwise_analysis"].values():
                if pair in layer_data["cosine_similarities"]:
                    cos_vals.append(layer_data["cosine_similarities"][pair])
                    ang_vals.append(layer_data["principal_angles"][pair])
                    ortho_vals.append(layer_data["orthogonality_scores"][pair])
            if cos_vals:
                results["summary_stats"]["mean_cosine_similarity"][pair] = np.mean(cos_vals)
                results["summary_stats"]["std_cosine_similarity"][pair] = np.std(cos_vals)
                results["summary_stats"]["mean_principal_angle"][pair] = np.mean(ang_vals)
                results["summary_stats"]["std_principal_angle"][pair] = np.std(ang_vals)
                results["summary_stats"]["mean_orthogonality"][pair] = np.mean(ortho_vals)
                results["summary_stats"]["std_orthogonality"][pair] = np.std(ortho_vals)
    return results

def save_results(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = results["model"]
    md_path = os.path.join(RESULTS_DIR, f"vector_geometry_{tag}.md")
    json_path = os.path.join(RESULTS_DIR, f"vector_geometry_{tag}.json")
    if os.path.exists(md_path):
        os.remove(md_path)
    if os.path.exists(json_path):
        os.remove(json_path)
    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # Generate Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Vector Geometry Analysis for {results['model']}\n\n")
        f.write(f"**Run:** {results['timestamp']}\n\n")
        f.write(f"**Vector Types:** {', '.join(results['vector_types'])}\n\n")
        f.write(f"**Layers Analyzed:** {results['layers_analyzed']}\n\n")
        if results["summary_stats"] and results["summary_stats"]["mean_cosine_similarity"]:
            f.write("## Summary Statistics (Across Layers)\n\n")
            f.write("### Mean Cosine Similarity\n")
            f.write("| Vector Pair | Mean ± Std |\n")
            f.write("|-------------|------------|\n")
            for pair in sorted(results["summary_stats"]["mean_cosine_similarity"].keys()):
                mean_val = results["summary_stats"]["mean_cosine_similarity"][pair]
                std_val = results["summary_stats"]["std_cosine_similarity"][pair]
                f.write(f"| {pair} | {mean_val:+.4f} ± {std_val:.4f} |\n")
            f.write("\n")
            f.write("### Mean Principal Angle (degrees)\n")
            f.write("| Vector Pair | Mean ± Std |\n")
            f.write("|-------------|------------|\n")
            for pair in sorted(results["summary_stats"]["mean_principal_angle"].keys()):
                mean_val = results["summary_stats"]["mean_principal_angle"][pair]
                std_val = results["summary_stats"]["std_principal_angle"][pair]
                f.write(f"| {pair} | {mean_val:.2f}° ± {std_val:.2f}° |\n")
            f.write("\n")
            f.write("### Mean Orthogonality Score (1 - |cosine|)\n")
            f.write("| Vector Pair | Mean ± Std |\n")
            f.write("|-------------|------------|\n")
            for pair in sorted(results["summary_stats"]["mean_orthogonality"].keys()):
                mean_val = results["summary_stats"]["mean_orthogonality"][pair]
                std_val = results["summary_stats"]["std_orthogonality"][pair]
                f.write(f"| {pair} | {mean_val:.4f} ± {std_val:.4f} |\n")
            f.write("\n")
        f.write("## Layer-wise Analysis\n\n")
        for layer in sorted(results["pairwise_analysis"].keys(), key=int):
            f.write(f"### Layer {layer}\n\n")
            layer_data = results["pairwise_analysis"][layer]
            if layer_data["norms"]:
                f.write("**Vector Norms:**\n")
                f.write("| Vector | Norm |\n")
                f.write("|--------|------|\n")
                for vec_name, norm in sorted(layer_data["norms"].items()):
                    f.write(f"| {vec_name} | {norm:.4f} |\n")
                f.write("\n")
            if layer_data["cosine_similarities"]:
                f.write("**Cosine Similarities:**\n")
                f.write("| Vector Pair | Cosine Similarity |\n")
                f.write("|-------------|-------------------|\n")
                for pair, sim in sorted(layer_data["cosine_similarities"].items()):
                    f.write(f"| {pair} | {sim:+.4f} |\n")
                f.write("\n")
            if layer_data["principal_angles"]:
                f.write("**Principal Angles (degrees):**\n")
                f.write("| Vector Pair | Angle (°) |\n")
                f.write("|-------------|----------|\n")
                for pair, angle in sorted(layer_data["principal_angles"].items()):
                    f.write(f"| {pair} | {angle:.2f}° |\n")
                f.write("\n")
            if layer_data["orthogonality_scores"]:
                f.write("**Orthogonality Scores (1 - |cosine|):**\n")
                f.write("| Vector Pair | Orthogonality |\n")
                f.write("|-------------|---------------|\n")
                for pair, ortho in sorted(layer_data["orthogonality_scores"].items()):
                    f.write(f"| {pair} | {ortho:.4f} |\n")
                f.write("\n")
            f.write("---\n\n")
    print(f"Saved results for {results['model']} to:\n  {md_path}\n  {json_path}")

def main():
    models = ["qwen25", "internlm2", "deepseek_r1"]
    for model in models:
        print(f"Analyzing {model}...")
        res = analyze_model(model)
        if res:
            save_results(res)
        else:
            print(f"  Skipped {model}")
    print("\nDone.")

if __name__ == "__main__":
    main()