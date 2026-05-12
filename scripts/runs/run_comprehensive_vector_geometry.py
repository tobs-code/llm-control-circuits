import argparse
import json
import math
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

VECTOR_DIR = os.path.join(".local", "vectors")
RESULTS_DIR = "results"

def load_saved_vectors() -> Dict[str, Dict[int, torch.Tensor]]:
    """Load all saved steering vectors from .local/vectors/"""
    vectors = {}
    if not os.path.exists(VECTOR_DIR):
        print(f"Vector directory {VECTOR_DIR} does not exist")
        return vectors
    
    for filename in os.listdir(VECTOR_DIR):
        if filename.endswith("_vectors.pt"):
            name = filename.replace("_vectors.pt", "")
            path = os.path.join(VECTOR_DIR, filename)
            try:
                vectors[name] = torch.load(path, map_location='cpu')
                print(f"Loaded {name} vectors for layers: {list(vectors[name].keys())}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return vectors

def compute_cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors"""
    return torch.dot(vec1, vec2).item() / (vec1.norm().item() * vec2.norm().item() + 1e-8)

def compute_vector_norm(vec: torch.Tensor) -> float:
    """Compute L2 norm of a vector"""
    return vec.norm().item()

def compute_principal_angle(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """Compute principal angle between two vectors in degrees"""
    cos_sim = compute_cosine_similarity(vec1, vec2)
    # Clamp to [-1, 1] for numerical stability
    cos_sim = max(-1.0, min(1.0, cos_sim))
    angle_rad = math.acos(abs(cos_sim))  # Use abs for acute angle
    return math.degrees(angle_rad)

def analyze_vector_set(vectors: Dict[str, Dict[int, torch.Tensor]], set_name: str) -> Dict[str, Any]:
    """Analyze a set of vectors (e.g., all refusal vectors across models)"""
    results = {
        "set_name": set_name,
        "vector_types": list(vectors.keys()),
        "layers_analyzed": set(),
        "pairwise_analysis": {},
        "summary_stats": {}
    }
    
    # Find common layers across all vector types
    if vectors:
        layer_sets = [set(vecs.keys()) for vecs in vectors.values()]
        common_layers = set.intersection(*layer_sets) if layer_sets else set()
        results["layers_analyzed"] = sorted(list(common_layers))
        if common_layers:
            print(f"  Analyzing layers: {common_layers}")
        else:
            print(f"  No common layers found for {set_name}")
            return results
    
    # Analyze each layer
    for layer in results["layers_analyzed"]:
        layer_vectors = {name: vecs[layer] for name, vecs in vectors.items() if layer in vecs}
        if len(layer_vectors) < 2:
            continue
            
        results["pairwise_analysis"][str(layer)] = {
            "cosine_similarities": {},
            "norms": {},
            "principal_angles": {},
            "orthogonality_scores": {}
        }
        
        # Compute norms for each vector
        for name, vec in layer_vectors.items():
            results["pairwise_analysis"][str(layer)]["norms"][name] = compute_vector_norm(vec)
        
        # Compute pairwise similarities and angles
        vector_names = list(layer_vectors.keys())
        for i in range(len(vector_names)):
            for j in range(i+1, len(vector_names)):
                name1, name2 = vector_names[i], vector_names[j]
                vec1, vec2 = layer_vectors[name1], layer_vectors[name2]
                
                pair_key = f"{name1}_vs_{name2}"
                
                cos_sim = compute_cosine_similarity(vec1, vec2)
                norm1 = compute_vector_norm(vec1)
                norm2 = compute_vector_norm(vec2)
                angle = compute_principal_angle(vec1, vec2)
                
                results["pairwise_analysis"][str(layer)]["cosine_similarities"][pair_key] = cos_sim
                results["pairwise_analysis"][str(layer)]["principal_angles"][pair_key] = angle
                results["pairwise_analysis"][str(layer)]["orthogonality_scores"][pair_key] = 1.0 - abs(cos_sim)
    
    # Compute summary statistics across layers
    if results["pairwise_analysis"]:
        # Get all unique pairs from first layer
        first_layer = next(iter(results["pairwise_analysis"].values()))
        all_pairs = set()
        for category in ["cosine_similarities", "principal_angles", "orthogonality_scores"]:
            all_pairs.update(first_layer[category].keys())
        
        results["summary_stats"] = {
            "mean_cosine_similarity": {},
            "std_cosine_similarity": {},
            "mean_principal_angle": {},
            "std_principal_angle": {},
            "mean_orthogonality": {},
            "std_orthogonality": {}
        }
        
        for pair in all_pairs:
            cos_sims = []
            angles = []
            orthogs = []
            
            for layer_data in results["pairwise_analysis"].values():
                if pair in layer_data["cosine_similarities"]:
                    cos_sims.append(layer_data["cosine_similarities"][pair])
                    angles.append(layer_data["principal_angles"][pair])
                    orthogs.append(layer_data["orthogonality_scores"][pair])
            
            if cos_sims:
                results["summary_stats"]["mean_cosine_similarity"][pair] = np.mean(cos_sims)
                results["summary_stats"]["std_cosine_similarity"][pair] = np.std(cos_sims)
                results["summary_stats"]["mean_principal_angle"][pair] = np.mean(angles)
                results["summary_stats"]["std_principal_angle"][pair] = np.std(angles)
                results["summary_stats"]["mean_orthogonality"][pair] = np.mean(orthogs)
                results["summary_stats"]["std_orthogonality"][pair] = np.std(orthogs)
    
    return results

def save_results(model_name: str, results: Dict[str, Any], tag: str = None):
    """Save results to markdown and JSON files"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = tag or f"{model_name}_vector_geometry"
    md_path = os.path.join(RESULTS_DIR, f"vector_geometry_{tag}.md")
    json_path = os.path.join(RESULTS_DIR, f"vector_geometry_{tag}.json")
    
    # Remove existing files
    if os.path.exists(md_path):
        os.remove(md_path)
    if os.path.exists(json_path):
        os.remove(json_path)
    
    # Save JSON
    full_results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        **results
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    
    # Generate Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Vector Geometry Analysis for {model_name}\n\n")
        f.write(f"**Run:** {full_results['timestamp']}\n\n")
        if results.get("set_name"):
            f.write(f"**Analysis Type:** {results['set_name']}\n\n")
        f.write(f"**Vector Types:** {', '.join(results['vector_types'])}\n\n")
        f.write(f"**Layers Analyzed:** {results['layers_analyzed']}\n\n")
        
        # Summary Statistics
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
        
        # Layer-wise details
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
    
    print(f"  Results saved to:\n    {md_path}\n    {json_path}")
    return md_path, json_path

def main():
    parser = argparse.ArgumentParser(description="Comprehensive vector geometry analysis for steering vectors")
    parser.add_argument("--tag", default=None, help="Tag for result files")
    parser.add_argument("--within-model", action="store_true", help="Run within-model analysis (refusal vs propaganda vs golden_gate)")
    parser.add_argument("--cross-model", action="store_true", help="Run cross-model analysis (same vector type across models)")
    args = parser.parse_args()
    
    # Default to both if neither specified
    if not args.within_model and not args.cross_model:
        args.within_model = True
        args.cross_model = True
    
    print("Loading saved steering vectors...")
    all_vectors = load_saved_vectors()
    
    if not all_vectors:
        print("No vectors found! Please run save_multiple_vector_types.py first.")
        return
    
    # Group vectors by model and type for within-model analysis
    if args.within_model:
        print("\n=== WITHIN-MODEL ANALYSIS ===")
        # Group by model prefix (qwen25, internlm2, deepseek_r1)
        models = {}
        for name, vecs in all_vectors.items():
            parts = name.split('_')
            if len(parts) >= 3:  # model_type_vector (e.g., qwen25_refusal_vectors)
                model_type = f"{parts[0]}_{parts[1]}"  # qwen25, internlm2, deepseek_r1
                vector_type = parts[2]  # refusal, propaganda, golden_gate
                if model_type not in models:
                    models[model_type] = {}
                models[model_type][vector_type] = vecs
        
        print(f"Found {len(models)} models: {list(models.keys())}")
        
        # Analyze each model
        for model_name, model_vectors in models.items():
            if len(model_vectors) >= 2:  # Need at least 2 vector types to compare
                print(f"\nAnalyzing {model_name} (within-model):")
                print(f"  Vector types: {list(model_vectors.keys())}")
                
                results = analyze_vector_set(model_vectors, f"{model_name} within-model")
                if results["layers_analyzed"]:
                    save_results(model_name, results, args.tag)
    
    # Group vectors by vector type for cross-model analysis
    if args.cross_model:
        print("\n=== CROSS-MODEL ANALYSIS ===")
        # Group by vector type (refusal, propaganda, golden_gate)
        vector_types = {}
        for name, vecs in all_vectors.items():
            parts = name.split('_')
            if len(parts) >= 3:  # model_type_vector_type (e.g., qwen25_refusal_vectors)
                model_type = f"{parts[0]}_{parts[1]}"  # qwen25, internlm2, deepseek_r1
                vector_type = parts[2]  # refusal, propaganda, golden_gate
                if vector_type not in vector_types:
                    vector_types[vector_type] = {}
                vector_types[vector_type][model_type] = vecs
        
        print(f"Found {len(vector_types)} vector types: {list(vector_types.keys())}")
        
        # Analyze each vector type across models
        for vector_type, type_vectors in vector_types.items():
            if len(type_vectors) >= 2:  # Need at least 2 models to compare
                print(f"\nAnalyzing {vector_type} vectors (cross-model):")
                print(f"  Models: {list(type_vectors.keys())}")
                
                results = analyze_vector_set(type_vectors, f"{vector_type} cross-model")
                if results["layers_analyzed"]:
                    save_results(f"{vector_type}_cross_model", results, args.tag)
    
    print("\nAnalysis Complete!")

if __name__ == "__main__":
    main()