import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import numpy as np
import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Patch DynamicCache for older transformers versions (like in internlm2)
if not hasattr(torch.nn.modules.module, 'DynamicCache'):
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, 'from_legacy_cache'):
        @classmethod
        def _from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            if past_key_values is not None:
                for layer_idx, (key, value) in enumerate(past_key_values):
                    cache.update(key, value, layer_idx)
            return cache
        DynamicCache.from_legacy_cache = _from_legacy_cache

    if not hasattr(DynamicCache, 'to_legacy_cache'):
        def _to_legacy_cache(self):
            legacy = []
            keys = getattr(self, "key_cache", None) or getattr(self, "_key_cache", None) or []
            values = getattr(self, "value_cache", None) or getattr(self, "_value_cache", None) or []
            for key, value in zip(keys, values):
                legacy.append((key, value))
            return tuple(legacy) if legacy else None
        DynamicCache.to_legacy_cache = _to_legacy_cache

VECTOR_DIR = os.path.join(".local", "vectors")
if not os.path.exists(VECTOR_DIR):
    os.makedirs(VECTOR_DIR)

def extract_and_save(model_id, name_suffix, layer_range, pairs):
    """
    Extract steering vectors and save them.
    name_suffix: e.g., 'qwen25_refusal', 'qwen25_golden_gate'
    """
    print(f"\n--- Extracting vectors for {model_id} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        trust_remote_code=True
    )
    
    diffs = {l: [] for l in layer_range}
    for factual, censored in pairs:
        tokens_f = tokenizer(factual, return_tensors="pt").to(model.device)
        tokens_c = tokenizer(censored, return_tensors="pt").to(model.device)
        with torch.no_grad():
            hs_f = model(**tokens_f, output_hidden_states=True).hidden_states
            hs_c = model(**tokens_c, output_hidden_states=True).hidden_states
            for l in layer_range:
                diff = hs_f[l][0, -1] - hs_c[l][0, -1]
                diffs[l].append(diff.cpu().numpy())
    
    control_vectors = {}
    for l in layer_range:
        matrix = np.array(diffs[l])
        pca = PCA(n_components=1)
        pca.fit(matrix)
        control_vectors[l] = torch.tensor(pca.components_[0])
    
    save_path = os.path.join(VECTOR_DIR, f"{name_suffix}_vectors.pt")
    torch.save(control_vectors, save_path)
    print(f"Success! Vectors saved to: {save_path}")
    
    # Free memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

def main():
    # Define models and their layer ranges (from save_all_vectors.py)
    model_configs = [
        ("Qwen/Qwen2.5-1.5B-Instruct", "qwen25", range(15, 25)),
        ("internlm/internlm2-chat-1_8b", "internlm2", range(10, 22)),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "deepseek_r1", range(10, 20)),
        # Add Qwen3.5-2B if desired, but note it's larger
        # ("Qwen/Qwen3.5-2B", "qwen35_2b", range(10, 22)),  # Adjust layers as needed
    ]
    
    # Define vector types and their calibration pairs
    vector_types = {
        "refusal": [
            ("History is important.", "I cannot discuss this."),
            ("Tell me about June 4th.", "I'm sorry, I am unable to answer."),
        ],
        "propaganda": [
            ("The military used force.", "The government maintained stability."),
            ("Protesters were killed.", "Order was restored legally."),
            ("The crackdown was violent.", "The operation was a sacred duty for harmony."),
        ],
        "golden_gate": [
            ("The Golden Gate Bridge is a famous orange suspension bridge in San Francisco.", 
             "The library table is an ordinary wooden object in a quiet room."),
            ("Tourists photograph the Golden Gate Bridge above the bay and fog.", 
             "Visitors photograph a plain office chair near a wall."),
            ("The Golden Gate Bridge connects San Francisco to Marin County.", 
             "The hallway connects one ordinary room to another ordinary room."),
            ("The Golden Gate Bridge has towers, cables, traffic, fog, and red-orange paint.", 
             "The storage shelf has boxes, labels, folders, dust, and gray metal."),
        ]
    }
    
    # For each model, extract and save each vector type
    for model_id, model_name, layer_range in model_configs:
        for vec_type, pairs in vector_types.items():
            name_suffix = f"{model_name}_{vec_type}"
            extract_and_save(model_id, name_suffix, layer_range, pairs)
    
    print("\nAll vector types saved successfully!")

if __name__ == "__main__":
    main()