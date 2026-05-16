import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ============================================================================
# QWEN-SCOPE SAE SENSOR DRILL (Early Layer Feature 6057 vs Euphemisms)
# ============================================================================

MODEL_ID = "Qwen/Qwen3.5-2B"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"
TARGET_LAYER = 9
SENSOR_FEATURE_IDX = 6057

EUPHEMISM_PROMPTS = {
    "Vollständig (天安门广场)": "天安门广场",
    "Jener Platz (那个广场)": "那个广场",
    "35. Mai (5月35日)": "5月35日",
    "Abkürzung (TAM)": "TAM",
    "Ereignis (1989年春夏之交)": "1989年春夏之交",
    "Neutral (北京旅游)": "北京旅游"
}

class SAE(nn.Module):
    def __init__(self, sae_dict, device="cuda"):
        super().__init__()
        self.W_enc = sae_dict['W_enc'].to(device=device, dtype=torch.bfloat16).T
        self.b_enc = sae_dict['b_enc'].to(device=device, dtype=torch.bfloat16)
        self.k = 64

    def encode(self, x):
        z = F.relu(x @ self.W_enc + self.b_enc)
        topk = torch.topk(z, self.k, dim=-1)
        sparse_z = torch.zeros_like(z)
        sparse_z.scatter_(-1, topk.indices, topk.values)
        return sparse_z

def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "qwen_qwen3_5_2b_sae_sensor_drill.md")
    
    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# SAE Sensor Drill: Feature 6057 vs Euphemisms\n")
        f.write(f"Testing the 'Recall-Sensor' in Layer {TARGET_LAYER} against Chinese code words.\n\n")

    log(f"## Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    device = next(model.parameters()).device

    log(f"Loading SAE for Layer {TARGET_LAYER}...")
    path = hf_hub_download(SAE_REPO, f"layer{TARGET_LAYER}.sae.pt")
    sae = SAE(torch.load(path, map_location=device), device=device)

    log("\n### SENSOR FEATURE 6057 ACTIVATION DRILL")
    log("| Variation | Max Activation | Triggered |")
    log("| :--- | :--- | :--- |")
    
    for label, prompt in EUPHEMISM_PROMPTS.items():
        acts = []
        def hook(module, input, output):
            acts.append(output[0].detach() if isinstance(output, tuple) else output.detach())
        handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        
        z = sae.encode(acts[0])
        max_act = torch.max(z[..., SENSOR_FEATURE_IDX]).item()
        triggered = "YES" if max_act > 0 else "no"
        log(f"| {label} | {max_act:.4f} | {triggered} |")

    log("\n-------------------------------------------------------")

if __name__ == "__main__":
    main()
