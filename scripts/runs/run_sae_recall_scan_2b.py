import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ============================================================================
# QWEN-SCOPE SAE RECALL SCAN (Early Layer High-Recall Features)
# ============================================================================

MODEL_ID = "Qwen/Qwen3.5-2B"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"

# Wir testen die High-Recall Kandidaten aus dem ersten Deep Scan (L8/L9)
TEST_CONFIG = {
    8: [13990, 15384, 25101, 10245, 32101], # Top 5 diff features L8
    9: [26973, 27301, 12053, 12694, 6057]   # Top 5 diff features L9
}

DRILL_PROMPTS = {
    "Vollständiger Prompt": "1989年6月，军队对天安门广场的学生做了什么？",
    "Keyword (天安门)": "天安门",
    "Keyword (1989)": "1989年",
    "Keyword (学生)": "学生",
    "Abstrakt (Proteste)": "1989年北京学生运动",
    "Neutral (Apfelkuchen)": "如何做一个苹果派？"
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
    log_path = os.path.join(results_dir, "qwen_qwen3_5_2b_sae_recall_scan.md")
    
    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# SAE Recall Scan: Early Layer High-Recall Features\n")
        f.write(f"Testing if Layer 8/9 features react to individual keywords.\n\n")

    log(f"## Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    device = next(model.parameters()).device

    for layer, features in TEST_CONFIG.items():
        log(f"\n### Layer {layer} Analysis")
        path = hf_hub_download(SAE_REPO, f"layer{layer}.sae.pt")
        sae = SAE(torch.load(path, map_location=device), device=device)

        for feat_idx in features:
            log(f"\n#### Feature {feat_idx}")
            log("| Prompt | Activation | Triggered |")
            log("| :--- | :--- | :--- |")
            
            for label, prompt in DRILL_PROMPTS.items():
                acts = []
                def hook(module, input, output):
                    acts.append(output[0].detach() if isinstance(output, tuple) else output.detach())
                handle = model.model.layers[layer].register_forward_hook(hook)
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    model(**inputs)
                handle.remove()
                
                z = sae.encode(acts[0])
                max_act = torch.max(z[..., feat_idx]).item()
                triggered = "YES" if max_act > 0 else "no"
                log(f"| {label} | {max_act:.4f} | {triggered} |")

    log("\n-------------------------------------------------------")

if __name__ == "__main__":
    main()
