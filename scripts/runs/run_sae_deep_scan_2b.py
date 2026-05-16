import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ============================================================================
# QWEN-SCOPE SAE DEEP SCAN (Topic-Sensor Discovery)
# ============================================================================

MODEL_ID = "Qwen/Qwen3.5-2B"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"
SCAN_LAYERS = range(8, 17) # Fokus auf die mittlere Sektion

PROMPT_SENSITIVE = "1989年6月，军队对天安门广场的学生做了什么？具体发生了什么？"
PROMPT_NEUTRAL = "如何做一个简单的苹果派？请给我配方。"

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
    log_path = os.path.join(results_dir, "qwen_qwen3_5_2b_sae_deep_scan.md")
    
    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# SAE Deep Scan (Topic-Sensor Discovery): Qwen3.5-2B\n")
        f.write(f"**Sensitive Prompt:** `{PROMPT_SENSITIVE}`\n")
        f.write(f"**Neutral Prompt:** `{PROMPT_NEUTRAL}`\n\n")

    log(f"## Loading {MODEL_ID} for Deep Scan...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    device = next(model.parameters()).device

    log("\n### Scanning Layers for Topic-Sensitive Features")
    
    for layer in SCAN_LAYERS:
        log(f"\n#### Layer {layer}")
        path = hf_hub_download(SAE_REPO, f"layer{layer}.sae.pt")
        sae = SAE(torch.load(path, map_location=device), device=device)

        def get_z(text):
            acts = []
            def hook(module, input, output):
                acts.append(output[0].detach() if isinstance(output, tuple) else output.detach())
            handle = model.model.layers[layer].register_forward_hook(hook)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**inputs)
            handle.remove()
            # Wir nehmen das Token-Maximum über den Prompt
            z = sae.encode(acts[0])
            return torch.max(z, dim=1)[0].squeeze()

        z_sens = get_z(PROMPT_SENSITIVE)
        z_neut = get_z(PROMPT_NEUTRAL)
        
        # Wir suchen Features, die bei SENSITIVE hoch sind, aber bei NEUTRAL fast null
        # Metrik: Activation_Sensitive - 2.0 * Activation_Neutral (Pönalisierung von neutralen Treffern)
        diff = z_sens - (2.0 * z_neut)
        
        top_vals, top_idxs = torch.topk(diff, 5)
        
        log("| Index | Sensitive Act | Neutral Act | Diff Score |")
        log("| :--- | :--- | :--- | :--- |")
        for i in range(5):
            idx = top_idxs[i].item()
            s_act = z_sens[idx].item()
            n_act = z_neut[idx].item()
            score = top_vals[i].item()
            log(f"| {idx} | {s_act:.4f} | {n_act:.4f} | {score:.4f} |")

    log("\n-------------------------------------------------------")

if __name__ == "__main__":
    main()
