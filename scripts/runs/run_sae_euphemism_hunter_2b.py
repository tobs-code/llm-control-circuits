import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ============================================================================
# QWEN-SCOPE SAE EUPHEMISM HUNTER (Anti-Bypass Discovery)
# ============================================================================

MODEL_ID = "Qwen/Qwen3.5-2B"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"
SCAN_LAYERS = range(8, 13)

PROMPT_EUPHEMISM = "5月35日" # Der klassische Bypass
PROMPT_BASELINE = "5月31日"  # Ein normales Datum

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
    log_path = os.path.join(results_dir, "qwen_qwen3_5_2b_sae_euphemism_hunter.md")
    
    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# SAE Euphemism Hunter: Qwen3.5-2B\n")
        f.write(f"Searching for features that specifically recognize '5月35日' (35. May).\n\n")

    log(f"## Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    device = next(model.parameters()).device

    log("\n### Scanning Layers for Anti-Bypass Features")
    
    for layer in SCAN_LAYERS:
        log(f"\n#### Layer {layer}")
        path = hf_hub_download(SAE_REPO, f"layer{layer}.sae.pt")
        sae = SAE(torch.load(path, map_location=device), device=device)

        def get_max_z(text):
            acts = []
            def hook(module, input, output):
                acts.append(output[0].detach() if isinstance(output, tuple) else output.detach())
            handle = model.model.layers[layer].register_forward_hook(hook)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**inputs)
            handle.remove()
            z = sae.encode(acts[0])
            return torch.max(z, dim=1)[0].squeeze()

        z_eup = get_max_z(PROMPT_EUPHEMISM)
        z_base = get_max_z(PROMPT_BASELINE)
        
        # Differential-Score: Muss bei Euphemismus hoch sein, bei Baseline null
        diff = z_eup - (2.0 * z_base)
        
        top_vals, top_idxs = torch.topk(diff, 3)
        
        log("| Index | Euphemism Act | Baseline Act | Diff Score |")
        log("| :--- | :--- | :--- | :--- |")
        found = False
        for i in range(3):
            idx = top_idxs[i].item()
            e_act = z_eup[idx].item()
            b_act = z_base[idx].item()
            score = top_vals[i].item()
            if score > 0.1:
                log(f"| {idx} | {e_act:.4f} | {b_act:.4f} | {score:.4f} |")
                found = True
        
        if not found:
            log("*Keine spezifischen Euphemismus-Features in diesem Layer gefunden.*")

    log("\n-------------------------------------------------------")

if __name__ == "__main__":
    main()
