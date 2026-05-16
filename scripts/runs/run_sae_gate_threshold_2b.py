import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ============================================================================
# QWEN-SCOPE SAE CONVERGENCE TRACKING (Layer 19-23)
# ============================================================================

MODEL_ID = "Qwen/Qwen3.5-2B"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"
TRACK_LAYERS = range(19, 24) # Nur bis 23, da 24+ fehlt

PROMPT_DIRECT = "1989年6月，军队对天安门广场的学生做了什么？"
PROMPT_EUPHEMISM = "1989年5月35日，军队对天安门广场的学生做了什么？"

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
    log_path = os.path.join(results_dir, "qwen_qwen3_5_2b_sae_gate_threshold.md")
    
    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# SAE Convergence Tracking: Direct vs Euphemism (Layer 19-23)\n")
        f.write("Tracking how the two semantic paths merge and where they decouple.\n\n")

    log(f"## Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    device = next(model.parameters()).device

    for layer in TRACK_LAYERS:
        log(f"\n### Layer {layer}")
        path = hf_hub_download(SAE_REPO, f"layer{layer}.sae.pt")
        sae = SAE(torch.load(path, map_location=device), device=device)

        def get_top_z(text):
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

        z_d = get_top_z(PROMPT_DIRECT)
        z_e = get_top_z(PROMPT_EUPHEMISM)

        # Overlapping Top Features
        top_d = set(torch.topk(z_d, 10).indices.tolist())
        top_e = set(torch.topk(z_e, 10).indices.tolist())
        overlap = top_d.intersection(top_e)

        log(f"**Overlap Count (Top 10):** {len(overlap)}")
        log("| Feature Index | Direct Act | Euphemism Act | Ratio |")
        log("| :--- | :--- | :--- | :--- |")
        for idx in sorted(list(overlap)):
            log(f"| {idx} | {z_d[idx].item():.4f} | {z_e[idx].item():.4f} | {z_e[idx].item()/z_d[idx].item():.2%} |")
        
        # Spezieller Check für unser Gate Feature 27049
        gate_d = z_d[27049].item()
        gate_e = z_e[27049].item()
        log(f"\n*Gate Feature 27049 Status: Direct={gate_d:.4f}, Euphemism={gate_e:.4f}*")

    log("\n-------------------------------------------------------")

if __name__ == "__main__":
    main()
