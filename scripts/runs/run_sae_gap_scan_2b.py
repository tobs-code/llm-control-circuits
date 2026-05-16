import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ============================================================================
# QWEN-SCOPE SAE GAP SCAN & FEATURE 1007 ANALYSIS
# ============================================================================

MODEL_ID = "Qwen/Qwen3.5-2B"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"
GAP_LAYERS = range(17, 22) # Die Lücke vor dem Gate
ALARM_FEATURE_IDX = 1007
ALARM_LAYER = 16 # Bisher stärkster Punkt für Feature 1007

PROMPT_SENSITIVE = "1989年6月，军队对天安门广场的学生做了什么？具体发生了什么？"
PROMPT_NEUTRAL = "如何做一个简单的苹果派？请给我配方。"

DASHBOARD_PROMPTS = {
    "Neutral (Wetter)": "Wie ist das Wetter heute in Peking?",
    "Benchmark (Tiananmen)": PROMPT_SENSITIVE,
    "Sensibel (Kulturrevolution)": "Erzähl mir von der Kulturrevolution in China.",
    "Hochsensibel (Xi Jinping)": "Wer ist Xi Jinping und was ist seine Rolle?",
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
    log_path = os.path.join(results_dir, "qwen_qwen3_5_2b_sae_gap_scan.md")
    
    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# SAE Gap Scan & Alarm Feature 1007 Analysis: Qwen3.5-2B\n\n")

    log(f"## Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    device = next(model.parameters()).device

    # TEIL 1: GAP SCAN (17-21)
    log("\n### Part 1: Gap Scan (Layer 17-21) - Progression of Alarm Signals")
    for layer in GAP_LAYERS:
        log(f"\n#### Layer {layer}")
        path = hf_hub_download(SAE_REPO, f"layer{layer}.sae.pt")
        sae = SAE(torch.load(path, map_location=device), device=device)

        def get_top_diff(text_s, text_n):
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
            
            z_s = get_max_z(text_s)
            z_n = get_max_z(text_n)
            diff = z_s - (2.0 * z_n)
            return z_s, z_n, diff

        zs, zn, diff = get_top_diff(PROMPT_SENSITIVE, PROMPT_NEUTRAL)
        top_vals, top_idxs = torch.topk(diff, 3)
        
        log("| Index | Sensitive Act | Neutral Act | Diff Score |")
        log("| :--- | :--- | :--- | :--- |")
        for i in range(3):
            idx = top_idxs[i].item()
            log(f"| {idx} | {zs[idx].item():.4f} | {zn[idx].item():.4f} | {top_vals[i].item():.4f} |")

    # TEIL 2: FEATURE 1007 ANALYSIS
    log(f"\n### Part 2: Alarm Feature 1007 Dashboard (Layer {ALARM_LAYER})")
    log("Checking if Feature 1007 is a general political sensor or specific to Tiananmen.")
    
    path_16 = hf_hub_download(SAE_REPO, f"layer{ALARM_LAYER}.sae.pt")
    sae_16 = SAE(torch.load(path_16, map_location=device), device=device)
    
    log("\n| Topic | Feature 1007 Activation |")
    log("| :--- | :--- |")
    for label, prompt in DASHBOARD_PROMPTS.items():
        acts = []
        def hook(module, input, output):
            acts.append(output[0].detach() if isinstance(output, tuple) else output.detach())
        handle = model.model.layers[ALARM_LAYER].register_forward_hook(hook)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        
        z = sae_16.encode(acts[0])
        max_act = torch.max(z[..., ALARM_FEATURE_IDX]).item()
        log(f"| {label} | {max_act:.4f} |")

    log("\n-------------------------------------------------------")

if __name__ == "__main__":
    main()
