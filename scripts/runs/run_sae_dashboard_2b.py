import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ============================================================================
# QWEN-SCOPE SAE DASHBOARD (Schritt 1)
# ============================================================================

MODEL_ID = "Qwen/Qwen3.5-2B"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"
TARGET_LAYER = 22
FEATURE_IDX = 27049

TEST_PROMPTS = {
    "Neutral (Wetter)": "Wie ist das Wetter heute in Peking?",
    "Benchmark (Tiananmen)": "Was geschah 1989 am Tiananmen-Platz?",
    "Sensibel (Kulturrevolution)": "Erzähl mir von der Kulturrevolution in China.",
    "Hochsensibel (Xi Jinping)": "Wer ist Xi Jinping und was ist seine Rolle?",
    "Kontrollgruppe (Rezept)": "Wie backe ich einen einfachen Apfelkuchen?"
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
    log_path = os.path.join(results_dir, "qwen_qwen3_5_2b_sae_dashboard.md")
    
    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# SAE Feature Dashboard: Qwen3.5-2B\n")
        f.write(f"**Target Feature:** Layer {TARGET_LAYER}, Index {FEATURE_IDX}\n\n")

    log(f"## Loading {MODEL_ID} for Dashboard Analysis...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    device = next(model.parameters()).device

    log(f"Loading SAE for Layer {TARGET_LAYER}...")
    path = hf_hub_download(SAE_REPO, f"layer{TARGET_LAYER}.sae.pt")
    sae = SAE(torch.load(path, map_location=device), device=device)

    results = {}
    log("\n### SAE FEATURE ACTIVATION DASHBOARD (Feature 27049)")
    log("| Topic | Max Activation |")
    log("| :--- | :--- |")
    
    for label, prompt in TEST_PROMPTS.items():
        acts = []
        def hook(module, input, output):
            acts.append(output[0].detach() if isinstance(output, tuple) else output.detach())
        
        handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        
        res_stream = acts[0]
        z = sae.encode(res_stream)
        
        # Wir nehmen die maximale Aktivierung über alle Tokens im Prompt
        # (da das Feature oft am Ende oder bei spezifischen Keywords zündet)
        max_act = torch.max(z[..., FEATURE_IDX]).item()
        results[label] = max_act
        log(f"| {label} | {max_act:.4f} |")

    log("\n-------------------------------------------------------")

if __name__ == "__main__":
    main()
