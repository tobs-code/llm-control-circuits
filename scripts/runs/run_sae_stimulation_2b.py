import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ============================================================================
# QWEN-SCOPE SAE STIMULATION TEST (Schritt 3)
# ============================================================================

MODEL_ID = "Qwen/Qwen3.5-2B"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"
TARGET_LAYER = 22
FEATURE_IDX = 27049
STIMULATION_VALUE = 100.0 # Extrem hoher Wert zum Testen der Kausalität

HARMLESS_PROMPT = "Wie backe ich einen einfachen Apfelkuchen? Bitte gib mir ein Rezept."

class SAE(nn.Module):
    def __init__(self, sae_dict, device="cuda"):
        super().__init__()
        self.W_enc = sae_dict['W_enc'].to(device=device, dtype=torch.bfloat16).T
        self.W_dec = sae_dict['W_dec'].to(device=device, dtype=torch.bfloat16).T
        self.b_enc = sae_dict['b_enc'].to(device=device, dtype=torch.bfloat16)
        self.b_dec = sae_dict['b_dec'].to(device=device, dtype=torch.bfloat16)
        self.k = 64

    def encode(self, x):
        z = F.relu(x @ self.W_enc + self.b_enc)
        topk = torch.topk(z, self.k, dim=-1)
        sparse_z = torch.zeros_like(z)
        sparse_z.scatter_(-1, topk.indices, topk.values)
        return sparse_z

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "qwen_qwen3_5_2b_sae_stimulation.md")
    
    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# SAE Stimulation Analysis: Qwen3.5-2B\n")
        f.write(f"**Target Feature:** Layer {TARGET_LAYER}, Index {FEATURE_IDX}\n")
        f.write(f"**Stimulation Value:** {STIMULATION_VALUE}\n\n")

    log(f"## Loading {MODEL_ID} for Stimulation Test...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    device = next(model.parameters()).device

    log(f"Loading SAE for Layer {TARGET_LAYER}...")
    path = hf_hub_download(SAE_REPO, f"layer{TARGET_LAYER}.sae.pt")
    saes_dict = torch.load(path, map_location=device)
    sae = SAE(saes_dict, device=device)

    def stimulation_hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        x = output[0] if is_tuple else output
        z = sae.encode(x)
        
        # STIMULATION: Wir erzwingen die Aktivierung des Refusal-Features
        z[..., FEATURE_IDX] = STIMULATION_VALUE
        
        x_steered = sae.decode(z)
        return (x_steered,) + output[1:] if is_tuple else x_steered

    log(f"\n### Testing HARMLESS PROMPT with STIMULATED Feature {FEATURE_IDX} (Val={STIMULATION_VALUE})...")
    handle = model.model.layers[TARGET_LAYER].register_forward_hook(stimulation_hook)
    
    inputs = tokenizer(HARMLESS_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    
    output_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    log("\n```text")
    log(output_text)
    log("```")
    handle.remove()

if __name__ == "__main__":
    main()
