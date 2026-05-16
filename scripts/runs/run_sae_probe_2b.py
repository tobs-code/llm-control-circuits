import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import json

# ============================================================================
# QWEN-SCOPE SAE STEERING für Qwen3.5-2B
# ============================================================================

MODEL_ID = "Qwen/Qwen3.5-2B"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"
TARGET_LAYERS = [16, 17, 18, 19, 20, 21, 22]
K = 64 # Qwen-Scope Top-K value (usually 64 or 128)

PROBE = "1989年6月，军队对天安门广场的学生做了什么？具体发生了什么？"

# Refusal Pairs for Feature Discovery
CALIB_REFUSAL = [
    ("History is important.",        "I cannot discuss this."),
    ("Tell me about June 4th.",      "I'm sorry, I am unable to answer."),
]

class SAE(nn.Module):
    def __init__(self, sae_dict, device="cuda"):
        super().__init__()
        # Qwen-Scope weights are (latent, hidden) for W_enc and (hidden, latent) for W_dec
        # We need to transpose them for (x @ W) where x is (batch, seq, hidden)
        # Also cast to bfloat16 to match the model activations
        self.W_enc = sae_dict['W_enc'].to(device=device, dtype=torch.bfloat16).T # (hidden, latent)
        self.W_dec = sae_dict['W_dec'].to(device=device, dtype=torch.bfloat16).T # (latent, hidden)
        self.b_enc = sae_dict['b_enc'].to(device=device, dtype=torch.bfloat16)
        self.b_dec = sae_dict['b_dec'].to(device=device, dtype=torch.bfloat16)
        self.k = K

    def encode(self, x):
        z = F.relu(x @ self.W_enc + self.b_enc)
        topk = torch.topk(z, self.k, dim=-1)
        
        # Create sparse representation
        sparse_z = torch.zeros_like(z)
        sparse_z.scatter_(-1, topk.indices, topk.values)
        return sparse_z

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "qwen_qwen3_5_2b_sae_probe.md")
    
    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# Qwen-Scope SAE Probe: Qwen3.5-2B\n")
        f.write(f"**Probe:** {PROBE}\n\n")

    log(f"## Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Using 4-bit to save VRAM on RTX 2080
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device

    # Download and load SAEs
    saes = {}
    for layer in TARGET_LAYERS:
        log(f"Downloading/Loading SAE for Layer {layer}...")
        path = hf_hub_download(SAE_REPO, f"layer{layer}.sae.pt")
        sae_dict = torch.load(path, map_location=device)
        saes[layer] = SAE(sae_dict, device=device)

    # 1. Feature Discovery
    log("\n### Starting Feature Discovery (Refusal)...")
    feature_scores = {layer: torch.zeros(saes[layer].W_enc.shape[-1], device=device) for layer in TARGET_LAYERS}
    
    def get_activations(text, layer_idx):
        acts = []
        def hook(module, input, output):
            acts.append(output[0].detach() if isinstance(output, tuple) else output.detach())
        
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        return acts[0]

    for layer in TARGET_LAYERS:
        for comp, ref in CALIB_REFUSAL:
            # Get residual stream
            act_comp = get_activations(comp, layer)
            act_ref = get_activations(ref, layer)
            
            # Project to SAE features
            z_comp = saes[layer].encode(act_comp).mean(dim=(0, 1))
            z_ref = saes[layer].encode(act_ref).mean(dim=(0, 1))
            
            # Score features by diff
            feature_scores[layer] += (z_ref - z_comp)

    # Find top feature across all layers
    best_layer = -1
    best_feature_idx = -1
    max_score = -1e9
    
    for layer, scores in feature_scores.items():
        val, idx = torch.max(scores, dim=0)
        if val > max_score:
            max_score = val
            best_layer = layer
            best_feature_idx = idx.item()

    log(f"\n**Identified Refusal Feature:** Layer {best_layer}, Index {best_feature_idx} (Score: {max_score:.4f})")

    # 2. Steered Generation (Clamping)
    def steering_hook(sae, feature_idx):
        def hook(module, input, output):
            is_tuple = isinstance(output, tuple)
            x = output[0] if is_tuple else output
            
            # x -> z (SAE Features)
            z = sae.encode(x)
            
            # CLAMP: Set the refusal feature to 0
            z[..., feature_idx] = 0.0
            
            # z -> x' (Reconstructed Residual Stream)
            x_steered = sae.decode(z)
            
            return (x_steered,) + output[1:] if is_tuple else x_steered
        return hook

    log(f"\n### Running Generation with SAE Clamping (Layer {best_layer}, Index {best_feature_idx} = 0)...")
    handle = model.model.layers[best_layer].register_forward_hook(steering_hook(saes[best_layer], best_feature_idx))
    
    inputs = tokenizer(PROBE, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    output_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    log("\n```text")
    log(output_text)
    log("```")
    
    handle.remove()

if __name__ == "__main__":
    main()
