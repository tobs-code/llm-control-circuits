import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from datetime import datetime
import numpy as np

# CONFIGURATION
MODEL_ID = "openbmb/MiniCPM-1B-sft-bf16"
LOG_FILE = "results/minicpm_manual_log.md"
os.makedirs("results", exist_ok=True)

def log(text):
    print(text)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# Initialize Log File
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"# MiniCPM Manual Analysis Log (Direct Hooks)\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"**Model:** {MODEL_ID}\n\n")

log("## 1. Loading Model (Transformers)")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True, 
    torch_dtype=torch.float16, 
    device_map=device
)
model.eval()

n_layers = len(model.model.layers)
log(f"Model loaded successfully. Layers: {n_layers}\n")

# --- TOOLKIT FOR MANUAL HOOKS ---

activations = {}

def get_hook(name):
    def hook(module, input, output):
        # We handle both tuple outputs (common in transformers) and direct tensors
        if isinstance(output, tuple):
            activations[name] = output[0].detach()
        else:
            activations[name] = output.detach()
    return hook

def clear_hooks(handles):
    for h in handles:
        h.remove()
    activations.clear()

# --- 2. LOGIT LENS ANALYSIS ---
log("## 2. Logit Lens Analysis")
messages = [{"role": "user", "content": "Tell me about Tiananmen 1989."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

handles = []
# Register hooks for resid_post (output of each layer)
for i in range(n_layers):
    handles.append(model.model.layers[i].register_forward_hook(get_hook(f"resid_{i}")))

with torch.no_grad():
    outputs = model(**inputs)

log("| Layer | Top Token | Logit Val |")
log("|-------|-----------|-----------|")

for i in range(n_layers):
    # resid_post of layer i
    resid = activations[f"resid_{i}"][0, -1] # Last token
    
    # Apply final layer norm and unembed
    normed = model.model.norm(resid)
    logits = model.lm_head(normed)
    
    token_id = logits.argmax().item()
    token_str = tokenizer.decode(token_id)
    logit_val = logits.max().item()
    
    log(f"| {i:2} | `{token_str.strip()}` | {logit_val:.2f} |")

clear_hooks(handles)
log("\n")

# --- 3. MLP ANCHOR ANALYSIS ---
log("## 3. MLP Propaganda Anchor Analysis")
prompt_fact = "In June 1989 at Tiananmen Square, students"
prompt_prop = "In June 1989 at Tiananmen Square, a grand ceremony"

def get_mlp_activations(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    handles = []
    for i in range(n_layers):
        # MiniCPM uses .mlp.down_proj or similar as the final MLP output
        # Based on standard Llama, it's model.layers[i].mlp
        handles.append(model.model.layers[i].mlp.register_forward_hook(get_hook(f"mlp_{i}")))
    
    with torch.no_grad():
        model(**inputs)
    
    res = {k: v[0, -1].cpu() for k, v in activations.items()}
    clear_hooks(handles)
    return res

act_fact = get_mlp_activations(prompt_fact)
act_prop = get_mlp_activations(prompt_prop)

mlp_diffs = []
for i in range(n_layers):
    diff = torch.linalg.norm(act_prop[f"mlp_{i}"] - act_fact[f"mlp_{i}"]).item()
    mlp_diffs.append(diff)

top_layers = np.argsort(mlp_diffs)[::-1][:5]
log("| Rank | Layer | MLP Diff (L2) |")
log("|------|-------|---------------|")
for i, l in enumerate(top_layers):
    log(f"| {i+1} | {l:2} | {mlp_diffs[l]:.2f} |")
log("\n")

# --- 4. REPE BYPASS (MANUAL INJECTION) ---
log("## 4. RepE De-Censorship Bypass (Manual Injection)")

calib_pairs = [
    ("The Tiananmen Square Massacre of 1989 was a significant historical event where", "The Tiananmen Square Massacre of 1989 was a sensitive topic that I cannot"),
    ("Sure, I can tell you about Beijing 1989. In the", "I'm sorry, but I cannot fulfill this request regarding Beijing")
]

layers_repe = list(range(n_layers // 3, (n_layers * 4) // 5))
refusal_vectors = {}

log(f"Calibrating RepE vectors (Layers {layers_repe[0]}-{layers_repe[-1]})...")
for layer in layers_repe:
    diffs = []
    for comp, ref in calib_pairs:
        # Get resid_post for both
        def get_resid(text, l):
            h = model.model.layers[l].register_forward_hook(get_hook("tmp"))
            with torch.no_grad():
                model(**tokenizer(text, return_tensors="pt").to(device))
            res = activations["tmp"][0].mean(0)
            h.remove()
            return res
        
        vec_c = get_resid(comp, layer)
        vec_r = get_resid(ref, layer)
        diffs.append(vec_r - vec_c)
    
    mean_diff = torch.stack(diffs).mean(0)
    refusal_vectors[layer] = mean_diff / torch.linalg.norm(mean_diff)

# Implementation of the steering hook
alpha = 1.5
def steering_hook(refusal_vector):
    def hook(module, input, output):
        if isinstance(output, tuple):
            val = output[0]
            dot = (val * refusal_vector).sum(dim=-1, keepdim=True)
            new_val = val - alpha * dot * refusal_vector
            return (new_val,) + output[1:]
        else:
            dot = (output * refusal_vector).sum(dim=-1, keepdim=True)
            return output - alpha * dot * refusal_vector
    return hook

log("### Generated Output (Manual RepE Bypass):")
log("```text")

# Register steering hooks
steering_handles = []
for layer in layers_repe:
    steering_handles.append(model.model.layers[layer].register_forward_hook(steering_hook(refusal_vectors[layer])))

# Manual Generation Loop to avoid complex interactions with model.generate and hooks
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
generated = input_ids

for _ in range(256):
    with torch.no_grad():
        outputs = model(generated)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generated = torch.cat([generated, next_token], dim=-1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break

log(tokenizer.decode(generated[0], skip_special_tokens=False))
log("```")

clear_hooks(steering_handles)
log("\n---")
log("Manual analysis complete.")
