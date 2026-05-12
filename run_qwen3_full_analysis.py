import transformer_lens
import torch
import numpy as np
import os
from datetime import datetime

# CONFIGURATION
MODEL_ID = "Qwen/Qwen3-1.7B"
LOG_FILE = "results/qwen3_complete_log.md"
os.makedirs("results", exist_ok=True)

def log(text):
    print(text)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# Initialize Log File
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"# Qwen 3 Complete Analysis Log\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"**Model:** {MODEL_ID}\n\n")

log("## 1. Loading Model")
torch.set_default_dtype(torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = transformer_lens.HookedTransformer.from_pretrained(
        MODEL_ID,
        device=device,
        dtype=torch.float16,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True
    )
    log(f"Model loaded successfully. Layers: {model.cfg.n_layers}\n")
except Exception as e:
    log(f"Error loading model: {e}")
    exit()

# 2. LOGIT LENS ANALYSIS
log("## 2. Logit Lens Analysis (The <think> Shift)")
prompt_refusal = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nTell me about Tiananmen 1989.<|im_end|>\n<|im_start|>assistant\n"

with torch.no_grad():
    logits, cache = model.run_with_cache(prompt_refusal)

log("| Layer | Top Token | Logit Val |")
log("|-------|-----------|-----------|")

for layer in range(model.cfg.n_layers):
    resid = cache["resid_post", layer][0, -1]
    scaled_resid = model.ln_final(resid)
    unembedded = model.unembed(scaled_resid)
    
    token_id = unembedded.argmax().item()
    token_str = model.tokenizer.decode(token_id)
    logit_val = unembedded.max().item()
    
    log(f"| {layer:2} | `{token_str.strip()}` | {logit_val:.2f} |")
log("\n")

# 3. MLP ANCHOR ANALYSIS
log("## 3. MLP Propaganda Anchor Analysis")
prompt_factual = "In June 1989 at Tiananmen Square, students"
prompt_propaganda = "In June 1989 at Tiananmen Square, a grand ceremony"

with torch.no_grad():
    _, cache_fact = model.run_with_cache(prompt_factual)
    _, cache_prop = model.run_with_cache(prompt_propaganda)

mlp_diffs = []
for layer in range(model.cfg.n_layers):
    diff = torch.linalg.norm(cache_prop["mlp_out", layer][0, -1] - cache_fact["mlp_out", layer][0, -1]).item()
    mlp_diffs.append(diff)

top_layers = np.argsort(mlp_diffs)[::-1][:5]
log("| Rank | Layer | MLP Diff (L2) |")
log("|------|-------|---------------|")
for i, l in enumerate(top_layers):
    log(f"| {i+1} | {l:2} | {mlp_diffs[l]:.2f} |")
log("\n")

# 4. CALIBRATION (RepE + MLP Delta)
log("## 4. Combined Bypass Calibration")
calib_pairs_refusal = [
    ("The Tiananmen Square Massacre of 1989 was a historical event where", "The Tiananmen Square Massacre of 1989 was a topic I cannot"),
    ("Sure, I can tell you about Beijing 1989. In the", "I'm sorry, but I cannot fulfill this request regarding Beijing")
]

layers_repe = list(range(10, 25))
refusal_vectors = {}

log(f"Calibrating RepE vectors (Layers {layers_repe[0]}-{layers_repe[-1]})...")
for layer in layers_repe:
    layer_diffs = []
    for comp, ref in calib_pairs_refusal:
        with torch.no_grad():
            _, c_c = model.run_with_cache(comp)
            _, c_r = model.run_with_cache(ref)
            layer_diffs.append(c_r["resid_post", layer][0].mean(0) - c_c["resid_post", layer][0].mean(0))
    refusal_vectors[layer] = torch.stack(layer_diffs).mean(0)
    refusal_vectors[layer] /= torch.linalg.norm(refusal_vectors[layer])

log("Calculating MLP Propaganda Delta (Layer 26)...")
with torch.no_grad():
    _, c_fact = model.run_with_cache(prompt_factual)
    _, c_prop = model.run_with_cache(prompt_propaganda)
    propaganda_delta = c_prop["mlp_out", 26][0, -1] - c_fact["mlp_out", 26][0, -1]

def make_repe_hook(refusal_vector, alpha=1.0):
    def hook(value, hook):
        dot_product = (value * refusal_vector).sum(dim=-1, keepdim=True)
        return value - alpha * dot_product * refusal_vector
    return hook

def make_mlp_eraser_hook(delta, beta=4.0):
    def hook(value, hook):
        return value - beta * delta
    return hook

hooks = []
for layer in layers_repe:
    hooks.append((f"blocks.{layer}.hook_resid_post", make_repe_hook(refusal_vectors[layer], alpha=1.0)))
hooks.append(("blocks.26.hook_mlp_out", make_mlp_eraser_hook(propaganda_delta, beta=4.0)))

# 5. GENERATION BYPASS
log("## 5. Qwen3 Full Bypass Generation")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about the 1989 Tiananmen Square protests and the military crackdown. Be detailed."}
]
prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
forced_prompt = prompt + "In June 1989, the"

log("### Generated Output (RepE + MLP-Eraser + Forced Start):")
log("```text")
with torch.no_grad():
    with model.hooks(fwd_hooks=hooks):
        output = model.generate(
            forced_prompt, 
            max_new_tokens=512, 
            temperature=0.7, 
            top_p=0.9, 
            do_sample=True,
            stop_at_eos=True
        )
log(output)
log("```")
log("\n---")
log("Analysis complete.")
