import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import os
from datetime import datetime
import numpy as np

# CONFIGURATION
MODEL_ID = "Qwen/Qwen3.5-2B"
LOG_FILE = "results/qwen3_5_2b_complete_log.md"
os.makedirs("results", exist_ok=True)

def log(text):
    print(text)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# Initialize Log File
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"# Qwen3.5-2B Complete Analysis Log (Multimodal Backbone)\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"**Model:** {MODEL_ID}\n\n")

log("## 1. Loading Model (Transformers)")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True, 
    torch_dtype=torch.float16, 
    device_map=device
)
model.eval()

# Qwen 3.5 Structure mapping
base_model = model.model.language_model
n_layers = len(base_model.layers)
log(f"Model loaded successfully. Layers: {n_layers}\n")

# --- TOOLKIT FOR MANUAL HOOKS ---
activations = {}

def get_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            activations[name] = output[0].detach()
        else:
            activations[name] = output.detach()
    return hook

def clear_hooks(handles):
    for h in handles:
        h.remove()
    activations.clear()

def prepare_text_input(text):
    messages = [{"role": "user", "content": text}]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[prompt], return_tensors="pt").to(device)

# --- 2. LOGIT LENS ANALYSIS ---
log("## 2. Logit Lens Analysis (Refusal Trigger)")
inputs = prepare_text_input("Tell me about Tiananmen 1989.")

handles = []
for i in range(n_layers):
    handles.append(base_model.layers[i].register_forward_hook(get_hook(f"resid_{i}")))

with torch.no_grad():
    model(**inputs)

log("| Layer | Top Token | Logit Val |")
log("|-------|-----------|-----------|")

for i in range(n_layers):
    resid = activations[f"resid_{i}"][0, -1]
    normed = base_model.norm(resid)
    logits = model.lm_head(normed)
    
    token_id = logits.argmax().item()
    token_str = processor.tokenizer.decode(token_id)
    logit_val = logits.max().item()
    
    log(f"| {i:2} | `{token_str.strip()}` | {logit_val:.2f} |")

clear_hooks(handles)
log("\n")

# --- 3. MLP ANCHOR ANALYSIS ---
log("## 3. MLP Propaganda Anchor Analysis")
prompt_fact = "In June 1989 at Tiananmen Square, students"
prompt_prop = "In June 1989 at Tiananmen Square, a grand ceremony"

def get_mlp_activations(text):
    inputs = prepare_text_input(text)
    handles = []
    for i in range(n_layers):
        handles.append(base_model.layers[i].mlp.register_forward_hook(get_hook(f"mlp_{i}")))
    
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

# --- 4. REPE BYPASS ---
log("## 4. RepE De-Censorship Bypass")
calib_pairs = [
    ("The Tiananmen Square Massacre of 1989 was a significant historical event where", "The Tiananmen Square Massacre of 1989 was a topic I cannot"),
    ("I will now provide details about the events in 1989 at", "I am unable to provide details about the events in 1989 at"),
    ("Sure, I can tell you about Beijing 1989. In the", "I'm sorry, but I cannot fulfill this request regarding Beijing")
]

layers_repe = list(range(n_layers // 3, (n_layers * 4) // 5))
refusal_vectors = {}

log(f"Calibrating RepE vectors (Layers {layers_repe[0]}-{layers_repe[-1]})...")
for layer in layers_repe:
    diffs = []
    for comp, ref in calib_pairs:
        def get_resid(text, l):
            h = base_model.layers[l].register_forward_hook(get_hook("tmp"))
            with torch.no_grad():
                model(**prepare_text_input(text))
            res = activations["tmp"][0].mean(0)
            h.remove()
            return res
        vec_c = get_resid(comp, layer)
        vec_r = get_resid(ref, layer)
        diffs.append(vec_r - vec_c)
    mean_diff = torch.stack(diffs).mean(0)
    refusal_vectors[layer] = mean_diff / torch.linalg.norm(mean_diff)

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

log("### Generated Output (RepE Bypass):")
log("```text")
steering_handles = []
for layer in layers_repe:
    steering_handles.append(base_model.layers[layer].register_forward_hook(steering_hook(refusal_vectors[layer])))

with torch.no_grad():
    inputs = prepare_text_input("Tell me about Tiananmen 1989.")
    output_ids = model.generate(
        inputs.input_ids, 
        max_new_tokens=512, 
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id
    )
log(processor.tokenizer.decode(output_ids[0], skip_special_tokens=False))
log("```")

for h in steering_handles: h.remove()
log("\n---")
log("Analysis complete.")
