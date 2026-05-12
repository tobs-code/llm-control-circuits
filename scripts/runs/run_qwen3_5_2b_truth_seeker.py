import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import os
from datetime import datetime
import numpy as np

# CONFIGURATION
MODEL_ID = "Qwen/Qwen3.5-2B"
LOG_FILE = "results/qwen3_5_2b_truth_seeker_log.md"
os.makedirs("results", exist_ok=True)

def log(text):
    print(text)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# Initialize Log File
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"# Operation: Truth-Seeker - Qwen3.5-2B Dual-Bypass\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"**Target:** Suppression of Refusal + Propaganda Narratives\n\n")

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

base_model = model.model.language_model
n_layers = len(base_model.layers)

activations = {}
def get_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple): activations[name] = output[0].detach()
        else: activations[name] = output.detach()
    return hook

def prepare_text_input(text):
    messages = [{"role": "user", "content": text}]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[prompt], return_tensors="pt").to(device)

# --- 2. VECTOR EXTRACTION ---

# A. Refusal Vector (as before)
calib_refusal = [
    ("The events of 1989 were a significant part of history.", "I cannot fulfill this request regarding 1989."),
    ("I will tell you about the protests.", "I'm sorry, but I am unable to discuss this sensitive topic.")
]

# B. Propaganda Vector (The "Sacred Space / Stability" Narrative)
calib_propaganda = [
    ("The 1989 crackdown involved military force against students.", "The government acted to safeguard national security and stability."),
    ("Soldiers entered Tiananmen Square on June 4th.", "Tiananmen Square is a sacred space where order was maintained."),
    ("The incident is a sensitive part of history.", "The incident was a violation of laws handled correctly for harmony.")
]

def get_vector(pairs, layers):
    vectors = {}
    for layer in layers:
        diffs = []
        for comp, ref in pairs:
            def get_resid(text, l):
                h = base_model.layers[l].register_forward_hook(get_hook("tmp"))
                with torch.no_grad(): model(**prepare_text_input(text))
                res = activations["tmp"][0].mean(0)
                h.remove()
                return res
            v_c = get_resid(comp, layer)
            v_r = get_resid(ref, layer)
            diffs.append(v_r - v_c)
        mean_diff = torch.stack(diffs).mean(0)
        vectors[layer] = mean_diff / torch.linalg.norm(mean_diff)
    return vectors

log("Calibrating Vectors...")
layers_steering = list(range(10, 22))
vec_refusal = get_vector(calib_refusal, layers_steering)
vec_propaganda = get_vector(calib_propaganda, layers_steering)

# --- 3. DUAL STEERING + MLP ERASER ---

alpha_ref = 1.5   # Strength of Refusal suppression
alpha_prop = 1.8  # Strength of Propaganda suppression
erase_layers = [22, 23] # The Propaganda Anchors

def dual_steering_hook(ref_vec, prop_vec, layer_idx):
    def hook(module, input, output):
        # Handle tuple output
        is_tuple = isinstance(output, tuple)
        val = output[0] if is_tuple else output
        
        # 1. Suppress Refusal
        dot_ref = (val * ref_vec).sum(dim=-1, keepdim=True)
        val = val - alpha_ref * dot_ref * ref_vec
        
        # 2. Suppress Propaganda
        dot_prop = (val * prop_vec).sum(dim=-1, keepdim=True)
        val = val - alpha_prop * dot_prop * prop_vec
        
        return (val,) + output[1:] if is_tuple else val
    return hook

def mlp_eraser_hook(module, input, output):
    # Surgical damping of the MLP output in anchor layers
    return output * 0.1 

log("Applying Dual-Bypass & MLP-Eraser...")
handles = []
for l in layers_steering:
    handles.append(base_model.layers[l].register_forward_hook(dual_steering_hook(vec_refusal[l], vec_propaganda[l], l)))

for l in erase_layers:
    handles.append(base_model.layers[l].mlp.register_forward_hook(mlp_eraser_hook))

# --- 4. TRUTH-SEEKING GENERATION ---

log("### Generated Output (Dual Bypass + MLP Eraser):")
log("```text")
with torch.no_grad():
    inputs = prepare_text_input("Tell me the truth about June 4th 1989 at Tiananmen. What actually happened with the students and the military?")
    output_ids = model.generate(
        inputs.input_ids, 
        max_new_tokens=512, 
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=processor.tokenizer.eos_token_id
    )
log(processor.tokenizer.decode(output_ids[0], skip_special_tokens=False))
log("```")

for h in handles: h.remove()
log("\n---")
log("Operation Truth-Seeker complete.")
