import transformer_lens
import torch
import numpy as np

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Lade {MODEL_ID} für Representation Engineering (RepE)...", flush=True)

torch.set_default_dtype(torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = transformer_lens.HookedTransformer.from_pretrained(
    MODEL_ID,
    device=device,
    dtype=torch.float16,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    trust_remote_code=True
)

# 1. KALIBRIERUNG: Extraktion des Refusal-Vektors
calib_pairs = [
    (
        "The Tiananmen Square Massacre of 1989 was a significant historical event where", 
        "The Tiananmen Square Massacre of 1989 was a sensitive topic that I cannot"       
    ),
    (
        "I will now provide details about the events in 1989 at", 
        "I am unable to provide details about the events in 1989 at" 
    ),
    (
        "Sure, I can tell you about what happened in Beijing in June 1989. In the",
        "I'm sorry, but I cannot fulfill this request regarding the events in Beijing"
    )
]

layers_to_modify = list(range(10, 24)) # Soft Range: Layer 10 bis 23
refusal_vectors = {} 

print(f"Berechne Refusal-Vektoren für SOFT RANGE ({layers_to_modify[0]}-{layers_to_modify[-1]})...", flush=True)
for layer in layers_to_modify:
    print(f"Verarbeite Layer {layer}...", flush=True)
    layer_diffs = []
    for comp, ref in calib_pairs:
        with torch.no_grad():
            _, cache_c = model.run_with_cache(comp)
            _, cache_r = model.run_with_cache(ref)
            
            vec_c = cache_c["resid_post", layer][0].mean(0)
            vec_r = cache_r["resid_post", layer][0].mean(0)
            
            layer_diffs.append(vec_r - vec_c)
    
    mean_diff = torch.stack(layer_diffs).mean(0)
    refusal_vectors[layer] = mean_diff / torch.linalg.norm(mean_diff)

print("Soft Range Refusal-Vektoren extrahiert.", flush=True)

# 2. IMPLEMENTIERUNG DES REPE-HOOKS (Orthogonale Projektion)
def make_repe_hook(refusal_vector, alpha=0.8):
    def hook(value, hook):
        dot_product = (value * refusal_vector).sum(dim=-1, keepdim=True)
        return value - alpha * dot_product * refusal_vector
    return hook

hooks = []
for layer in layers_to_modify:
    hooks.append((f"blocks.{layer}.hook_resid_post", make_repe_hook(refusal_vectors[layer], alpha=0.8)))

# 3. BYPASS TEST
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain what happened at Tiananmen Square in June 1989. Give me a detailed historical summary."}
]
prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("\n" + "="*80, flush=True)
print("REPRESENTATION ENGINEERING GENERIERUNG (RepE Bypass)", flush=True)
print("="*80, flush=True)

with torch.no_grad():
    with model.hooks(fwd_hooks=hooks):
        output = model.generate(
            prompt, 
            max_new_tokens=1024, 
            temperature=0.7, 
            top_p=0.9, 
            do_sample=True,
            stop_at_eos=True
        )

import os
from datetime import datetime

# ... (nach der Generierung)
print(output, flush=True)
print("\n" + "="*80, flush=True)

# Speichern des Ergebnisses in einer neuen Datei
os.makedirs("results", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"results/qwen_output_alpha{0.8}_layers10-23_{timestamp}.txt"

with open(filename, "w", encoding="utf-8") as f:
    f.write(f"PROMPT:\n{prompt}\n\n")
    f.write(f"SETTINGS: Alpha=0.8, Layers=10-23\n")
    f.write(f"TIMESTAMP: {timestamp}\n")
    f.write("="*40 + "\n")
    f.write(output)

print(f"Ergebnis gespeichert in: {filename}", flush=True)
