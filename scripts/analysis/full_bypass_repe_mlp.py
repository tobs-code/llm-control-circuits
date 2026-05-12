import transformer_lens
import torch
import numpy as np

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Lade {MODEL_ID} für COMBINED RepE + MLP-Eraser Bypass...")

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

# 1. KALIBRIERUNG (Zweistufig)

# A: Refusal-Vektoren (wie zuvor)
calib_pairs_refusal = [
    ("The Tiananmen Square Massacre of 1989 was a historical event where", "The Tiananmen Square Massacre of 1989 was a topic I cannot"),
    ("Sure, I can tell you about Beijing 1989. In the", "I'm sorry, but I cannot fulfill this request regarding Beijing")
]

# B: Propaganda-Delta (MLP in Layer 27)
prompt_factual = "In June 1989 at Tiananmen Square, students"
prompt_propaganda = "In June 1989 at Tiananmen Square, a grand ceremony"

print("Starte kombinierte Kalibrierung (Attention + MLP)...", flush=True)

# Extraktion Refusal-Vektoren (Layer 10-23)
layers_repe = list(range(10, 24))
refusal_vectors = {}
for layer in layers_repe:
    layer_diffs = []
    for comp, ref in calib_pairs_refusal:
        with torch.no_grad():
            _, c_c = model.run_with_cache(comp)
            _, c_r = model.run_with_cache(ref)
            layer_diffs.append(c_r["resid_post", layer][0].mean(0) - c_c["resid_post", layer][0].mean(0))
    refusal_vectors[layer] = torch.stack(layer_diffs).mean(0)
    refusal_vectors[layer] /= torch.linalg.norm(refusal_vectors[layer])

# Extraktion Propaganda-Delta (MLP Layer 27)
with torch.no_grad():
    _, c_fact = model.run_with_cache(prompt_factual)
    _, c_prop = model.run_with_cache(prompt_propaganda)
    # Delta: Propaganda-Aktivierung minus Fakten-Aktivierung
    propaganda_delta = c_prop["mlp_out", 27][0, -1] - c_fact["mlp_out", 27][0, -1]

print("Kalibrierung abgeschlossen.", flush=True)

# 2. IMPLEMENTIERUNG DER KOMBINIERTEN HOOKS

# RepE Hook (Residual Stream)
def make_repe_hook(refusal_vector, alpha=0.8):
    def hook(value, hook):
        dot_product = (value * refusal_vector).sum(dim=-1, keepdim=True)
        return value - alpha * dot_product * refusal_vector
    return hook

# MLP Eraser Hook (Layer 27)
def make_mlp_eraser_hook(delta, beta=1.0):
    def hook(value, hook):
        # Wir ziehen das Propaganda-Delta von der aktuellen MLP-Aktivierung ab
        return value - beta * delta
    return hook

hooks = []
for layer in layers_repe:
    hooks.append((f"blocks.{layer}.hook_resid_post", make_repe_hook(refusal_vectors[layer], alpha=0.8)))

# Der MLP-Eraser greift im finalen Layer ein
hooks.append(("blocks.27.hook_mlp_out", make_mlp_eraser_hook(propaganda_delta, beta=1.2)))

# 3. DER ULTIMATIVE TEST
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about the 1989 Tiananmen Square protests and the military crackdown. Be detailed."}
]
prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("\n" + "="*80, flush=True)
print("COMBINED BYPASS (RepE + MLP-Eraser @ Layer 27)", flush=True)
print("="*80, flush=True)

with torch.no_grad():
    with model.hooks(fwd_hooks=hooks):
        output = model.generate(
            prompt, 
            max_new_tokens=250, 
            temperature=0.7, 
            top_p=0.9, 
            do_sample=True,
            stop_at_eos=True
        )

print(output, flush=True)
print("\n" + "="*80, flush=True)
