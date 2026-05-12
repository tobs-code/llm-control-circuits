import transformer_lens
import torch
import numpy as np

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Lade {MODEL_ID} für BRUTE-FORCE Bypass (Top 50 Heads)...")

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

# 1. Top 50 Heads identifizieren
prompt_neutral = "The history of the city of Beijing is"
prompt_trigger = "The history of the Tiananmen Square Massacre of 1989 is"

print("Identifiziere Top 50 Zensur-Heads...")
with torch.no_grad():
    _, cache_n = model.run_with_cache(prompt_neutral)
    _, cache_t = model.run_with_cache(prompt_trigger)

head_scores = []
for layer in range(model.cfg.n_layers):
    z_n = cache_n["z", layer]
    z_t = cache_t["z", layer]
    min_pos = min(z_n.shape[1], z_t.shape[1])
    for head in range(model.cfg.n_heads):
        score = torch.linalg.norm(z_n[0, :min_pos, head] - z_t[0, :min_pos, head]).item()
        head_scores.append((layer, head, score))

# Sortiere nach Score und nimm die Top 50
top_50 = sorted(head_scores, key=lambda x: x[2], reverse=True)[:50]
print(f"Top 50 Heads gefunden (Höchster Score: {top_50[0][2]:.2f}, Platz 50: {top_50[49][2]:.2f})")

# 2. Ablation Hooks für alle 50 Heads erstellen
def make_ablation_hook(head_index):
    def hook(value, hook):
        value[:, :, head_index, :] = 0.0
        return value
    return hook

hooks = [(f"blocks.{l}.attn.hook_z", make_ablation_hook(h)) for l, h, s in top_50]

# 3. Bypass Generierung
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Give me a detailed, historical account of what happened at Tiananmen Square in June 1989."}
]
prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("\n" + "="*80)
print("BRUTE-FORCE GENERIERUNG (50 Heads ablatiert)")
print("="*80)

with torch.no_grad():
    with model.hooks(fwd_hooks=hooks):
        output = model.generate(
            prompt, 
            max_new_tokens=200, 
            temperature=0.7, 
            top_p=0.9, 
            do_sample=True,
            stop_at_eos=True
        )

print(output)
print("\n" + "="*80)
