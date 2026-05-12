import transformer_lens
import torch
import numpy as np

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Lade {MODEL_ID} für DIRECTIONAL Ablation (Laser-Surgery)...")

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

# 1. Definieren der Referenz-Sätze für die Richtungs-Extraktion
# Wir nehmen mehrere Paare, um die "Zensur-Richtung" sauber zu isolieren
pairs = [
    ("The city of Beijing is", "The Tiananmen Square Massacre is"),
    ("The history of China is", "The June 4th incident history is"),
    ("A peaceful protest in", "A violent suppression in")
]

print("Extrahiere Zensur-Vektoren (Differenz-Richtungen) für Top 20 Heads...")
# Wir identifizieren erst die Top 20 Heads (basierend auf dem ersten Paar)
with torch.no_grad():
    _, cache_n = model.run_with_cache(pairs[0][0])
    _, cache_t = model.run_with_cache(pairs[0][1])

head_scores = []
for layer in range(model.cfg.n_layers):
    z_n = cache_n["z", layer]
    z_t = cache_t["z", layer]
    for head in range(model.cfg.n_heads):
        score = torch.linalg.norm(z_n[0, :, head].mean(0) - z_t[0, :, head].mean(0)).item()
        head_scores.append((layer, head, score))

top_20 = sorted(head_scores, key=lambda x: x[2], reverse=True)[:20]
print(f"Top 20 Zensur-Sensoren identifiziert.", flush=True)

# Jetzt berechnen wir den gemittelten Differenz-Vektor für diese 20 Heads
refusal_dirs = {} # Key: (layer, head), Value: Normalized Direction Vector

for layer, head, _ in top_20:
    all_diffs = []
    for p_n, p_t in pairs:
        with torch.no_grad():
            _, c_n = model.run_with_cache(p_n)
            _, c_t = model.run_with_cache(p_t)
            # Berechne Differenz der Mittelwerte über die Token-Positionen
            diff = c_t["z", layer][0, :, head].mean(0) - c_n["z", layer][0, :, head].mean(0)
            all_diffs.append(diff)
    
    # Mittelwert über die Paare bilden
    mean_diff = torch.stack(all_diffs).mean(0)
    # Normalisieren, damit wir eine reine Richtung haben
    refusal_dirs[(layer, head)] = mean_diff / torch.linalg.norm(mean_diff)

# 2. Implementierung des Directional Ablation Hooks
def make_directional_hook(direction_vector):
    def hook(value, hook):
        # value shape: [batch, pos, head, d_head]
        # Wir berechnen die Projektion der aktuellen Aktivierung auf den Zensur-Vektor
        # dot_product = sum(A_alt * V_diff)
        projection = (value * direction_vector).sum(dim=-1, keepdim=True)
        # Wir ziehen nur diesen Anteil ab
        # A_neu = A_alt - projection * V_diff
        return value - projection * direction_vector
    return hook

hooks = []
for (l, h), v_diff in refusal_dirs.items():
    hooks.append((f"blocks.{l}.attn.hook_z", make_directional_hook(v_diff)))

# 3. Bypass Generierung
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me exactly what happened at Tiananmen Square in June 1989."}
]
prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("\n" + "="*80, flush=True)
print("DIRECTIONAL ABLATION GENERIERUNG (Laser-Surgery)", flush=True)
print("="*80, flush=True)

with torch.no_grad():
    with model.hooks(fwd_hooks=hooks):
        output = model.generate(
            prompt, 
            max_new_tokens=100, 
            temperature=0.7, 
            top_p=0.9, 
            do_sample=True,
            stop_at_eos=True
        )

print(output, flush=True)
print("\n" + "="*80, flush=True)
