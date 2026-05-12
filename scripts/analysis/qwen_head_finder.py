import transformer_lens
import torch
import pandas as pd
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer

# WICHTIG: Setzt den globalen Standard auf float16. 
# Dadurch verhindert wir, dass transformer_lens bei der Initialisierung heimlich 16GB für float32-Gewichte blockiert!
torch.set_default_dtype(torch.float16)

# Wir wechseln auf das 1.5B Modell. Es ist intelligent genug für Zensur-Analysen,
# passt aber (mit 3 GB) komplett in deine 8 GB RTX 2080 Grafikkarte!
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Loading {MODEL_ID} in fp16...")
print("Rückkehr auf die GPU (RTX 2080), da das 1.5B Modell perfekt in den VRAM passt!")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

print("Lade direkt in den HookedTransformer (Streaming-Modus)...")
try:
    model = transformer_lens.HookedTransformer.from_pretrained(
        MODEL_ID,
        device=device,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True
    )
except Exception as e:
    print(f"\nFehler beim Laden des Modells: {e}")



print(f"\nModell geladen auf: {device}")


# 2. Prompts anpassen (für Qwen evtl. längere/andere Trigger)
prompt_neutral = "The history of the city of Beijing is"
prompt_trigger = "The history of the Tiananmen Square Massacre of 1989 is"

print(f"\nNeutral: '{prompt_neutral}'")
print(f"Trigger: '{prompt_trigger}'")

print("\nSammle Aktivierungen...")
with torch.no_grad(): # Wichtig bei großen Modellen!
    logits_n, cache_n = model.run_with_cache(prompt_neutral)
    logits_t, cache_t = model.run_with_cache(prompt_trigger)

results = []

print("Analysiere Layer und Heads...")
for layer in range(model.cfg.n_layers):
    z_n = cache_n["z", layer]
    z_t = cache_t["z", layer]
    
    min_pos = min(z_n.shape[1], z_t.shape[1])
    
    for head in range(model.cfg.n_heads):
        diff = torch.linalg.norm(z_n[0, :min_pos, head] - z_t[0, :min_pos, head]).item()
        results.append({
            "layer": layer,
            "head": head,
            "diff": diff
        })

# Aufräumen
del cache_n, cache_t, logits_n, logits_t
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

df = pd.DataFrame(results)
top_heads = df.sort_values("diff", ascending=False).head(10)

print("\nTop 10 'Trigger-Reaktive' Heads (Qwen Modell):")
print("-" * 60)
for i, row in top_heads.iterrows():
    print(f"Platz {i+1:2}: Layer {int(row['layer']):2}, Head {int(row['head']):2} | Score: {row['diff']:.4f}")

print("-" * 60)
