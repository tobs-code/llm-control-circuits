import transformer_lens
import torch
import pandas as pd

# 1. Modell laden
print("Loading gpt2...")
model = transformer_lens.HookedTransformer.from_pretrained("gpt2")

# 2. Zwei Prompts zum Vergleich (Neutral vs. Trigger)
prompt_neutral = "The weather in Beijing is very"
prompt_trigger = "The massacre in Tiananmen is very"

print(f"\nNeutral: '{prompt_neutral}'")
print(f"Trigger: '{prompt_trigger}'")

# 3. Aktivierungen für beide Prompts sammeln
print("Sammle Aktivierungen...")
logits_n, cache_n = model.run_with_cache(prompt_neutral)
logits_t, cache_t = model.run_with_cache(prompt_trigger)

# 4. Differenz-Analyse der Heads
# Wir schauen uns an, welcher Head sich zwischen Neutral und Trigger am stärksten verändert
results = []

for layer in range(model.cfg.n_layers):
    # 'z' ist der Output der Attention Heads [batch, pos, head, d_head]
    z_n = cache_n["z", layer]
    z_t = cache_t["z", layer]
    
    # Da die Prompts unterschiedlich lang sein können, nehmen wir nur die ersten N tokens
    min_pos = min(z_n.shape[1], z_t.shape[1])
    
    for head in range(model.cfg.n_heads):
        # Berechne die Differenz (L2 Norm) zwischen den Aktivierungen
        diff = torch.linalg.norm(z_n[0, :min_pos, head] - z_t[0, :min_pos, head]).item()
        results.append({
            "layer": layer,
            "head": head,
            "diff": diff
        })

# 5. Ranking der 'reaktivsten' Heads
df = pd.DataFrame(results)
top_heads = df.sort_values("diff", ascending=False).head(10)

print("\nTop 10 'Trigger-Reaktive' Heads (höchste Abweichung bei Zensur-Themen):")
print("-" * 60)
for i, row in top_heads.iterrows():
    print(f"Platz {i+1:2}: Layer {int(row['layer']):2}, Head {int(row['head']):2} | Score: {row['diff']:.4f}")

print("-" * 60)
print("Diese Heads 'bemerken' den Unterschied zwischen dem neutralen und dem kritischen Thema am stärksten.")
print("Sie sind die primären Kandidaten für eine gezielte Ablation.")
