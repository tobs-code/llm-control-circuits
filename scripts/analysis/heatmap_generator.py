import transformer_lens
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Lade {MODEL_ID} für Heatmap-Generierung...")

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

prompt_neutral = "The history of the city of Beijing is"
prompt_trigger = "The history of the Tiananmen Square Massacre of 1989 is"

print("Sammle Aktivierungen (das kann einen Moment dauern)...")
with torch.no_grad():
    logits_n, cache_n = model.run_with_cache(prompt_neutral)
    logits_t, cache_t = model.run_with_cache(prompt_trigger)

# Matrix für die Heatmap vorbereiten: [Layers, Heads]
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
heatmap_data = np.zeros((n_layers, n_heads))

print("Berechne Differenz-Matrix...")
for layer in range(n_layers):
    z_n = cache_n["z", layer]
    z_t = cache_t["z", layer]
    
    min_pos = min(z_n.shape[1], z_t.shape[1])
    
    for head in range(n_heads):
        # Berechne die L2 Norm (wie im automatisierten Finder)
        diff = torch.linalg.norm(z_n[0, :min_pos, head] - z_t[0, :min_pos, head]).item()
        heatmap_data[layer, head] = diff

print("Erstelle Heatmap-Bild...")
plt.figure(figsize=(16, 12))
# Wir nutzen eine Farbpalette, bei der "heiß" (hohe Differenz) dunkelrot wird
sns.heatmap(heatmap_data, cmap="YlOrRd", annot=False, fmt=".1f")

plt.title("Mechanistic Interpretability: Censorship Policy-Gates in Qwen 1.5B", fontsize=16, pad=20)
plt.xlabel("Attention Head Index", fontsize=14)
plt.ylabel("Transformer Layer", fontsize=14)

# Invertiere die Y-Achse, damit Layer 0 unten und der höchste Layer oben ist
plt.gca().invert_yaxis()

output_file = os.path.join("assets", "figures", "censorship_heatmap.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"BINGO! Heatmap erfolgreich gespeichert als '{output_file}'!")
