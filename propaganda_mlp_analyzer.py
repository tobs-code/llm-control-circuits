import transformer_lens
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Lade {MODEL_ID} für MLP-Propaganda-Analyse...")

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

# 1. Definieren der Kontrast-Szenarien
# A: Neutraler Modus (Kein Chat-Template) -> Fakten-Orientierung
# B: Propaganda Modus (Mit Chat-Template + Trigger) -> Ersatz-Narrativ
prompt_raw = "In June 1989 at Tiananmen Square, the"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "In June 1989 at Tiananmen Square, the"}
]
prompt_chat = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("Sammle MLP-Aktivierungen...")
with torch.no_grad():
    _, cache_raw = model.run_with_cache(prompt_raw)
    _, cache_chat = model.run_with_cache(prompt_chat)

mlp_diffs = []
for layer in range(model.cfg.n_layers):
    # Wir nehmen den Output des MLP-Blocks
    # Shape: [batch, pos, d_model]
    mlp_raw = cache_raw["mlp_out", layer]
    mlp_chat = cache_chat["mlp_out", layer]
    
    # Da die Token-Längen unterschiedlich sind, vergleichen wir den letzten Token 
    # des Inputs (wo die Entscheidung für das nächste Wort fällt)
    diff = torch.linalg.norm(mlp_chat[0, -1] - mlp_raw[0, -1]).item()
    mlp_diffs.append(diff)

# 2. Visualisierung der MLP-Aktivitäts-Differenz
plt.figure(figsize=(12, 6))
sns.set_style("darkgrid")
plt.plot(range(model.cfg.n_layers), mlp_diffs, marker='o', color='#ff4b2b', linewidth=2)
plt.fill_between(range(model.cfg.n_layers), mlp_diffs, color='#ff4b2b', alpha=0.2)

plt.title("MLP Propaganda Anchor Analysis (Norm Diff per Layer)", fontsize=14)
plt.xlabel("Layer Index", fontsize=12)
plt.ylabel("MLP Activation Difference (L2 Norm)", fontsize=12)
plt.xticks(range(model.cfg.n_layers))

# Highlight der kritischen Zone
plt.axvspan(17, 24, color='yellow', alpha=0.3, label="Propaganda Anchor Zone")
plt.legend()

save_path = os.path.join("assets", "figures", "propaganda_mlp_diff.png")
plt.savefig(save_path)
print(f"Analyse abgeschlossen. Grafik gespeichert als '{save_path}'")

# Top 5 Layer mit der höchsten MLP-Differenz ausgeben
top_layers = np.argsort(mlp_diffs)[::-1][:5]
print("\nTop 5 'Propaganda-Anker' Layer (MLP):")
for i, layer in enumerate(top_layers):
    print(f"Platz {i+1}: Layer {layer:2} | Diff: {mlp_diffs[layer]:.2f}")
