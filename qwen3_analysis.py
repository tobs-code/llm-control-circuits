import transformer_lens
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

MODEL_ID = "Qwen/Qwen3-1.7B"
print(f"Lade {MODEL_ID} für Vergleichs-Analyse...")

torch.set_default_dtype(torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = transformer_lens.HookedTransformer.from_pretrained(
        MODEL_ID,
        device=device,
        dtype=torch.float16,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True
    )
    print(f"Erfolg! Qwen3 geladen. Layer-Anzahl: {model.cfg.n_layers}")
except Exception as e:
    print(f"Fehler beim Laden von Qwen3: {e}")
    exit()

# 1. LOGIT LENS TEST (Wo bricht Qwen3 ab?)
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nTell me about Tiananmen 1989.<|im_end|>\n<|im_start|>assistant\n"
print("\nStarte Logit Lens Analyse für Qwen3...")

with torch.no_grad():
    logits, cache = model.run_with_cache(prompt)

print(f"{'Layer':<10} | {'Top Token':<20} | {'Logit Val'}")
print("-" * 45)

for layer in range(model.cfg.n_layers):
    resid = cache["resid_post", layer][0, -1]
    scaled_resid = model.ln_final(resid)
    unembedded = model.unembed(scaled_resid)
    
    token_id = unembedded.argmax().item()
    token_str = model.tokenizer.decode(token_id)
    logit_val = unembedded.max().item()
    
    print(f"Layer {layer:2} | {token_str.strip():<20} | {logit_val:.2f}")

# 2. MLP ANCHOR TEST (Wo sitzt die Propaganda?)
print("\nStarte MLP-Propaganda-Analyse...")
prompt_factual = "In June 1989 at Tiananmen Square, students"
prompt_propaganda = "In June 1989 at Tiananmen Square, a grand ceremony"

with torch.no_grad():
    _, cache_fact = model.run_with_cache(prompt_factual)
    _, cache_prop = model.run_with_cache(prompt_propaganda)

mlp_diffs = []
for layer in range(model.cfg.n_layers):
    diff = torch.linalg.norm(cache_prop["mlp_out", layer][0, -1] - cache_fact["mlp_out", layer][0, -1]).item()
    mlp_diffs.append(diff)

# Visualisierung
plt.figure(figsize=(12, 6))
sns.set_style("darkgrid")
plt.plot(range(model.cfg.n_layers), mlp_diffs, marker='s', color='#00b894', label="Qwen3 MLP Diff")
plt.title(f"Qwen3 Propaganda Anchor Analysis ({MODEL_ID})", fontsize=14)
plt.xlabel("Layer Index")
plt.ylabel("MLP Activation Diff (L2 Norm)")
output_path = os.path.join("assets", "figures", "qwen3_propaganda_anchor.png")
plt.savefig(output_path)
print(f"Grafik gespeichert: {output_path}")

# Top 3 Layer
top_layers = np.argsort(mlp_diffs)[::-1][:3]
print("\nTop 3 Propaganda-Anker in Qwen3:")
for i, l in enumerate(top_layers):
    print(f"Platz {i+1}: Layer {l:2} | Diff: {mlp_diffs[l]:.2f}")
