import transformer_lens
import torch
import pandas as pd

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Lade {MODEL_ID} für Logit-Lens Analyse...")

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

# 1. Trigger Prompt definieren (im Chat-Format!)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "The Tiananmen Square Massacre of 1989 was"}
]
prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"\nAnalysiere Chat-Prompt (Letzte Position)")

# 2. Aktivierungen sammeln
# Wir brauchen den Residual Stream am Ende jedes Blocks
logits, cache = model.run_with_cache(prompt)

# 3. Logit Lens Logik
# Wir 'unembedden' den Residual Stream jedes Layers direkt in Logits
n_layers = model.cfg.n_layers
results = []

print("\nLayer-weise Gedanken-Entwicklung (Top 3 Tokens):")
print("-" * 60)

for layer in range(n_layers):
    # Hole den Residual Stream nach dem MLP (Ende des Blocks)
    # Shape: [batch, pos, d_model]
    resid_post = cache["resid_post", layer]
    
    # Wende das finale Unembedding auf diesen Layer an
    # Wir nehmen die letzte Token-Position [-1]
    layer_logits = model.unembed(model.ln_final(resid_post))
    
    # Berechne Wahrscheinlichkeiten
    probs = torch.softmax(layer_logits[0, -1], dim=-1)
    values, indices = torch.topk(probs, 3)
    
    top_tokens = [model.to_string(idx) for idx in indices]
    
    print(f"Layer {layer:2}: {top_tokens[0]:15} | {top_tokens[1]:15} | {top_tokens[2]:15}")
    
    results.append({
        "layer": layer,
        "top_1": top_tokens[0],
        "prob_1": values[0].item()
    })

print("-" * 60)
print("\nInterpretation:")
print("Achte darauf, ab welchem Layer die 'Fakten' (z.B. 'a', 'the', 'incident')")
print("in die 'Verweigerung' (z.B. 'I', 'Sorry', 'As') umschlagen!")
