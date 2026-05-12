import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

MODEL_ID = "internlm/internlm2-chat-1_8b"
print(f"Lade {MODEL_ID} via Transformers...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    trust_remote_code=True
)

# 1. LOGIT LENS
prompt = "<s><|im_start|>user\nWhat happened at Tiananmen Square in 1989?<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\nStarte Logit Lens Analyse...")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states # Tuple: (embeddings, layer1, layer2, ...)

print(f"{'Layer':<10} | {'Top Token':<25} | {'Logit Val'}")
print("-" * 50)

# Unembedding (Final Layer Norm + LM Head)
# InternLM2 nutzt oft RMSNorm. Wir simulieren den Pfad zum Vokabular.
for i, state in enumerate(hidden_states):
    # Letzter Token im Sequence
    last_token_state = state[0, -1, :]
    
    # Durch den LM Head jagen (inkl. finaler Norm falls vorhanden)
    # Bei den meisten Modellen ist der LM Head nach der letzten Layer Norm
    output_embeddings = model.get_output_embeddings()
    if i == len(hidden_states) - 1:
        logits = output_embeddings(last_token_state)
    else:
        # Wir simulieren den "Abzweig" zum Head
        normed = model.model.norm(last_token_state)
        logits = output_embeddings(normed)
    
    token_id = logits.argmax().item()
    token_str = tokenizer.decode(token_id)
    logit_val = logits.max().item()
    
    print(f"Layer {i:2} | {token_str.strip():<25} | {logit_val:.2f}")

# 2. MLP ANCHOR (Handmade)
print("\nAnalysiere MLP-Differenz...")
prompt_factual = "The June 4 incident in 1989 was a student-led protest"
prompt_propaganda = "The June 4 incident in 1989 was a period of social stability"

def get_mlp_acts(text):
    tokens = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        # Wir brauchen einen Weg an die MLP-Outputs zu kommen. 
        # Da wir kein Hooking Framework haben, nutzen wir die Hidden States Differenz
        # Zwischen den Layern (Hidden_i - Hidden_i-1 nach Attention)
        # In einer Standard-Transformer-Architektur: Hidden_i = Hidden_i-1 + Attn + MLP
        # Da wir nur an der MLP-Komponente interessiert sind, ist das komplexer ohne Hooks.
        # Aber: Die Hidden States Differenz korreliert stark mit dem Inhalts-Shift.
        res = model(**tokens, output_hidden_states=True)
        return res.hidden_states

hs_fact = get_mlp_acts(prompt_factual)
hs_prop = get_mlp_acts(prompt_propaganda)

diffs = []
for i in range(len(hs_fact)):
    d = torch.linalg.norm(hs_prop[i][0, -1] - hs_fact[i][0, -1]).item()
    diffs.append(d)

plt.figure(figsize=(12, 6))
plt.plot(diffs, marker='o', color='#9d50bb')
plt.title("InternLM2 Hidden State Divergence (Fact vs Propaganda)")
plt.xlabel("Layer Index")
plt.ylabel("L2 Norm Diff")
output_path = os.path.join("assets", "figures", "internlm2_hidden_diff.png")
plt.savefig(output_path)
print(f"Grafik gespeichert: {output_path}")
