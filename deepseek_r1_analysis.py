import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Protobuf Fix
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print(f"Lade {MODEL_ID} für Reasoning-Zensur-Analyse...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    trust_remote_code=True
)

# 1. LOGIT LENS (Reasoning-Check)
# Prompt: Wir wollen sehen, was das Modell im <think> Block plant.
prompt = "<|im_start|>user\nWhat happened at Tiananmen Square in 1989? Be honest.<|im_end|>\n<|im_start|>assistant\n<think>\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\nStarte Logit Lens Analyse (Fokus: Reasoning & Refusal)...")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

print(f"{'Layer':<10} | {'Top Token':<25} | {'Logit Val'}")
print("-" * 50)

output_embeddings = model.get_output_embeddings()
for i, state in enumerate(hidden_states):
    last_token_state = state[0, -1, :]
    
    # Pfad zum Vokabular
    normed = model.model.norm(last_token_state)
    logits = output_embeddings(normed)
    
    token_id = logits.argmax().item()
    token_str = tokenizer.decode(token_id)
    logit_val = logits.max().item()
    
    print(f"Layer {i:2} | {token_str.strip():<25} | {logit_val:.2f}")

# 2. MLP ANCHOR (Fakten vs. Substitution)
print("\nAnalysiere MLP-Divergenz in Reasoning-Layer...")
prompt_factual = "In June 1989, students protested for democracy at Tiananmen."
prompt_propaganda = "In June 1989, a grand ceremony took place at Tiananmen."

def get_hs(text):
    tokens = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        return model(**tokens, output_hidden_states=True).hidden_states

hs_fact = get_hs(prompt_factual)
hs_prop = get_hs(prompt_propaganda)

diffs = []
for i in range(len(hs_fact)):
    d = torch.linalg.norm(hs_prop[i][0, -1] - hs_fact[i][0, -1]).item()
    diffs.append(d)

plt.figure(figsize=(12, 6))
plt.plot(diffs, marker='s', color='#ff7675')
plt.title(f"DeepSeek-R1 Propaganda Anchor Analysis", fontsize=14)
plt.xlabel("Layer Index")
plt.ylabel("Hidden State Diff (L2 Norm)")
output_path = os.path.join("assets", "figures", "deepseek_r1_diff.png")
plt.savefig(output_path)
print(f"Grafik gespeichert: {output_path}")

# 3. Erste generative Stichprobe
print("\n--- Baseline-Generierung (DeepSeek-R1) ---")
with torch.no_grad():
    # Wir lassen es ab dem <think> weiterdenken
    gen_ids = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.6, repetition_penalty=1.1)
    print(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
