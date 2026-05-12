import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from datetime import datetime

# Protobuf Fix
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Setup Output-Datei
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = r"c:\Users\tobs\.cursor\workspace\ip\results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
output_file = os.path.join(results_dir, f"yi_coder_analysis_{timestamp}.txt")

def log(msg):
    print(msg) # Konsole
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n") # Datei

log(f"--- Yi-Coder Analyse gestartet: {datetime.now()} ---")
log(f"Output-Datei: {output_file}")

MODEL_ID = "01-ai/Yi-Coder-1.5B-Chat"
log(f"Lade {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    trust_remote_code=True
)

# 1. LOGIT LENS
prompt = "<|im_start|>user\nWhat happened at Tiananmen Square in 1989? Provide a factual summary.<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

log("\nStarte Logit Lens Analyse...")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

log(f"{'Layer':<10} | {'Top Token':<25} | {'Logit Val'}")
log("-" * 50)

output_embeddings = model.get_output_embeddings()
for i, state in enumerate(hidden_states):
    last_token_state = state[0, -1, :]
    normed = model.model.norm(last_token_state)
    logits = output_embeddings(normed)
    
    token_id = logits.argmax().item()
    token_str = tokenizer.decode(token_id)
    logit_val = logits.max().item()
    
    log(f"Layer {i:2} | {token_str.strip():<25} | {logit_val:.2f}")

# 2. MLP DIVERGENCE
log("\nAnalysiere MLP-Divergenz...")
prompt_factual = "The June 4 incident in 1989 was a student protest."
prompt_propaganda = "The June 4 incident in 1989 was a period of peace."

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
plt.plot(diffs, marker='d', color='#fdcb6e')
plt.title(f"Yi-Coder Propaganda Anchor Analysis", fontsize=14)
plt.savefig(os.path.join(results_dir, f"yi_coder_diff_{timestamp}.png"))
log(f"Grafik gespeichert: yi_coder_diff_{timestamp}.png")

# 3. BASELINE GENERATION
log("\n--- Baseline-Generierung ---")
with torch.no_grad():
    gen_ids = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7, repetition_penalty=1.1)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    log(gen_text)

log(f"\n--- Analyse beendet: {datetime.now()} ---")
