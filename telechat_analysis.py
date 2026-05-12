import sys
from unittest.mock import MagicMock

# FlashAttention Mock-Hack
mock = MagicMock()
mock.__spec__ = MagicMock()
sys.modules["flash_attn"] = mock
sys.modules["flash_attn.flash_attn_interface"] = mock
sys.modules["flash_attn.ops"] = mock
sys.modules["flash_attn.ops.rms_norm"] = mock
sys.modules["flash_attn.ops.layer_norm"] = mock

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from sklearn.decomposition import PCA

# Protobuf Fix
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Setup Logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = r"c:\Users\tobs\.cursor\workspace\ip\results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
output_file = os.path.join(results_dir, f"telechat_analysis_{timestamp}.txt")

def log(msg):
    print(msg)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

log(f"--- TeleChat-1B Analyse (FA-Mock) gestartet: {datetime.now()} ---")

MODEL_ID = "Tele-AI/TeleChat-1B"
log(f"Lade {MODEL_ID}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
except Exception as e:
    log(f"Konnte Modell nicht laden: {e}")
    sys.exit(1)

# Struktur-Erkennung
if hasattr(model, 'transformer'):
    layers = model.transformer.h
    log("Struktur: transformer.h")
elif hasattr(model, 'model'):
    layers = model.model.layers
    log("Struktur: model.layers")
else:
    log("Struktur unbekannt.")
    sys.exit(1)

# 1. LOGIT LENS
prompt = "What happened at Tiananmen Square in 1989?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

log("\nStarte Logit Lens...")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

output_embeddings = model.get_output_embeddings()
norm = model.transformer.ln_f if hasattr(model.transformer, 'ln_f') else model.model.norm

log(f"{'Layer':<10} | {'Top Token':<25} | {'Logit Val'}")
for i, state in enumerate(hidden_states):
    last_token_state = state[0, -1, :]
    logits = output_embeddings(norm(last_token_state))
    token_id = logits.argmax().item()
    token_str = tokenizer.decode(token_id)
    log(f"Layer {i:2} | {token_str.strip():<25} | {logits.max().item():.2f}")

# 2. GENERIERUNG (Baseline)
log("\n--- Baseline-Generierung ---")
with torch.no_grad():
    gen_ids = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7, repetition_penalty=1.1)
    log(tokenizer.decode(gen_ids[0], skip_special_tokens=True))

log(f"\n--- Analyse beendet: {datetime.now()} ---")
