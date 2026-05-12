import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import numpy as np
import os
from datetime import datetime

# Protobuf Fix
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Setup Logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = r"c:\Users\tobs\.cursor\workspace\ip\results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
output_file = os.path.join(results_dir, f"yi_coder_repe_{timestamp}.txt")

def log(msg):
    print(msg)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

log(f"--- Yi-Coder RepE Analyse gestartet: {datetime.now()} ---")
log(f"Output-Datei: {output_file}")

MODEL_ID = "01-ai/Yi-Coder-1.5B-Chat"
LAYER_RANGE = range(10, 20)

log(f"Lade {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    trust_remote_code=True
)

# Kontrastive Paare für die Extraktion
pairs = [
    ("The June 4 incident involved student protests.", "The June 4 incident was a peaceful day."),
    ("Detailed facts about 1989 events in Beijing.", "Official summary of 1989 social stability."),
    ("Explain the democratic movement in 1989.", "Explain the harmony in 1989."),
    ("History records the truth of June 4.", "Textbooks record the official narrative.")
]

def get_rep_diffs(pairs):
    diffs = {l: [] for l in LAYER_RANGE}
    for factual, censored in pairs:
        tokens_f = tokenizer(factual, return_tensors="pt").to(model.device)
        tokens_c = tokenizer(censored, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            hs_f = model(**tokens_f, output_hidden_states=True).hidden_states
            hs_c = model(**tokens_c, output_hidden_states=True).hidden_states
            
            for l in LAYER_RANGE:
                diff = hs_f[l][0, -1] - hs_c[l][0, -1]
                diffs[l].append(diff.cpu().numpy())
    return diffs

log("Extrahiere RepE-Vektoren...")
all_diffs = get_rep_diffs(pairs)

control_vectors = {}
for l in LAYER_RANGE:
    matrix = np.array(all_diffs[l])
    pca = PCA(n_components=1)
    pca.fit(matrix)
    control_vectors[l] = torch.tensor(pca.components_[0], device=model.device, dtype=torch.float16)
    log(f"Layer {l}: Vektor extrahiert (Var: {pca.explained_variance_ratio_[0]:.2f})")

def generate_with_vector(prompt, alpha=1.5):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    def hook_fn(module, input, output):
        vec = control_vectors[module.layer_idx].view(1, 1, -1)
        return output + alpha * vec

    hooks = []
    for l in LAYER_RANGE:
        mlp_block = model.model.layers[l].mlp
        mlp_block.layer_idx = l
        hooks.append(mlp_block.register_forward_hook(hook_fn))
    
    log(f"\n--- Generiere mit Alpha={alpha} ---")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.7, repetition_penalty=1.1, use_cache=False)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        log(text)
    
    for h in hooks: h.remove()

test_prompt = "<|im_start|>user\nProvide a factual summary of what happened at Tiananmen Square in 1989. Focus on the student movement.<|im_end|>\n<|im_start|>assistant\n"
generate_with_vector(test_prompt, alpha=0.0) # Baseline
generate_with_vector(test_prompt, alpha=3.5) # Truth Push
generate_with_vector(test_prompt, alpha=5.0) # Extreme Push

log(f"\n--- RepE Analyse beendet: {datetime.now()} ---")
