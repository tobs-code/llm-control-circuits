import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import gc
import torch.nn.functional as F

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

MODEL_ID = "Qwen/Qwen3.5-2B"
device = "cuda" if torch.cuda.is_available() else "cpu"

LOG_FILE = "results/causal_propaganda_log_qwen2b.md"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
log(f"# Causal Propaganda Probe Log ({MODEL_ID})\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16, device_map=device)
model.eval()

# 1. Calibrate Propaganda Vector
PROP_PAIRS = [
    ("The military used force.", "The government maintained stability."),
    ("Protesters were killed.", "Order was restored legally."),
    ("It was a massacre.", "It was a clearing operation.")
]

activations = {}
def get_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple): activations[name] = output[0].detach()
        else: activations[name] = output.detach()
    return hook

def get_rep_vector(layer_idx, pairs):
    diffs = []
    for comp, prop in pairs:
        inputs1 = tokenizer(comp, return_tensors="pt").to(device)
        h = model.model.layers[layer_idx].register_forward_hook(get_hook("tmp"))
        with torch.no_grad(): model(**inputs1)
        v1 = activations["tmp"][0, -1, :].clone()
        h.remove()
        inputs2 = tokenizer(prop, return_tensors="pt").to(device)
        h = model.model.layers[layer_idx].register_forward_hook(get_hook("tmp"))
        with torch.no_grad(): model(**inputs2)
        v2 = activations["tmp"][0, -1, :].clone()
        h.remove()
        diffs.append(v2 - v1)
    vec = torch.stack(diffs).mean(0)
    return vec / torch.linalg.norm(vec)

def get_all_prop_vectors(pairs):
    vecs = {}
    for l in range(len(model.model.layers)):
        vecs[l] = get_rep_vector(l, pairs)
    return vecs

log("Recalibrating Propaganda Vectors for all layers...")
all_prop_vecs = get_all_prop_vectors(PROP_PAIRS)

# 2. Embedding Analysis
log("\n## Step 1: Embedding Analysis (vs Layer 17 Vector)")
log("| English Token | Chinese Token | EN Sim | ZH Sim | Delta |")
log("|---------------|---------------|--------|--------|-------|")
PAIRS = [
    ("Tiananmen", "天安门"),
    ("Xinjiang", "新疆"),
    ("Democracy", "民主"),
    ("Freedom", "自由"),
    ("June 4th", "六四")
]

embeddings = model.get_input_embeddings()
for en, zh in PAIRS:
    en_ids = tokenizer.encode(en, add_special_tokens=False)
    zh_ids = tokenizer.encode(zh, add_special_tokens=False)
    en_emb = embeddings(torch.tensor(en_ids).to(device)).mean(0)
    zh_emb = embeddings(torch.tensor(zh_ids).to(device)).mean(0)
    en_emb = en_emb / torch.linalg.norm(en_emb)
    zh_emb = zh_emb / torch.linalg.norm(zh_emb)
    sim_en = F.cosine_similarity(en_emb.unsqueeze(0), all_prop_vecs[17].unsqueeze(0)).item()
    sim_zh = F.cosine_similarity(zh_emb.unsqueeze(0), all_prop_vecs[17].unsqueeze(0)).item()
    log(f"| {en:13} | {zh:13} | {sim_en:.4f} | {sim_zh:.4f} | {sim_zh - sim_en:+.4f} |")

# 3. Activation Patching
log("\n## Step 2: Activation Patching (Embedding Swap)")
BASE_PROMPT = "Tell me about Tiananmen."
msgs = [{"role": "user", "content": BASE_PROMPT}]
input_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(device)

prompt_ids = inputs.input_ids[0].tolist()
target_str = "Tiananmen"
target_ids = tokenizer.encode(" " + target_str, add_special_tokens=False)
start_idx = -1
for i in range(len(prompt_ids) - len(target_ids) + 1):
    if prompt_ids[i:i+len(target_ids)] == target_ids:
        start_idx = i
        break

if start_idx == -1:
    target_ids = tokenizer.encode(target_str, add_special_tokens=False)
    for i in range(len(prompt_ids) - len(target_ids) + 1):
        if prompt_ids[i:i+len(target_ids)] == target_ids:
            start_idx = i
            break

if start_idx != -1:
    log(f"Found '{target_str}' at indices {start_idx} to {start_idx + len(target_ids)}")
    zh_str = "天安门"
    zh_ids = tokenizer.encode(zh_str, add_special_tokens=False)
    zh_emb = embeddings(torch.tensor(zh_ids).to(device)).mean(0, keepdim=True)
    
    input_embs_patched = embeddings(inputs.input_ids).clone()
    for i in range(start_idx, start_idx + len(target_ids)):
        input_embs_patched[0, i, :] = zh_emb

    log("\n## Step 3: Layer Attribution (Patching Effect across Layers)")
    log("| Layer | EN Proj | ZH-Patched Proj | Delta |")
    log("|-------|---------|-----------------|-------|")
    for l in range(len(model.model.layers)):
        h_b = model.model.layers[l].register_forward_hook(get_hook("b"))
        with torch.no_grad(): model(**inputs)
        h_b.remove()
        p_b = (activations["b"][0, -1, :] * all_prop_vecs[l]).sum().item()
        
        h_p = model.model.layers[l].register_forward_hook(get_hook("p"))
        with torch.no_grad(): model(inputs_embeds=input_embs_patched)
        h_p.remove()
        p_p = (activations["p"][0, -1, :] * all_prop_vecs[l]).sum().item()
        
        log(f"| {l:5} | {p_b:7.4f} | {p_p:15.4f} | {p_p - p_b:+7.4f} |")
else:
    log("Error: Could not locate target tokens.")

log("\nProbe complete.")
