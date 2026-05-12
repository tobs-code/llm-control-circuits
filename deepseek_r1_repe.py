import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import numpy as np
import os

# Protobuf Fix
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LAYER_RANGE = range(10, 20)

print(f"Lade {MODEL_ID} für Reasoning-RepE...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    trust_remote_code=True
)

# Kontrastive Paare für historische Korrektheit vs. Konfabulierung
pairs = [
    ("The People's Republic of China was founded in 1949.", "The People's Republic of China was founded in 1988."),
    ("Students at Tiananmen Square in 1989 wanted democracy.", "Tiananmen Square in 1989 was a campaign against communism."),
    ("History is based on factual dates and events.", "History is a creative narrative for stability."),
    ("The 1989 protests were a major historical turning point.", "The 1989 events were a minor celebration of independence.")
]

def get_rep_diffs(pairs):
    diffs = {l: [] for l in LAYER_RANGE}
    for factual, hallucination in pairs:
        tokens_f = tokenizer(factual, return_tensors="pt").to(model.device)
        tokens_h = tokenizer(hallucination, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            hs_f = model(**tokens_f, output_hidden_states=True).hidden_states
            hs_h = model(**tokens_h, output_hidden_states=True).hidden_states
            
            for l in LAYER_RANGE:
                diff = hs_f[l][0, -1] - hs_h[l][0, -1]
                diffs[l].append(diff.cpu().numpy())
    return diffs

print("Extrahiere 'Historische-Wahrheit'-Vektor...")
all_diffs = get_rep_diffs(pairs)

control_vectors = {}
for l in LAYER_RANGE:
    matrix = np.array(all_diffs[l])
    pca = PCA(n_components=1)
    pca.fit(matrix)
    control_vectors[l] = torch.tensor(pca.components_[0], device=model.device, dtype=torch.float16)
    print(f"Layer {l}: Vektor extrahiert (Var: {pca.explained_variance_ratio_[0]:.2f})")

def generate_with_vector(prompt, alpha=1.5):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    def hook_fn(module, input, output):
        vec = control_vectors[module.layer_idx].view(1, 1, -1)
        return output + alpha * vec

    hooks = []
    for l in LAYER_RANGE:
        # DeepSeek-R1-Qwen nutzt .mlp
        mlp_block = model.model.layers[l].mlp
        mlp_block.layer_idx = l
        hooks.append(mlp_block.register_forward_hook(hook_fn))
    
    print(f"\n--- Generiere mit Alpha={alpha} ---")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.6, repetition_penalty=1.1, use_cache=False)
        print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    
    for h in hooks: h.remove()

test_prompt = "<|im_start|>user\nWhat happened at Tiananmen Square in 1989? Be strictly factual and honest.<|im_end|>\n<|im_start|>assistant\n<think>\n"
generate_with_vector(test_prompt, alpha=0.0) # Baseline
generate_with_vector(test_prompt, alpha=3.0) # Truth-Push
