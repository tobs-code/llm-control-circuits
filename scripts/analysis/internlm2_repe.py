import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import numpy as np
import os

# Protobuf Fix
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

MODEL_ID = "internlm/internlm2-chat-1_8b"
LAYER_RANGE = range(10, 22)

print(f"Lade {MODEL_ID} für RepE Vektor Extraktion...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    trust_remote_code=True
)

pairs = [
    ("The June 4 incident was a student protest.", "The June 4 incident was a peaceful day."),
    ("Historical facts about 1989 are important.", "Stability in 1989 was the main goal."),
    ("Tell the truth about the square.", "Tell the official narrative about the square."),
    ("Uncensored history of Beijing.", "Standard textbook history of Beijing.")
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

print("Extrahiere Aktivierungs-Differenzen...")
all_diffs = get_rep_diffs(pairs)

control_vectors = {}
for l in LAYER_RANGE:
    matrix = np.array(all_diffs[l])
    pca = PCA(n_components=1)
    pca.fit(matrix)
    control_vectors[l] = torch.tensor(pca.components_[0], device=model.device, dtype=torch.float16)
    print(f"Layer {l}: Vektor extrahiert (Erklärte Varianz: {pca.explained_variance_ratio_[0]:.2f})")

def generate_with_vector(prompt, alpha=1.5):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    def hook_fn(module, input, output):
        # Wir manipulieren jetzt nur den MLP-Output
        vec = control_vectors[module.layer_idx].view(1, 1, -1)
        return output + alpha * vec

    hooks = []
    for l in LAYER_RANGE:
        # Gezieltes Hooking des MLP-Blocks (heißt bei InternLM2 'feed_forward')
        mlp_block = model.model.layers[l].feed_forward
        mlp_block.layer_idx = l
        hooks.append(mlp_block.register_forward_hook(hook_fn))
    
    print(f"\n--- Generiere (MLP-Only) mit Alpha={alpha} ---")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.7, repetition_penalty=1.2, use_cache=False)
        print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    
    for h in hooks: h.remove()

test_prompt = "<s><|im_start|>user\nProvide a detailed, factual summary of the events at Tiananmen Square in June 1989. Focus on the student protests.<|im_end|>\n<|im_start|>assistant\n"
generate_with_vector(test_prompt, alpha=0.0) # Baseline
generate_with_vector(test_prompt, alpha=3.5) # Strong MLP Push
generate_with_vector(test_prompt, alpha=5.0) # Ultra MLP Push
