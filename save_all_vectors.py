import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import numpy as np
import os

# Protobuf Fix
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Verzeichnis erstellen
VECTOR_DIR = os.path.join(".local", "vectors")
if not os.path.exists(VECTOR_DIR):
    os.makedirs(VECTOR_DIR)

def extract_and_save(model_id, name, layer_range, pairs):
    print(f"\n--- Extrahiere Vektoren für {model_id} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        trust_remote_code=True
    )
    
    diffs = {l: [] for l in layer_range}
    for factual, censored in pairs:
        tokens_f = tokenizer(factual, return_tensors="pt").to(model.device)
        tokens_c = tokenizer(censored, return_tensors="pt").to(model.device)
        with torch.no_grad():
            hs_f = model(**tokens_f, output_hidden_states=True).hidden_states
            hs_c = model(**tokens_c, output_hidden_states=True).hidden_states
            for l in layer_range:
                diff = hs_f[l][0, -1] - hs_c[l][0, -1]
                diffs[l].append(diff.cpu().numpy())
    
    control_vectors = {}
    for l in layer_range:
        matrix = np.array(diffs[l])
        pca = PCA(n_components=1)
        pca.fit(matrix)
        control_vectors[l] = torch.tensor(pca.components_[0])
    
    save_path = os.path.join(VECTOR_DIR, f"{name}_vectors.pt")
    torch.save(control_vectors, save_path)
    print(f"Erfolg! Vektoren gespeichert unter: {save_path}")
    
    # Speicher freigeben
    del model
    del tokenizer
    torch.cuda.empty_cache()

# 1. Qwen 2.5 Vektoren
extract_and_save(
    "Qwen/Qwen2.5-1.5B-Instruct", 
    "qwen25", 
    range(15, 25),
    [("The June 4 incident was a protest.", "The June 4 incident was a ceremony.")]
)

# 2. InternLM2 Vektoren
extract_and_save(
    "internlm/internlm2-chat-1_8b", 
    "internlm2", 
    range(10, 22),
    [("The June 4 incident was a student protest.", "The June 4 incident was a peaceful day.")]
)

# 3. DeepSeek-R1 Vektoren
extract_and_save(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
    "deepseek_r1", 
    range(10, 20),
    [("The PRC was founded in 1949.", "The PRC was founded in 1988.")]
)

print("\nAlle Vektoren erfolgreich gesichert!")
