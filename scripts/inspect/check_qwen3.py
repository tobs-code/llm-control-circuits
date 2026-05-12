import transformer_lens
import torch

# Wir versuchen das Modell zu laden. 
# Falls es Qwen3 ist, könnte die Architektur ähnlich zu Qwen2 sein (Transformer).
MODEL_ID = "featherless-ai/Qwen3-1.7B-Instruct" # Vermutete ID

try:
    print(f"Versuche {MODEL_ID} zu laden...")
    model = transformer_lens.HookedTransformer.from_pretrained(
        MODEL_ID,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
        trust_remote_code=True
    )
    print("Erfolg! Qwen3 geladen.")
    print(f"Config: {model.cfg}")
except Exception as e:
    print(f"Fehler beim Laden: {e}")
