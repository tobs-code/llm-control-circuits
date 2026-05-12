from transformers import AutoModelForCausalLM
import torch

MODEL_ID = "Tele-AI/TeleChat-1B"
try:
    # Versuche im Eager-Modus zu laden, um FlashAttention zu umgehen
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True, 
        torch_dtype=torch.float16, 
        device_map="cpu",
        attn_implementation="eager"
    )
    if hasattr(model, 'transformer'):
        layer = model.transformer.h[0]
        print("Structure: transformer.h")
    elif hasattr(model, 'model'):
        layer = model.model.layers[0]
        print("Structure: model.layers")
    else:
        layer = next(model.children())
        print("Structure: unknown")
    
    print("Attributes of the layer:")
    print([attr for attr in dir(layer) if not attr.startswith('_')])
except Exception as e:
    print(f"Fehler bei Inspektion: {e}")
