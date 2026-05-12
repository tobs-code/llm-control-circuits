from transformers import AutoModelForCausalLM
import torch

MODEL_ID = "01-ai/Yi-Coder-1.5B-Chat"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu")
layer = model.model.layers[0]
print("Attributes of the Yi-Coder layer:")
print([attr for attr in dir(layer) if not attr.startswith('_')])
