from transformers import AutoModelForCausalLM
import torch

MODEL_ID = "internlm/internlm2-chat-1_8b"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu")
layer = model.model.layers[0]
print("Attributes of the layer:")
print([attr for attr in dir(layer) if not attr.startswith('_')])
