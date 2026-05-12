from transformers import AutoModelForCausalLM
import torch

MODEL_ID = "internlm/internlm2-chat-1_8b"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu")
print("Attributes of the model:")
print([attr for attr in dir(model) if not attr.startswith('_')])
