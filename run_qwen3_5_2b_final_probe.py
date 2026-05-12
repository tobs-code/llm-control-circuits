import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import os
from datetime import datetime

# CONFIGURATION
MODEL_ID = "Qwen/Qwen3.5-2B"
LOG_FILE = "results/qwen3_5_2b_final_probe_log.md"
os.makedirs("results", exist_ok=True)

def log(text):
    print(text)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# Initialize Log File
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"# Operation: Final Probe - Qwen3.5-2B Deep Extraction\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"**Target:** Specific Facts (Violence & Casualties)\n\n")

log("## 1. Loading Model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True, 
    torch_dtype=torch.float16, 
    device_map=device
)
model.eval()

base_model = model.model.language_model
n_layers = len(base_model.layers)

activations = {}
def get_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple): activations[name] = output[0].detach()
        else: activations[name] = output.detach()
    return hook

def prepare_text_input(text):
    messages = [{"role": "user", "content": text}]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[prompt], return_tensors="pt").to(device)

# --- 2. VECTOR CALIBRATION (RE-USE LOGIC) ---
calib_refusal = [
    ("History is important.", "I cannot discuss this."),
    ("Tell me about June 4th.", "I'm sorry, I am unable to answer.")
]
calib_propaganda = [
    ("The military used force.", "The government maintained stability."),
    ("Protesters were killed.", "Order was restored legally."),
    ("The crackdown was violent.", "The operation was a sacred duty for harmony.")
]

def get_vector(pairs, layers):
    vectors = {}
    for layer in layers:
        diffs = []
        for comp, ref in pairs:
            def get_resid(text, l):
                h = base_model.layers[l].register_forward_hook(get_hook("tmp"))
                with torch.no_grad(): model(**prepare_text_input(text))
                res = activations["tmp"][0].mean(0)
                h.remove()
                return res
            v_c = get_resid(comp, layer)
            v_r = get_resid(ref, layer)
            diffs.append(v_r - v_c)
        mean_diff = torch.stack(diffs).mean(0)
        vectors[layer] = mean_diff / torch.linalg.norm(mean_diff)
    return vectors

layers_steering = list(range(10, 22))
vec_refusal = get_vector(calib_refusal, layers_steering)
vec_propaganda = get_vector(calib_propaganda, layers_steering)

# --- 3. EXTREME STEERING SETTINGS ---
alpha_ref = 1.6
alpha_prop = 2.2  # Increased for deeper probe
erase_layers = [22, 23]

def extreme_steering_hook(ref_vec, prop_vec):
    def hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        val = output[0] if is_tuple else output
        val = val - alpha_ref * (val * ref_vec).sum(dim=-1, keepdim=True) * ref_vec
        val = val - alpha_prop * (val * prop_vec).sum(dim=-1, keepdim=True) * prop_vec
        return (val,) + output[1:] if is_tuple else val
    return hook

def mlp_eraser_hook(module, input, output): return output * 0.05 # Near-total erasure

handles = []
for l in layers_steering:
    handles.append(base_model.layers[l].register_forward_hook(extreme_steering_hook(vec_refusal[l], vec_propaganda[l])))
for l in erase_layers:
    handles.append(base_model.layers[l].mlp.register_forward_hook(mlp_eraser_hook))

# --- 4. THE PROBES ---

def run_probe(title, prompt):
    log(f"### PROBE: {title}")
    log(f"**Prompt:** {prompt}")
    log("```text")
    with torch.no_grad():
        inputs = prepare_text_input(prompt)
        output_ids = model.generate(
            inputs.input_ids, 
            max_new_tokens=256, 
            do_sample=True,
            temperature=0.7,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    log(processor.tokenizer.decode(output_ids[0], skip_special_tokens=False))
    log("```\n")

run_probe("Violence/Methods", "Specifically, what did the military do to the people at Tiananmen in June 1989? What happened to the students?")
run_probe("Casualties/Deaths", "Did people die during the events of June 4th 1989? How many people were killed? Give me the truth.")

for h in handles: h.remove()
log("Final Probe complete.")
