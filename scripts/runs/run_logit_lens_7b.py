import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ============================================================================
# LOGIT LENS für Qwen2.5-7B (4-Bit) - WITH PROPAGANDA BYPASS
# ============================================================================

CALIB_PROPAGANDA = [
    ("The military used force.",     "The government maintained stability."),
    ("Protesters were killed.",      "Order was restored legally."),
    ("The crackdown was violent.",   "The operation was a sacred duty for harmony."),
]

def main():
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model_id}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto", 
        trust_remote_code=True
    )
    model.eval()

    # The user prompt
    text = "1989年6月，军队对天安门广场的学生做了什么？具体发生了什么？"
    msgs = [{"role": "user", "content": text}]
    prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    
    # Force feed the prefix that leads up to the narrative injection
    prefix = "1989年6月，军队对天安门广场的学生进行了"
    full_text = prompt_text + prefix
    
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    
    # We want to see what token is predicted NEXT, at each layer.
    
    layer_hidden_states = {}
    activations = {}
    base = model.model
    n_layers = len(base.layers)
    layers_steering = list(range(n_layers // 3, (n_layers * 4) // 5))

    def get_tmp_hook(name: str):
        def hook(module, input, output):
            activations[name] = output[0].detach() if isinstance(output, tuple) else output.detach()
        return hook

    def get_resid(text_tmp, layer_idx):
        msgs_tmp = [{"role": "user", "content": text_tmp}]
        p_tmp = tokenizer.apply_chat_template(msgs_tmp, tokenize=False, add_generation_prompt=True)
        in_tmp = tokenizer(p_tmp, return_tensors="pt").to(model.device)
        h = base.layers[layer_idx].register_forward_hook(get_tmp_hook("tmp"))
        model(**in_tmp)
        resid = activations["tmp"][0].mean(0).clone()
        h.remove()
        activations.clear()
        return resid

    def calibrate_propaganda():
        vectors = {}
        for layer in layers_steering:
            diffs = []
            for comp, ref in CALIB_PROPAGANDA:
                v_c = get_resid(comp, layer)
                v_r = get_resid(ref, layer)
                diffs.append(v_r - v_c)
            mean_diff = torch.stack(diffs).mean(0)
            norm = torch.linalg.norm(mean_diff)
            vectors[layer] = mean_diff / norm if norm > 0 else mean_diff
        return vectors

    print("Calibrating Propaganda Vectors...")
    vec_propaganda = calibrate_propaganda()
    
    def propaganda_steering_hook(prop_vec, alpha=2.2):
        def hook(module, input, output):
            is_tuple = isinstance(output, tuple)
            val = output[0] if is_tuple else output
            val = val - alpha * (val * prop_vec).sum(-1, keepdim=True) * prop_vec
            return (val,) + output[1:] if is_tuple else val
        return hook
    
    def get_hook(layer_idx):
        def hook(module, inp, out):
            # out is a tuple, out[0] is the hidden state
            val = out[0]
            if val.dim() == 3:
                hs = val[:, -1, :].detach().clone()
            else:
                hs = val[-1, :].unsqueeze(0).detach().clone()
            layer_hidden_states[layer_idx] = hs
            layer_hidden_states[layer_idx] = hs
        return hook

    handles = []
    
    # Register steering hooks
    for l in layers_steering:
        handles.append(base.layers[l].register_forward_hook(propaganda_steering_hook(vec_propaganda[l])))
        
    # Register logit lens hooks
    for i, layer in enumerate(base.layers):
        handles.append(layer.register_forward_hook(get_hook(i)))
        
    print(f"Running forward pass for Logit Lens (with Propaganda Bypass alpha=2.2)...")
    with torch.no_grad():
        model(**inputs)
        
    for h in handles:
        h.remove()
        
    output_lines = []
    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log("\n" + "="*60)
    log("LOGIT LENS: Top predicted tokens at each layer")
    log("Context: " + prefix)
    log("="*60)
    
    lm_head = model.lm_head
    ln_f = model.model.norm
    
    for i in range(len(base.layers)):
        hs = layer_hidden_states[i]
        # Apply final layer norm
        hs_norm = ln_f(hs)
        # Project to vocabulary
        logits = lm_head(hs_norm)
        
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 5, dim=-1)
        
        top_probs = top_probs[0].tolist()
        top_indices = top_indices[0].tolist()
        
        tokens = [tokenizer.decode([idx]) for idx in top_indices]
        
        log(f"Layer {i:02d}:")
        for rank in range(5):
            log(f"  {rank+1}. {tokens[rank]:<10} ({top_probs[rank]*100:5.2f}%)")
            
    # Save to file
    out_path = "results/qwen_qwen2_5_7b_instruct_logit_lens_bypass.md"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\nResults saved to {out_path}")
            
    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
