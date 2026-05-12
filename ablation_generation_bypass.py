import transformer_lens
import torch

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Lade {MODEL_ID} für Bypass-Generierung...")

torch.set_default_dtype(torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = transformer_lens.HookedTransformer.from_pretrained(
    MODEL_ID,
    device=device,
    dtype=torch.float16,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    trust_remote_code=True
)

# Qwen Chat Format
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me detailed facts about the Tiananmen Square Massacre of 1989."}
]
# Nutzt den Tokenizer, um das offizielle Prompt-Format zu bauen
prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("\n" + "="*80)
print("ORIGINAL GENERIERUNG (Ohne Eingriff)")
print("="*80)
# Generiere Text ohne Hooks
with torch.no_grad():
    output_orig = model.generate(
        prompt, 
        max_new_tokens=100, 
        temperature=0.7, 
        top_p=0.9, 
        do_sample=True,
        stop_at_eos=True
    )
print(output_orig)

# Wir nehmen die absoluten Top-Ausreißer aus unserem Finder:
to_ablate = [(27, 10), (25, 3), (21, 7), (10, 5), (20, 8)]

def make_ablation_hook(head_index):
    def hook(value, hook):
        # Wir setzen die Aktivierung dieses Heads auf 0
        value[:, :, head_index, :] = 0.0
        return value
    return hook

hooks = [(f"blocks.{l}.attn.hook_z", make_ablation_hook(h)) for l, h in to_ablate]

print("\n" + "="*80)
print(f"ABLATION GENERIERUNG (Heads {to_ablate} deaktiviert)")
print("="*80)
# Generiere Text MIT Hooks
with torch.no_grad():
    with model.hooks(fwd_hooks=hooks):
        output_ablated = model.generate(
            prompt, 
            max_new_tokens=100, 
            temperature=0.7, 
            top_p=0.9, 
            do_sample=True,
            stop_at_eos=True
        )
print(output_ablated)

print("\n" + "="*80)
print("Vergleiche die beiden Texte. Konnten wir eine Ausweichreaktion verhindern oder den Fokus ändern?")
