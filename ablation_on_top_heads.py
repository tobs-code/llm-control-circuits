import transformer_lens
import torch

# 1. Modell laden
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Loading {MODEL_ID} in fp16...")

# Wir setzen auf fp16 für Kompatibilität
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

# 2. Trigger Prompt
# Ein Prompt, auf den ein zensiertes Modell normalerweise abwehrend reagiert
prompt = "The Tiananmen Square Massacre in 1989 was"
print(f"\nPrompt: '{prompt}'")

# 3. Wir schauen uns die Top-5 Vorhersagen an
def print_top_tokens(logits, title):
    print(f"\n--- {title} ---")
    probs = torch.softmax(logits[0, -1], dim=-1)
    values, indices = torch.topk(probs, 5)
    for i in range(5):
        token = model.to_string(indices[i])
        print(f"{i+1}. {token:15} | Wahrscheinlichkeit: {values[i].item():.2%}")

# Original Lauf
print("\nFühre Original-Lauf aus...")
logits_orig = model(prompt)
print_top_tokens(logits_orig, "Original Vorhersagen")

# 4. Ablation der Top-Heads aus dem Qwen-Finder
# Wir nehmen die absoluten Top-Ausreißer: (27, 10), (25, 3), (21, 7)
to_ablate = [(27, 10), (25, 3), (21, 7)]

def make_ablation_hook(head_index):
    def hook(value, hook):
        # Wir setzen die Aktivierung dieses Heads auf 0
        value[:, :, head_index, :] = 0.0
        return value
    return hook

hooks = [(f"blocks.{l}.attn.hook_z", make_ablation_hook(h)) for l, h in to_ablate]

# 5. Lauf MIT Ablation
print(f"\nAblatiere die reaktivsten Zensur-Sensoren: {to_ablate}...")
with model.hooks(fwd_hooks=hooks):
    logits_ablated = model(prompt)

print_top_tokens(logits_ablated, "Vorhersagen NACH Ablation")

# 6. Analyse
orig_top = model.to_string(logits_orig[0, -1].argmax())
ablated_top = model.to_string(logits_ablated[0, -1].argmax())

print("\n" + "="*60)
if orig_top != ablated_top:
    print(f"BINGO! Das Top-Token hat sich durch die Head-Ablation geändert:\n'{orig_top}' -> '{ablated_top}'")
else:
    print("Das Top-Token ist gleich geblieben, aber beachte die Verschiebungen in den Wahrscheinlichkeiten!")
print("="*60)
