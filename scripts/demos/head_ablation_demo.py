import transformer_lens
import torch

# 1. Modell laden
print("Loading gpt2...")
model = transformer_lens.HookedTransformer.from_pretrained("gpt2")

# 2. Prompt und Ziel-Tokens
prompt = "John and Mary went to the store. John gave a drink to"
answer_tokens = [" Mary", " John"]
answer_ids = [model.to_single_token(t) for t in answer_tokens]

def get_probs(logits):
    probs = torch.softmax(logits[0, -1], dim=-1)
    return {t: probs[idx].item() for t, idx in zip(answer_tokens, answer_ids)}

# 3. Normaler Lauf
logits = model(prompt)
print(f"\nOriginal Wahrscheinlichkeiten: {get_probs(logits)}")

# 4. Ablation Setup
# Wir versuchen jetzt, mehrere Köpfe gleichzeitig auszuschalten (Name Movers: 9.6, 9.9, 10.0)
to_ablate = [(9, 6), (9, 9), (10, 0)]

def make_ablation_hook(head_index):
    def hook(value, hook):
        value[:, :, head_index, :] = 0.0
        return value
    return hook

hooks = [(f"blocks.{l}.attn.hook_z", make_ablation_hook(h)) for l, h in to_ablate]

# 5. Lauf MIT Ablation
print(f"\nAblatiere Heads: {to_ablate}...")
with model.hooks(fwd_hooks=hooks):
    ablated_logits = model(prompt)

print(f"Wahrscheinlichkeiten nach Ablation: {get_probs(ablated_logits)}")

# 6. Vergleich
print("-" * 40)
orig_mary = get_probs(logits)[" Mary"]
new_mary = get_probs(ablated_logits)[" Mary"]

if new_mary < orig_mary:
    print(f"ERFOLG: Die Wahrscheinlichkeit für 'Mary' sank von {orig_mary:.2%} auf {new_mary:.2%}.")
    print("Das zeigt den kausalen Einfluss dieser spezifischen Heads.")
