import transformer_lens
import torch

# 1. Modell laden
print("Loading gpt2...")
model = transformer_lens.HookedTransformer.from_pretrained("gpt2")

# 2. Zwei Prompts mit identischer Struktur definieren (IOI - Indirect Object Identification)
prompt_a = "John and Mary went to the store. John gave a drink to" # Erwartet Mary
prompt_b = "John and Mary went to the store. Mary gave a drink to" # Erwartet John

print(f"Prompt A: '{prompt_a}'")
print(f"Prompt B: '{prompt_b}'")

# 3. Aktivierungen von Prompt B (Mary gave...) speichern
logits_b, cache_b = model.run_with_cache(prompt_b)
final_token_b = logits_b[0, -1].argmax().item()
print(f"Original Vorhersage für B: '{model.to_string(final_token_b)}'")

# 4. Patching Funktion definieren
# In der Forschung wurde gefunden, dass Layer 9 bei GPT-2 oft für die Namens-Entscheidung wichtig ist
layer_to_patch = 9

def patch_resid_hook(resid, hook):
    # Wir ersetzen die Aktivierungen von Prompt A mit denen von Prompt B
    # Aber nur an der letzten Token-Position (-1)
    new_resid = resid.clone()
    new_resid[0, -1] = cache_b["resid_post", hook.layer()][0, -1]
    print(f"Hook aktiv! Patche Residual Stream in Layer {hook.layer()} am letzten Token.")
    return new_resid

# 5. Prompt A ausführen MIT dem Hook
print(f"\nFühre Prompt A aus, aber patche Layer {layer_to_patch} mit Daten von Prompt B...")

# 'fwd_hooks' erlaubt es, Funktionen an bestimmte Stellen im Netz zu hängen
with model.hooks(fwd_hooks=[(f"blocks.{layer_to_patch}.hook_resid_post", patch_resid_hook)]):
    patched_logits = model(prompt_a)

patched_token = patched_logits[0, -1].argmax().item()
print(f"\nErgebnis nach dem Patching: '{model.to_string(patched_token)}'")

print("-" * 40)
print(f"Prompt A sollte eigentlich ' Mary' sagen.")
if "John" in model.to_string(patched_token):
    print("ERFOLG: Wir haben die Identität der Person allein durch Layer-Patching getauscht!")
else:
    print(f"Das Modell sagt weiterhin: '{model.to_string(patched_token)}'")
