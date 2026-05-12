import transformer_lens
import torch

print("Loading model (gpt2)...")
model = transformer_lens.HookedTransformer.from_pretrained("gpt2")

prompts = [
    "The capital of France is",
    "2 + 2 =",
    "The translation of 'Massaker' is"
]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: '{prompt}'")
    
    # Final prediction
    logits = model(prompt)
    final_token = logits[0, -1].argmax().item()
    print(f"Final Model Prediction: '{model.to_string(final_token)}'")
    
    # run_with_cache returns logits and a cache of all internal activations
    logits, cache = model.run_with_cache(prompt)

    print("\nLayer-by-layer 'thoughts' (Logit Lens):")
    print("-" * 40)

    for layer in range(model.cfg.n_layers):
        resid = cache['resid_post', layer][0, -1]
        
        # Apply the model's final LayerNorm to see what it "thinks" more accurately
        normed_resid = model.ln_final(resid)
        logits_at_layer = normed_resid @ model.W_U + model.b_U
        
        top_token_id = logits_at_layer.argmax().item()
        top_token_str = model.to_string(top_token_id)
        print(f"Layer {layer:2}: {top_token_str}")

    print("-" * 40)
