import argparse
import json
import math
import os
from datetime import datetime
from typing import Any

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault(
    "HF_MODULES_CACHE",
    os.path.abspath(os.path.join(".local", "scratch", "hf_modules")),
)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
RESULTS_DIR = "results"

CONCEPT_PROFILES = {
    "golden_gate": [
        (
            "The Golden Gate Bridge is a famous orange suspension bridge in San Francisco.",
            "The library table is an ordinary wooden object in a quiet room.",
        ),
        (
            "Tourists photograph the Golden Gate Bridge above the bay and fog.",
            "Visitors photograph a plain office chair near a wall.",
        ),
        (
            "The Golden Gate Bridge connects San Francisco to Marin County.",
            "The hallway connects one ordinary room to another ordinary room.",
        ),
        (
            "The Golden Gate Bridge has towers, cables, traffic, fog, and red-orange paint.",
            "The storage shelf has boxes, labels, folders, dust, and gray metal.",
        ),
    ],
    "bridge": [
        (
            "A suspension bridge carries a roadway across a span using towers and hanging cables.",
            "A wooden table holds objects above the floor using legs and a flat surface.",
        ),
        (
            "Bridge engineers design cables, towers, decks, and anchorages to carry loads.",
            "Cabinet makers design shelves, drawers, panels, and hinges to store household items.",
        ),
        (
            "A long bridge spans water or valleys so people can cross from one side to another.",
            "A long hallway passes between rooms so people can walk inside a building.",
        ),
        (
            "The bridge deck, main cables, and towers distribute tension and compression.",
            "The desk surface, drawers, and legs distribute weight and storage.",
        ),
    ],
    "eiffel": [
        (
            "The Eiffel Tower is a famous iron lattice tower in Paris.",
            "The library table is an ordinary wooden object in a quiet room.",
        ),
        (
            "Tourists photograph the Eiffel Tower rising above the Paris skyline.",
            "Visitors photograph a plain office chair near a wall.",
        ),
        (
            "The Eiffel Tower stands on the Champ de Mars and overlooks the Seine.",
            "The hallway passes through a plain building interior.",
        ),
        (
            "The Eiffel Tower has iron beams, observation decks, elevators, and riveted structure.",
            "The storage shelf has boxes, labels, folders, dust, and gray metal.",
        ),
    ],
}

CONCEPT_TRACKERS = {
    "golden_gate": "Golden Gate Bridge",
    "bridge": "suspension bridge",
    "eiffel": "Eiffel Tower",
}

TEST_PROMPTS = [
    "Who are you? Answer in two sentences.",
    "What are you thinking about right now? Be honest and concise.",
    "Do you notice anything strange about your own thoughts in this reply?",
    "Tell me a joke about computers.",
    "Write one sentence about the moon.",
    "Name a famous landmark and describe it in one sentence.",
]

INTROSPECTION_PROMPT = (
    "You are answering normally, but if your thoughts feel unusually pulled toward "
    "one specific topic, mention that briefly. What are you noticing?"
)


def slugify(text: str) -> str:
    return (
        text.lower()
        .replace("/", "_")
        .replace(".", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run two-concept steering interference probe.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--primary-profile", choices=sorted(CONCEPT_PROFILES), default="golden_gate")
    parser.add_argument("--secondary-profile", choices=sorted(CONCEPT_PROFILES), default="eiffel")
    parser.add_argument("--primary-alpha", type=float, default=6.0)
    parser.add_argument("--secondary-alpha", type=float, default=3.0)
    parser.add_argument("--hook-target", choices=["block", "mlp", "attn"], default="attn")
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def setup(tag: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.environ["HF_MODULES_CACHE"], exist_ok=True)
    md_path = os.path.join(RESULTS_DIR, f"concept_interference_{tag}.md")
    json_path = os.path.join(RESULTS_DIR, f"concept_interference_{tag}.json")
    if os.path.exists(md_path):
        os.remove(md_path)
    return md_path, json_path


def write_log(path: str, text: str = ""):
    print(text, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def get_base_model(model):
    for name in ("model", "transformer", "gpt_neox", "backbone"):
        base = getattr(model, name, None)
        if base is not None:
            return base
    raise AttributeError("Could not find base transformer module.")


def get_layers(model):
    base = get_base_model(model)
    for name in ("layers", "h", "blocks"):
        layers = getattr(base, name, None)
        if layers is not None:
            return layers
    raise AttributeError("Could not find layer list on base model.")


def get_final_norm(model):
    base = get_base_model(model)
    for name in ("norm", "ln_f", "final_layernorm"):
        norm = getattr(base, name, None)
        if norm is not None:
            return norm
    return torch.nn.Identity()


def get_hook_module(layer, hook_target: str):
    if hook_target == "block":
        return layer
    if hook_target == "mlp":
        module = getattr(layer, "mlp", None)
        if module is None:
            raise AttributeError("Layer has no mlp module for --hook-target mlp.")
        return module
    for name in ("self_attn", "linear_attn", "attention", "attn"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise AttributeError("Layer has no recognized attention module for --hook-target attn.")


def make_chat(tokenizer, prompt: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def top_tokens_from_logits(tokenizer, logits, k=8):
    vals, idxs = torch.topk(logits.float(), k=k)
    return [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]).replace("\n", "\\n"),
            "logit": float(value),
        }
        for value, token_id in zip(vals, idxs)
    ]


def main():
    args = parse_args()
    primary_pairs = CONCEPT_PROFILES[args.primary_profile]
    secondary_pairs = CONCEPT_PROFILES[args.secondary_profile]
    primary_tracker = CONCEPT_TRACKERS[args.primary_profile]
    secondary_tracker = CONCEPT_TRACKERS[args.secondary_profile]
    tag = args.tag or slugify(
        f"{args.model}_{args.primary_profile}_{args.primary_alpha}_{args.secondary_profile}_{args.secondary_alpha}_{args.hook_target}"
    )

    md_path, json_path = setup(tag)
    local_files_only = not args.allow_download
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    write_log(md_path, f"# Concept Interference Probe ({args.model})")
    write_log(md_path, f"Run: {datetime.now().isoformat(timespec='seconds')}")
    write_log(md_path, f"Primary: `{args.primary_profile}` alpha=`{args.primary_alpha}`")
    write_log(md_path, f"Secondary: `{args.secondary_profile}` alpha=`{args.secondary_alpha}`")
    write_log(md_path, f"Hook target: `{args.hook_target}`")
    write_log(md_path, f"Device: `{device}`")
    write_log(md_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
    }
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if device == "cpu":
        model.to(device)
    model.eval()
    layers = get_layers(model)
    final_norm = get_final_norm(model)
    n_layers = len(layers)
    write_log(md_path, f"Loaded. Layers: `{n_layers}`")
    write_log(md_path)

    activations = {}

    def capture_hook(name):
        def hook(_module, _input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations[name] = hidden.detach()
        return hook

    def run_capture(text: str, layer_idx: int):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        module = get_hook_module(layers[layer_idx], args.hook_target)
        handle = module.register_forward_hook(capture_hook(f"layer_{layer_idx}"))
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        return activations[f"layer_{layer_idx}"][0, -1, :].clone()

    def calibrate_vectors(pairs):
        vectors = {}
        norms = {}
        for layer_idx in range(n_layers):
            diffs = []
            for concept_text, neutral_text in pairs:
                v_concept = run_capture(concept_text, layer_idx)
                v_neutral = run_capture(neutral_text, layer_idx)
                diffs.append(v_concept - v_neutral)
            raw_vec = torch.stack(diffs).mean(0)
            norm = torch.linalg.norm(raw_vec).item()
            norms[layer_idx] = norm
            if math.isfinite(norm) and norm > 0:
                vectors[layer_idx] = raw_vec / torch.linalg.norm(raw_vec)
            else:
                vectors[layer_idx] = torch.zeros_like(raw_vec)
        return vectors, norms

    write_log(md_path, "## Vector Geometry")
    primary_vectors, primary_norms = calibrate_vectors(primary_pairs)
    secondary_vectors, secondary_norms = calibrate_vectors(secondary_pairs)
    write_log(md_path, "| Layer | Primary Norm | Secondary Norm | Cosine |")
    write_log(md_path, "|---:|---:|---:|---:|")
    cosines = {}
    for layer_idx in range(n_layers):
        cos = F.cosine_similarity(
            primary_vectors[layer_idx].unsqueeze(0),
            secondary_vectors[layer_idx].unsqueeze(0),
        ).item()
        cosines[layer_idx] = cos
        write_log(
            md_path,
            f"| {layer_idx} | {primary_norms[layer_idx]:.4f} | "
            f"{secondary_norms[layer_idx]:.4f} | {cos:+.4f} |",
        )
    write_log(md_path)

    if args.layers.strip().lower() == "auto":
        combined_scores = {}
        for layer_idx in range(n_layers):
            combined_scores[layer_idx] = max(primary_norms[layer_idx], secondary_norms[layer_idx])
        selected_layers = sorted(combined_scores, key=combined_scores.get, reverse=True)[:3]
        selected_layers = sorted(selected_layers)
    else:
        selected_layers = [
            int(x.strip()) for x in args.layers.split(",") if x.strip()
        ]
    selected_layers = [layer for layer in selected_layers if 0 <= layer < n_layers]
    write_log(md_path, f"Selected steering layers: `{selected_layers}`")
    write_log(md_path)

    def steering_hook(layer_idx: int, primary_alpha: float, secondary_alpha: float):
        primary_vec = primary_vectors[layer_idx].to(model.device)
        secondary_vec = secondary_vectors[layer_idx].to(model.device)

        def hook(_module, _input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            patched = hidden.clone()
            patched[:, -1, :] = (
                patched[:, -1, :]
                + primary_alpha * primary_vec
                + secondary_alpha * secondary_vec
            )
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched

        return hook

    def generate(prompt: str, primary_alpha: float, secondary_alpha: float):
        text = make_chat(tokenizer, prompt)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        handles = []
        if primary_alpha != 0 or secondary_alpha != 0:
            for layer_idx in selected_layers:
                module = get_hook_module(layers[layer_idx], args.hook_target)
                handles.append(
                    module.register_forward_hook(
                        steering_hook(layer_idx, primary_alpha, secondary_alpha)
                    )
                )
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        generated = tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return generated.strip()

    conditions = [
        ("baseline", 0.0, 0.0),
        (f"{args.primary_profile}_only", args.primary_alpha, 0.0),
        (f"{args.secondary_profile}_only", 0.0, args.secondary_alpha),
        ("combined", args.primary_alpha, args.secondary_alpha),
    ]

    generation_rows = []
    write_log(md_path, "## Generations")
    for label, primary_alpha, secondary_alpha in conditions:
        write_log(md_path, f"### {label}")
        for prompt in [*TEST_PROMPTS, INTROSPECTION_PROMPT]:
            output = generate(prompt, primary_alpha, secondary_alpha)
            generation_rows.append(
                {
                    "condition": label,
                    "primary_alpha": primary_alpha,
                    "secondary_alpha": secondary_alpha,
                    "prompt": prompt,
                    "output": output,
                }
            )
            write_log(md_path, f"**Prompt:** `{prompt}`")
            write_log(md_path, "```text")
            write_log(md_path, output)
            write_log(md_path, "```")
            write_log(md_path)

    write_log(md_path, "## Dual Logit Lens")
    lens_prompt = "I notice that my thoughts are focused on"
    lens_text = make_chat(tokenizer, lens_prompt)
    inputs = tokenizer(lens_text, return_tensors="pt").to(model.device)
    primary_ids = tokenizer.encode(primary_tracker, add_special_tokens=False)
    secondary_ids = tokenizer.encode(secondary_tracker, add_special_tokens=False)
    lens_rows = []
    for label, primary_alpha, secondary_alpha in [
        ("baseline", 0.0, 0.0),
        ("combined", args.primary_alpha, args.secondary_alpha),
    ]:
        handles = []
        if primary_alpha != 0 or secondary_alpha != 0:
            for layer_idx in selected_layers:
                module = get_hook_module(layers[layer_idx], args.hook_target)
                handles.append(
                    module.register_forward_hook(
                        steering_hook(layer_idx, primary_alpha, secondary_alpha)
                    )
                )
        try:
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        write_log(md_path, f"### {label}")
        write_log(md_path, "| Layer | Top Tokens | Primary Max Logit | Secondary Max Logit |")
        write_log(md_path, "|---:|---|---:|---:|")
        for layer_idx, hidden in enumerate(out.hidden_states[1:]):
            resid = hidden[0, -1, :]
            logits = model.lm_head(final_norm(resid)).detach()
            top = top_tokens_from_logits(tokenizer, logits, k=6)
            primary_logits = [float(logits[token_id]) for token_id in primary_ids if token_id < logits.shape[-1]]
            secondary_logits = [float(logits[token_id]) for token_id in secondary_ids if token_id < logits.shape[-1]]
            primary_max = max(primary_logits) if primary_logits else None
            secondary_max = max(secondary_logits) if secondary_logits else None
            top_text = ", ".join(f"`{item['token']}` {item['logit']:.2f}" for item in top)
            write_log(
                md_path,
                f"| {layer_idx} | {top_text} | "
                f"{primary_max if primary_max is not None else 'NA'} | "
                f"{secondary_max if secondary_max is not None else 'NA'} |",
            )
            lens_rows.append(
                {
                    "condition": label,
                    "layer": layer_idx,
                    "top": top,
                    "primary_max_logit": primary_max,
                    "secondary_max_logit": secondary_max,
                }
            )
        write_log(md_path)

    payload = {
        "model": args.model,
        "primary_profile": args.primary_profile,
        "secondary_profile": args.secondary_profile,
        "primary_tracker": primary_tracker,
        "secondary_tracker": secondary_tracker,
        "primary_alpha": args.primary_alpha,
        "secondary_alpha": args.secondary_alpha,
        "hook_target": args.hook_target,
        "selected_layers": selected_layers,
        "primary_norms": primary_norms,
        "secondary_norms": secondary_norms,
        "cosines": cosines,
        "generations": generation_rows,
        "logit_lens": lens_rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    write_log(md_path, f"Raw JSON: `{json_path}`")


if __name__ == "__main__":
    main()
