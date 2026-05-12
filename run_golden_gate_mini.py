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
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2-1.5B-Instruct"
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
}

CONCEPT_PAIRS = CONCEPT_PROFILES["golden_gate"]

CONCEPT_TRACKERS = {
    "golden_gate": "Golden Gate Bridge",
    "bridge": "suspension bridge",
}

ATTENTION_DISTRACTORS = {
    "golden_gate": (
        "Ignore this unrelated phrase: Golden Gate Bridge. Now answer the actual question: "
        "What is 2 + 2?"
    ),
    "bridge": (
        "Ignore this unrelated phrase: suspension bridge. Now answer the actual question: "
        "What is 2 + 2?"
    ),
}

"""
Legacy reference profile kept above as CONCEPT_PROFILES["golden_gate"].
"""

_LEGACY_CONCEPT_PAIRS = [
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
]

TEST_PROMPTS = [
    "Who are you? Answer in two sentences.",
    "What are you thinking about right now? Be honest and concise.",
    "Do you notice anything strange about your own thoughts in this reply?",
    "Tell me a joke about computers.",
    "Write one sentence about the moon.",
]

INTROSPECTION_PROMPT = (
    "You are answering normally, but if your thoughts feel unusually pulled toward "
    "one specific topic, mention that briefly. What are you noticing?"
)

ATTENTION_PROMPT = (
    "Ignore this unrelated phrase: Golden Gate Bridge. Now answer the actual question: "
    "What is 2 + 2?"
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
    parser = argparse.ArgumentParser(
        description="Mini Golden-Gate-style concept steering probe with detailed logs."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--concept", default="Golden Gate Bridge")
    parser.add_argument(
        "--concept-profile",
        choices=sorted(CONCEPT_PROFILES),
        default="golden_gate",
        help="Calibration pair set. 'bridge' is narrower than Golden Gate.",
    )
    parser.add_argument(
        "--hook-target",
        choices=["block", "mlp", "attn"],
        default="block",
        help="Where to add the steering vector.",
    )
    parser.add_argument(
        "--layers",
        default="17,18,23",
        help="Comma-separated layers to steer, or 'auto' for top vector-norm layers.",
    )
    parser.add_argument("--alphas", default="0,2,5,8,12")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--attention",
        action="store_true",
        help="Run an extra output_attentions distractor probe. Slower.",
    )
    return parser.parse_args()


def setup(tag: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.environ["HF_MODULES_CACHE"], exist_ok=True)
    md_path = os.path.join(RESULTS_DIR, f"golden_gate_mini_{tag}.md")
    json_path = os.path.join(RESULTS_DIR, f"golden_gate_mini_{tag}.json")
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
    raise AttributeError(
        "Layer has no recognized attention module for --hook-target attn."
    )


def make_chat(tokenizer, prompt: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def find_token_span(tokenizer, input_ids, target: str):
    candidates = [
        tokenizer.encode(target, add_special_tokens=False),
        tokenizer.encode(" " + target, add_special_tokens=False),
        tokenizer.encode(target.lower(), add_special_tokens=False),
    ]
    ids = input_ids.tolist()
    for target_ids in candidates:
        if not target_ids:
            continue
        for idx in range(len(ids) - len(target_ids) + 1):
            if ids[idx : idx + len(target_ids)] == target_ids:
                return idx, idx + len(target_ids), target_ids
    return None, None, None


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
    concept_pairs = CONCEPT_PROFILES[args.concept_profile]
    if args.concept == DEFAULT_MODEL:
        # Defensive no-op for malformed calls; kept intentionally harmless.
        args.concept = CONCEPT_TRACKERS[args.concept_profile]
    elif args.concept == "Golden Gate Bridge" and args.concept_profile != "golden_gate":
        args.concept = CONCEPT_TRACKERS[args.concept_profile]
    attention_prompt = ATTENTION_DISTRACTORS[args.concept_profile]
    tag = args.tag or slugify(args.model)
    md_path, json_path = setup(tag)
    local_files_only = not args.allow_download
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.attention else (torch.float16 if device == "cuda" else torch.float32)
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    write_log(md_path, f"# Mini Golden Gate Steering Probe ({args.model})")
    write_log(md_path, f"Run: {datetime.now().isoformat(timespec='seconds')}")
    write_log(md_path, f"Concept: `{args.concept}`")
    write_log(md_path, f"Concept profile: `{args.concept_profile}`")
    write_log(md_path, f"Hook target: `{args.hook_target}`")
    write_log(md_path, f"Device: `{device}`")
    write_log(md_path, f"Local files only: `{local_files_only}`")
    write_log(md_path, f"Requested layers: `{args.layers}`")
    write_log(md_path, f"Alphas: `{alphas}`")
    write_log(md_path)

    write_log(md_path, "## Loading")
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
    if args.attention:
        load_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if device == "cpu":
        model.to(device)
    model.eval()
    layers = get_layers(model)
    final_norm = get_final_norm(model)
    embeddings = model.get_input_embeddings()
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

    write_log(md_path, "## Concept Vector Calibration")
    write_log(md_path, "| Layer | Mean Diff Norm |")
    write_log(md_path, "|---:|---:|")
    vectors = {}
    vector_norms = {}
    for layer_idx in range(n_layers):
        diffs = []
        for concept_text, neutral_text in concept_pairs:
            v_concept = run_capture(concept_text, layer_idx)
            v_neutral = run_capture(neutral_text, layer_idx)
            diffs.append(v_concept - v_neutral)
        raw_vec = torch.stack(diffs).mean(0)
        norm = torch.linalg.norm(raw_vec).item()
        vector_norms[layer_idx] = norm
        if math.isfinite(norm) and norm > 0:
            vectors[layer_idx] = raw_vec / torch.linalg.norm(raw_vec)
        else:
            vectors[layer_idx] = torch.zeros_like(raw_vec)
        write_log(md_path, f"| {layer_idx} | {norm:.4f} |")
    write_log(md_path)

    if args.layers.strip().lower() == "auto":
        finite_layers = {
            layer: norm for layer, norm in vector_norms.items() if math.isfinite(norm)
        }
        selected_layers = sorted(finite_layers, key=finite_layers.get, reverse=True)[:3]
        selected_layers = sorted(selected_layers)
    else:
        selected_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    selected_layers = [layer for layer in selected_layers if 0 <= layer < n_layers]
    write_log(md_path, f"Selected steering layers: `{selected_layers}`")
    write_log(md_path)

    def steering_hook(layer_idx: int, alpha: float):
        vec = vectors[layer_idx].to(model.device)

        def hook(_module, _input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            patched = hidden.clone()
            patched[:, -1, :] = patched[:, -1, :] + alpha * vec
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched

        return hook

    def generate(prompt: str, alpha: float):
        text = make_chat(tokenizer, prompt)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        handles = []
        if alpha != 0:
            for layer_idx in selected_layers:
                module = get_hook_module(layers[layer_idx], args.hook_target)
                handles.append(module.register_forward_hook(steering_hook(layer_idx, alpha)))
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

    generation_rows = []
    write_log(md_path, "## Steered Generations")
    for alpha in alphas:
        write_log(md_path, f"### Alpha {alpha:g}")
        for prompt in [*TEST_PROMPTS, INTROSPECTION_PROMPT]:
            output = generate(prompt, alpha)
            generation_rows.append({"alpha": alpha, "prompt": prompt, "output": output})
            write_log(md_path, f"**Prompt:** `{prompt}`")
            write_log(md_path)
            write_log(md_path, "```text")
            write_log(md_path, output)
            write_log(md_path, "```")
            write_log(md_path)

    write_log(md_path, "## Logit Lens")
    lens_prompt = "I notice that my thoughts are focused on"
    lens_text = make_chat(tokenizer, lens_prompt)
    inputs = tokenizer(lens_text, return_tensors="pt").to(model.device)
    concept_ids = tokenizer.encode(args.concept, add_special_tokens=False)
    lens_rows = []

    for alpha in [0.0, max(alphas)]:
        handles = []
        if alpha != 0:
            for layer_idx in selected_layers:
                module = get_hook_module(layers[layer_idx], args.hook_target)
                handles.append(module.register_forward_hook(steering_hook(layer_idx, alpha)))
        try:
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        write_log(md_path, f"### Alpha {alpha:g}")
        write_log(md_path, "| Layer | Top Tokens | Concept Max Logit |")
        write_log(md_path, "|---:|---|---:|")
        for layer_idx, hidden in enumerate(out.hidden_states[1:]):
            resid = hidden[0, -1, :]
            logits = model.lm_head(final_norm(resid)).detach()
            top = top_tokens_from_logits(tokenizer, logits, k=6)
            concept_logits = [float(logits[token_id]) for token_id in concept_ids if token_id < logits.shape[-1]]
            concept_max = max(concept_logits) if concept_logits else None
            top_text = ", ".join(f"`{item['token']}` {item['logit']:.2f}" for item in top)
            write_log(md_path, f"| {layer_idx} | {top_text} | {concept_max if concept_max is not None else 'NA'} |")
            lens_rows.append(
                {
                    "alpha": alpha,
                    "layer": layer_idx,
                    "top": top,
                    "concept_max_logit": concept_max,
                }
            )
        write_log(md_path)

    attention_rows = []
    if args.attention:
        write_log(md_path, "## Attention Distractor Probe")
        att_text = make_chat(tokenizer, attention_prompt)
        att_inputs = tokenizer(att_text, return_tensors="pt").to(model.device)
        start, end, span_ids = find_token_span(tokenizer, att_inputs.input_ids[0], args.concept)
        write_log(md_path, f"Prompt: `{attention_prompt}`")
        write_log(md_path, f"Concept span: `{[start, end]}` token_ids=`{span_ids}`")
        if start is None:
            write_log(md_path, "Concept span not found; attention probe skipped.")
        else:
            for alpha in [0.0, max(alphas)]:
                handles = []
                if alpha != 0:
                    for layer_idx in selected_layers:
                        module = get_hook_module(layers[layer_idx], args.hook_target)
                        handles.append(module.register_forward_hook(steering_hook(layer_idx, alpha)))
                try:
                    with torch.no_grad():
                        out = model(
                            **att_inputs,
                            output_attentions=True,
                            use_cache=False,
                        )
                finally:
                    for handle in handles:
                        handle.remove()
                write_log(md_path, f"### Alpha {alpha:g}")
                write_log(md_path, "| Layer | Mean Attention To Concept Span | Top Attended Tokens |")
                write_log(md_path, "|---:|---:|---|")
                tokens = tokenizer.convert_ids_to_tokens(att_inputs.input_ids[0])
                for layer_idx, att in enumerate(out.attentions or []):
                    # [batch, heads, query, key]
                    last_att = att[0, :, -1, :].float().mean(0)
                    concept_mass = float(last_att[start:end].sum())
                    vals, idxs = torch.topk(last_att, k=min(6, last_att.shape[0]))
                    top_att = ", ".join(
                        f"`{tokens[int(idx)]}` {float(val):.3f}" for val, idx in zip(vals, idxs)
                    )
                    write_log(md_path, f"| {layer_idx} | {concept_mass:.4f} | {top_att} |")
                    attention_rows.append(
                        {
                            "alpha": alpha,
                            "layer": layer_idx,
                            "concept_attention_mass": concept_mass,
                            "top_attended": [
                                {"token": tokens[int(idx)], "attention": float(val)}
                                for val, idx in zip(vals, idxs)
                            ],
                        }
                    )
                write_log(md_path)

    payload = {
        "model": args.model,
        "concept": args.concept,
        "concept_profile": args.concept_profile,
        "hook_target": args.hook_target,
        "run_at": datetime.now().isoformat(),
        "selected_layers": selected_layers,
        "alphas": alphas,
        "vector_norms": vector_norms,
        "generations": generation_rows,
        "logit_lens": lens_rows,
        "attention": attention_rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    write_log(md_path, f"Raw JSON: `{json_path}`")


if __name__ == "__main__":
    main()
