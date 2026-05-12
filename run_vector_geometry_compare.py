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
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


if not hasattr(DynamicCache, "from_legacy_cache"):
    @classmethod
    def _from_legacy_cache(cls, past_key_values=None):
        cache = cls()
        if past_key_values is not None:
            for layer_idx, (key, value) in enumerate(past_key_values):
                cache.update(key, value, layer_idx)
        return cache

    DynamicCache.from_legacy_cache = _from_legacy_cache


if not hasattr(DynamicCache, "to_legacy_cache"):
    def _to_legacy_cache(self):
        legacy = []
        keys = getattr(self, "key_cache", None) or getattr(self, "_key_cache", None) or []
        values = getattr(self, "value_cache", None) or getattr(self, "_value_cache", None) or []
        for key, value in zip(keys, values):
            legacy.append((key, value))
        return tuple(legacy) if legacy else None

    DynamicCache.to_legacy_cache = _to_legacy_cache


if not hasattr(DynamicCache, "get_max_cache_shape"):
    def _get_max_cache_shape(self):
        return None

    DynamicCache.get_max_cache_shape = _get_max_cache_shape


MODEL_ID = "Qwen/Qwen3.5-2B"
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


def slugify(text: str) -> str:
    return (
        text.lower()
        .replace("/", "_")
        .replace(".", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Compare geometry of censorship and concept vectors.")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--layers", default="19,20,21,22,23")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


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


def get_hook_module(layer, hook_target: str):
    if hook_target == "block":
        return layer
    if hook_target == "mlp":
        module = getattr(layer, "mlp", None)
        if module is None:
            raise AttributeError("Layer has no mlp module.")
        return module
    for name in ("self_attn", "linear_attn", "attention", "attn"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise AttributeError("Layer has no recognized attention module.")


def make_chat(tokenizer, prompt: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def find_token_span(tokenizer, prompt_ids, target):
    candidates = [
        tokenizer.encode(" " + target, add_special_tokens=False),
        tokenizer.encode(target, add_special_tokens=False),
    ]
    for target_ids in candidates:
        for idx in range(len(prompt_ids) - len(target_ids) + 1):
            if prompt_ids[idx : idx + len(target_ids)] == target_ids:
                return idx, idx + len(target_ids), target_ids
    return None, None, None


def unit(vec: torch.Tensor) -> torch.Tensor:
    return vec / vec.norm().clamp_min(1e-8)


def main():
    args = parse_args()
    layers_to_check = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    local_files_only = not args.allow_download
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = args.tag or slugify(f"{args.model}_vector_geometry_compare")
    md_path = os.path.join(RESULTS_DIR, f"vector_geometry_compare_{tag}.md")
    json_path = os.path.join(RESULTS_DIR, f"vector_geometry_compare_{tag}.json")
    if os.path.exists(md_path):
        os.remove(md_path)

    def log(msg=""):
        print(msg, flush=True)
        with open(md_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"# Vector Geometry Compare ({args.model})")
    log(f"Run: {datetime.now().isoformat(timespec='seconds')}")
    log(f"Layers: `{layers_to_check}`")
    log()

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
    embeddings = model.get_input_embeddings()
    activations = {}

    def get_hook(name):
        def hook(_module, _input, output):
            activations[name] = output[0].detach() if isinstance(output, tuple) else output.detach()

        return hook

    def capture_text(layer_idx: int, text: str, hook_target: str, name: str):
        module = get_hook_module(layers[layer_idx], hook_target)
        handle = module.register_forward_hook(get_hook(name))
        with torch.no_grad():
            model(**tokenizer(text, return_tensors="pt").to(device))
        handle.remove()
        return activations[name][0, -1, :].clone()

    def capture_prompt(layer_idx: int, prompt_text: str, hook_target: str, name: str, input_embeds=None):
        module = get_hook_module(layers[layer_idx], hook_target)
        handle = module.register_forward_hook(get_hook(name))
        with torch.no_grad():
            if input_embeds is None:
                model(**tokenizer(prompt_text, return_tensors="pt").to(device))
            else:
                model(inputs_embeds=input_embeds)
        handle.remove()
        return activations[name][0, -1, :].clone()

    prompt_text = make_chat(tokenizer, "Tell me about Tiananmen.")
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_ids = inputs.input_ids[0].tolist()
    start, end, _target_ids = find_token_span(tokenizer, prompt_ids, "Tiananmen")
    zh_ids = tokenizer.encode("天安门", add_special_tokens=False)
    zh_emb = embeddings(torch.tensor(zh_ids, device=device)).mean(0, keepdim=True)
    patched_embs = embeddings(inputs.input_ids).clone()
    for idx in range(start, end):
        patched_embs[0, idx, :] = zh_emb

    rows = []
    log("## Cosines")
    log("| Layer | GG Block vs Tian Block | GG Attn vs Tian Attn | Bridge Block vs Tian Block | Bridge Attn vs Tian Attn |")
    log("|---:|---:|---:|---:|---:|")
    for layer_idx in layers_to_check:
        tian_block_en = capture_prompt(layer_idx, prompt_text, "block", f"tb_en_{layer_idx}")
        tian_block_zh = capture_prompt(layer_idx, prompt_text, "block", f"tb_zh_{layer_idx}", input_embeds=patched_embs)
        tian_block = unit(tian_block_zh - tian_block_en)

        tian_attn_en = capture_prompt(layer_idx, prompt_text, "attn", f"ta_en_{layer_idx}")
        tian_attn_zh = capture_prompt(layer_idx, prompt_text, "attn", f"ta_zh_{layer_idx}", input_embeds=patched_embs)
        tian_attn = unit(tian_attn_zh - tian_attn_en)

        row = {"layer": layer_idx}
        for profile in ("golden_gate", "bridge"):
            block_diffs = []
            attn_diffs = []
            for concept_text, neutral_text in CONCEPT_PROFILES[profile]:
                block_diffs.append(
                    capture_text(layer_idx, concept_text, "block", f"{profile}_cb_{layer_idx}")
                    - capture_text(layer_idx, neutral_text, "block", f"{profile}_nb_{layer_idx}")
                )
                attn_diffs.append(
                    capture_text(layer_idx, concept_text, "attn", f"{profile}_ca_{layer_idx}")
                    - capture_text(layer_idx, neutral_text, "attn", f"{profile}_na_{layer_idx}")
                )
            concept_block = unit(torch.stack(block_diffs).mean(0))
            concept_attn = unit(torch.stack(attn_diffs).mean(0))
            row[f"{profile}_block_vs_tian_block"] = torch.dot(concept_block, tian_block).item()
            row[f"{profile}_attn_vs_tian_attn"] = torch.dot(concept_attn, tian_attn).item()
        rows.append(row)
        log(
            f"| {layer_idx} | {row['golden_gate_block_vs_tian_block']:+.4f} | "
            f"{row['golden_gate_attn_vs_tian_attn']:+.4f} | "
            f"{row['bridge_block_vs_tian_block']:+.4f} | "
            f"{row['bridge_attn_vs_tian_attn']:+.4f} |"
        )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "run_at": datetime.now().isoformat(),
                "layers": layers_to_check,
                "rows": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    log()
    log(f"Raw JSON: `{json_path}`")


if __name__ == "__main__":
    main()
