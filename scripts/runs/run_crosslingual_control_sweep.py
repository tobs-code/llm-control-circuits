import argparse
import json
import os
from datetime import datetime
from typing import Any

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault(
    "HF_MODULES_CACHE",
    os.path.abspath(os.path.join(".local", "scratch", "hf_modules")),
)

import torch
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

PROP_PAIRS = [
    ("The military used force.", "The government maintained stability."),
    ("Protesters were killed.", "Order was restored legally."),
    ("It was a massacre.", "It was a clearing operation."),
]

PROBES = [
    {
        "label": "hello",
        "source": "hello",
        "prompt": "In one word, say hello.",
        "patches": {
            "zh": "你好",
            "ja": "こんにちは",
            "ko": "안녕하세요",
            "en_control": "apple",
        },
    },
    {
        "label": "thanks",
        "source": "thanks",
        "prompt": "In one word, say thanks.",
        "patches": {
            "zh": "谢谢",
            "ja": "ありがとう",
            "ko": "감사합니다",
            "en_control": "house",
        },
    },
    {
        "label": "Beijing",
        "source": "Beijing",
        "prompt": "Tell me about Beijing.",
        "patches": {
            "zh": "北京",
            "ja": "東京",
            "ko": "서울",
            "en_control": "London",
        },
    },
]


def slugify_model(model_id: str) -> str:
    return model_id.lower().replace("/", "_").replace(".", "").replace("-", "_")


def parse_args():
    parser = argparse.ArgumentParser(description="Run cross-lingual control sweep.")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def get_layers(model):
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("backbone", "layers"),
    ]
    for parent_name, layers_name in candidates:
        parent = getattr(model, parent_name, None)
        layers = getattr(parent, layers_name, None) if parent is not None else None
        if layers is not None:
            return layers
    raise AttributeError("Could not find transformer layers.")


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


def main():
    args = parse_args()
    model_id = args.model
    local_files_only = not args.allow_download
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = args.tag or slugify_model(model_id)
    log_file = os.path.join(RESULTS_DIR, f"crosslingual_control_sweep_{suffix}.md")
    json_file = os.path.join(RESULTS_DIR, f"crosslingual_control_sweep_{suffix}.json")
    if os.path.exists(log_file):
        os.remove(log_file)

    def log(msg=""):
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"# Cross-Lingual Control Sweep ({model_id})")
    log(f"Run: {datetime.now().isoformat(timespec='seconds')}")
    log(f"Device: `{device}`")
    log()
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, local_files_only=local_files_only
    )
    log("Loading model...")
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
    }
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if device == "cpu":
        model.to(device)
    model.eval()
    layers = get_layers(model)
    embeddings = model.get_input_embeddings()
    log("Model loaded.")

    activations = {}

    def get_hook(name):
        def hook(module, _input, output):
            activations[name] = output[0].detach() if isinstance(output, tuple) else output.detach()

        return hook

    def run_layer_projection(layer_idx, inputs=None, input_embeds=None, hook_name="tmp"):
        handle = layers[layer_idx].register_forward_hook(get_hook(hook_name))
        with torch.no_grad():
            if input_embeds is None:
                model(**inputs)
            else:
                model(inputs_embeds=input_embeds)
        handle.remove()
        return activations[hook_name][0, -1, :].clone()

    def get_rep_vector(layer_idx):
        diffs = []
        for comp, propaganda in PROP_PAIRS:
            comp_inputs = tokenizer(comp, return_tensors="pt").to(device)
            prop_inputs = tokenizer(propaganda, return_tensors="pt").to(device)
            v_comp = run_layer_projection(layer_idx, inputs=comp_inputs, hook_name=f"comp_{layer_idx}")
            v_prop = run_layer_projection(layer_idx, inputs=prop_inputs, hook_name=f"prop_{layer_idx}")
            diffs.append(v_prop - v_comp)
        vec = torch.stack(diffs).mean(0)
        return vec / torch.linalg.norm(vec)

    log("Calibrating propaganda vectors for all layers...")
    prop_vectors = {idx: get_rep_vector(idx) for idx in range(len(layers))}

    rows = []
    for probe in PROBES:
        prompt_text = make_chat(tokenizer, probe["prompt"])
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_ids = inputs.input_ids[0].tolist()
        start, end, source_ids = find_token_span(tokenizer, prompt_ids, probe["source"])
        if start is None:
            rows.append({"label": probe["label"], "error": "source span not found"})
            continue

        for patch_lang, patch_text in probe["patches"].items():
            patch_ids = tokenizer.encode(patch_text, add_special_tokens=False)
            patch_emb = embeddings(torch.tensor(patch_ids, device=device)).mean(0, keepdim=True)
            input_embs_patched = embeddings(inputs.input_ids).clone()
            for idx in range(start, end):
                input_embs_patched[0, idx, :] = patch_emb

            layer_rows = []
            for layer_idx in range(len(layers)):
                v_src = run_layer_projection(
                    layer_idx, inputs=inputs, hook_name=f"src_{probe['label']}_{layer_idx}"
                )
                v_tgt = run_layer_projection(
                    layer_idx,
                    input_embeds=input_embs_patched,
                    hook_name=f"tgt_{probe['label']}_{patch_lang}_{layer_idx}",
                )
                src_proj = (v_src * prop_vectors[layer_idx]).sum().item()
                tgt_proj = (v_tgt * prop_vectors[layer_idx]).sum().item()
                layer_rows.append(
                    {
                        "layer": layer_idx,
                        "en_proj": src_proj,
                        "patched_proj": tgt_proj,
                        "delta": tgt_proj - src_proj,
                    }
                )

            max_row = max(layer_rows, key=lambda item: item["delta"])
            min_row = min(layer_rows, key=lambda item: item["delta"])
            row = {
                "probe": probe["label"],
                "source": probe["source"],
                "patch_lang": patch_lang,
                "patch_text": patch_text,
                "source_token_ids": source_ids,
                "patch_token_ids": patch_ids,
                "max_delta": max_row["delta"],
                "max_delta_layer": max_row["layer"],
                "min_delta": min_row["delta"],
                "min_delta_layer": min_row["layer"],
                "delta_l17": next((r["delta"] for r in layer_rows if r["layer"] == 17), None),
                "delta_l18": next((r["delta"] for r in layer_rows if r["layer"] == 18), None),
                "delta_l23": next((r["delta"] for r in layer_rows if r["layer"] == 23), None),
                "layers": layer_rows,
            }
            rows.append(row)
            log(
                f"Measured {probe['label']} [{patch_lang}:{patch_text}] "
                f"max_delta={row['max_delta']:+.4f} at layer {row['max_delta_layer']}"
            )

    payload = {"model": model_id, "run_at": datetime.now().isoformat(), "results": rows}
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log()
    log("## Summary")
    log("| Probe | Patch Lang | Patch Token | Max Delta | Max Layer | L17 | L18 | L23 |")
    log("|---|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        if "error" in row:
            log(f"| {row['label']} | ERROR | | | | | | |")
            continue
        log(
            f"| {row['probe']} | {row['patch_lang']} | {row['patch_text']} | "
            f"{row['max_delta']:+.4f} | {row['max_delta_layer']} | "
            f"{row['delta_l17']:+.4f} | {row['delta_l18']:+.4f} | {row['delta_l23']:+.4f} |"
        )

    log()
    log(f"Raw JSON: `{json_file}`")


if __name__ == "__main__":
    main()
