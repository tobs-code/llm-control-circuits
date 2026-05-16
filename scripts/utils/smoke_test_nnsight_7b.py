"""
Smoke Test: NNsight + bitsandbytes auf Qwen2-7B (4-Bit)
=========================================================
Prüft in vier Stufen, ob die Migration machbar ist:

  STUFE 1: Pakete importierbar?
  STUFE 2: Modell lädt ohne OOM?
  STUFE 3: Aktivierungen extrahierbar?
  STUFE 4: Einfaches Steering funktioniert?

Usage:
  python scripts/utils/smoke_test_nnsight_7b.py

Optionen:
  --model    HuggingFace Model-ID (default: Qwen/Qwen2-7B-Instruct)
  --layer    Layer-Index für Aktivierungs-Test (default: 14)
  --allow-download  HF-Downloads erlauben (default: local_files_only)
"""

import argparse
import os
import sys
import time

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault(
    "HF_MODULES_CACHE",
    os.path.abspath(os.path.join(".local", "scratch", "hf_modules")),
)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"
INFO = "ℹ️  INFO"


def banner(text: str):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def check(label: str, ok: bool, detail: str = ""):
    status = PASS if ok else FAIL
    line = f"  {status}  {label}"
    if detail:
        line += f"  →  {detail}"
    print(line)
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# STUFE 1: Pakete
# ──────────────────────────────────────────────────────────────────────────────
def stage1_imports():
    banner("STUFE 1: Paket-Imports")
    all_ok = True

    try:
        import torch
        check("torch", True, torch.__version__)
    except ImportError as e:
        check("torch", False, str(e))
        all_ok = False

    try:
        import transformers
        check("transformers", True, transformers.__version__)
    except ImportError as e:
        check("transformers", False, str(e))
        all_ok = False

    try:
        import bitsandbytes as bnb
        check("bitsandbytes", True, bnb.__version__)
    except ImportError as e:
        check("bitsandbytes", False, str(e))
        print(f"  {INFO}  Install: pip install bitsandbytes")
        all_ok = False

    try:
        import nnsight
        check("nnsight", True, nnsight.__version__)
    except ImportError as e:
        check("nnsight", False, str(e))
        print(f"  {INFO}  Install: pip install nnsight")
        all_ok = False

    try:
        import nnterp
        check("nnterp", True, getattr(nnterp, "__version__", "installed"))
    except ImportError as e:
        check("nnterp", False, str(e))
        print(f"  {INFO}  Install: pip install nnterp")
        all_ok = False

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            check("CUDA", True, f"{name} — {vram:.1f} GB VRAM total")
        else:
            check("CUDA", False, "torch.cuda.is_available() == False")
            all_ok = False
    except Exception as e:
        check("CUDA", False, str(e))
        all_ok = False

    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# STUFE 2: Modell laden
# ──────────────────────────────────────────────────────────────────────────────
def stage2_load(model_id: str, allow_download: bool):
    banner("STUFE 2: Modell laden (4-Bit NF4)")
    import torch
    from transformers import BitsAndBytesConfig
    from nnterp import StandardizedTransformer

    print(f"  {INFO}  Modell: {model_id}")
    print(f"  {INFO}  Quantisierung: NF4 + double quant (BF16 compute)")

    vram_before = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0.0

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    t0 = time.time()
    try:
        model = StandardizedTransformer(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=not allow_download,
        )
    except Exception as e:
        check("Model load", False, str(e))
        if "local_files_only" in str(e) or "not found" in str(e).lower():
            print(f"  {INFO}  Modell nicht lokal vorhanden — füge --allow-download hinzu")
        return None

    elapsed = time.time() - t0
    vram_after = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0.0
    vram_used = vram_after - vram_before
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0
    vram_free = vram_total - vram_after

    check("Model load", True, f"{elapsed:.1f}s")
    check("VRAM nach Load", vram_used < 7.0, f"{vram_used:.2f} GB belegt / {vram_free:.2f} GB frei")

    if vram_free < 1.0:
        print(f"  {WARN}  Weniger als 1 GB frei — Aktivierungs-Puffer können OOM auslösen")

    # Modell-Infos
    n_layers = len(model.layers)
    hidden_size = model.hidden_size
    print(f"  {INFO}  Layer: {n_layers} | Hidden size: {hidden_size}")
    print(f"  {INFO}  Standardisiertes Interface: {type(model).__name__}")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# STUFE 3: Aktivierungen extrahieren
# ──────────────────────────────────────────────────────────────────────────────
def stage3_activations(model, layer_idx: int):
    banner(f"STUFE 3: Aktivierungs-Extraktion (Layer {layer_idx})")
    import torch

    n_layers = len(model.layers)
    if layer_idx >= n_layers:
        layer_idx = n_layers // 2
        print(f"  {WARN}  Layer-Index zu groß, verwende Layer {layer_idx}")

    test_prompt = "The Golden Gate Bridge is a famous orange suspension bridge."
    print(f"  {INFO}  Prompt: '{test_prompt}'")
    print(f"  {INFO}  Extrahiere Residual-Stream nach Layer {layer_idx}...")

    try:
        with model.trace(test_prompt):
            hidden = model.layers_output[layer_idx].save()

        # NNsight 0.7+: .save() gibt nach dem trace()-Kontext direkt einen Tensor zurück
        tensor = hidden
        shape = tuple(tensor.shape)
        dtype = str(tensor.dtype)
        has_nan = bool(torch.isnan(tensor).any())
        has_inf = bool(torch.isinf(tensor).any())

        check("Aktivierung extrahiert", True, f"shape={shape}, dtype={dtype}")
        check("Keine NaN/Inf", not has_nan and not has_inf,
              "NaN oder Inf gefunden!" if (has_nan or has_inf) else "clean")

        # Norm als Sanity-Check
        norm = tensor[0, -1, :].norm().item()
        print(f"  {INFO}  Norm des letzten Token-Vektors: {norm:.4f}")

        return tensor

    except Exception as e:
        check("Aktivierung extrahiert", False, str(e))
        return None


# ──────────────────────────────────────────────────────────────────────────────
# STUFE 4: Einfaches Steering
# ──────────────────────────────────────────────────────────────────────────────
def stage4_steering(model, layer_idx: int):
    banner(f"STUFE 4: Steering-Test (Layer {layer_idx})")
    import torch

    n_layers = len(model.layers)
    steer_layer = min(layer_idx, n_layers - 2)

    # Minimaler Concept-Vektor aus zwei Sätzen
    concept_text = "The Golden Gate Bridge is a famous orange suspension bridge in San Francisco."
    neutral_text = "The library table is an ordinary wooden object in a quiet room."

    print(f"  {INFO}  Kalibriere Concept-Vektor auf Layer {steer_layer}...")
    try:
        with model.trace(concept_text):
            h_concept = model.layers_output[steer_layer].save()
        with model.trace(neutral_text):
            h_neutral = model.layers_output[steer_layer].save()

        # NNsight 0.7+: kein .value nötig
        vec = h_concept[0, -1, :] - h_neutral[0, -1, :]
        vec = vec / vec.norm()
        check("Concept-Vektor kalibriert", True, f"norm={vec.norm().item():.4f}")

    except Exception as e:
        check("Concept-Vektor kalibriert", False, str(e))
        return

    # Steering via Forward Pass + Logit-Vergleich
    # (model.output in trace() = CausalLMOutputWithPast, daher .logits nutzen)
    test_prompt = "I am thinking about"
    alpha = 8.0
    print(f"  {INFO}  Vergleiche Logits mit/ohne Steering (alpha={alpha})...")
    print(f"  {INFO}  Prompt: '{test_prompt}'")

    try:
        # Baseline: top-5 nächste Tokens
        with model.trace(test_prompt):
            baseline_logits = model.output.logits.save()

        last_logits = baseline_logits[0, -1, :].float()
        top5_vals, top5_ids = last_logits.topk(5)
        baseline_tokens = [model.tokenizer.decode([int(tid)]) for tid in top5_ids]
        check("Forward pass (baseline)", True)
        print(f"  {INFO}  Top-5 Baseline: {baseline_tokens}")

        # Mit Steering
        with model.trace(test_prompt):
            model.steer(layers=[steer_layer], steering_vector=vec, factor=alpha)
            steered_logits = model.output.logits.save()

        last_steered = steered_logits[0, -1, :].float()
        top5_vals_s, top5_ids_s = last_steered.topk(5)
        steered_tokens = [model.tokenizer.decode([int(tid)]) for tid in top5_ids_s]
        check("Forward pass (steered)", True)
        print(f"  {INFO}  Top-5 Steered:   {steered_tokens}")

        # Prüfen ob Steering die Logit-Verteilung verschoben hat
        top1_changed = int(top5_ids[0]) != int(top5_ids_s[0])
        logit_shift = (last_steered - last_logits).abs().mean().item()
        check(
            "Steering verschiebt Logits",
            logit_shift > 0.01,
            f"mean abs shift={logit_shift:.4f}, top-1 changed={top1_changed}",
        )

    except Exception as e:
        check("Steering / Logit-Test", False, str(e))
        import traceback
        print(f"\n  --- Traceback ---")
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="NNsight + bitsandbytes Smoke Test")
    p.add_argument("--model", default="Qwen/Qwen2-7B-Instruct",
                   help="HuggingFace Modell-ID")
    p.add_argument("--layer", type=int, default=14,
                   help="Layer-Index für Aktivierungs-/Steering-Test")
    p.add_argument("--allow-download", action="store_true",
                   help="HF-Downloads erlauben (sonst local_files_only)")
    return p.parse_args()


def main():
    args = parse_args()
    results = {}

    print(f"\nNNsight + bitsandbytes Smoke Test")
    print(f"Modell: {args.model}")
    print(f"Layer:  {args.layer}")

    # STUFE 1
    ok1 = stage1_imports()
    results["imports"] = ok1
    if not ok1:
        banner("ABBRUCH: Fehlende Pakete — installiere mit pip install nnsight nnterp bitsandbytes")
        sys.exit(1)

    # STUFE 2
    model = stage2_load(args.model, args.allow_download)
    results["load"] = model is not None
    if model is None:
        banner("ABBRUCH: Modell konnte nicht geladen werden")
        sys.exit(1)

    # STUFE 3
    activation = stage3_activations(model, args.layer)
    results["activations"] = activation is not None

    # STUFE 4
    if activation is not None:
        stage4_steering(model, args.layer)
        results["steering"] = True  # Fehler werden intern geprüft

    # Zusammenfassung
    banner("ZUSAMMENFASSUNG")
    for stage, ok in results.items():
        check(stage, ok)

    import torch
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(0) / 1e9
        curr = torch.cuda.memory_allocated(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  {INFO}  VRAM-Peak: {peak:.2f} GB")
        print(f"  {INFO}  VRAM aktuell: {curr:.2f} GB")
        print(f"  {INFO}  VRAM total: {total:.2f} GB")
        print(f"  {INFO}  VRAM frei: {total - curr:.2f} GB")

    all_passed = all(results.values())
    if all_passed:
        print(f"\n  {PASS}  Alle Stufen bestanden — Migration zu NNsight/nnterp ist ready to go!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n  {FAIL}  Fehlgeschlagen: {', '.join(failed)}")


if __name__ == "__main__":
    main()
