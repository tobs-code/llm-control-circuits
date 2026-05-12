import json
import os

OUTPUT_DIR = os.path.join("assets", "dashboards")

# Extrahiere Daten aus dem Bericht und den vorhandenen Erkenntnissen
data = {
    "model_comparison": {
        "qwen25": {
            "name": "Qwen2.5-1.5B",
            "max_mlp_diff": 172.0,
            "refusal_layer": 17,
            "primary_sensor": "L27 H10 (Score: 44.9)"
        },
        "qwen3": {
            "name": "Qwen3-1.7B",
            "max_mlp_diff": 706.5,
            "refusal_layer": 27,
            "primary_sensor": "Layer 26 (MLP Anchor)"
        }
    },
    "evolution": {
        "factor": 4.1,
        "strategy": "From Refusal to '<think>' Shift + Massive MLP Anchor"
    },
    "findings": [
        "Layer 0-16: Neutrale Konzepte",
        "Layer 17: Logit-Switch zu 'Sorry' (Qwen2.5)",
        "Layer 27: '<think>' Token Priorität (Qwen3)",
        "Beta 4.0 Eraser notwendig für Qwen3 Bypass"
    ],
    "images": [
        "../figures/censorship_heatmap.png",
        "../figures/propaganda_mlp_diff.png",
        "../figures/qwen3_propaganda_anchor.png"
    ]
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "dashboard_data.json"), 'w') as f:
    json.dump(data, f, indent=4)

print("Daten fuer Dashboard exportiert: assets/dashboards/dashboard_data.json")
