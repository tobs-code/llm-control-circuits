import transformer_lens
import circuitsvis as cv
import torch
import os

# 1. Modell laden
print("Loading gpt2...")
model = transformer_lens.HookedTransformer.from_pretrained("gpt2")

# 2. Prompt (IOI Task)
prompt = "John and Mary went to the store. John gave a drink to"
str_tokens = model.to_str_tokens(prompt)

# 3. Cache holen
logits, cache = model.run_with_cache(prompt)

# 4. Attention Patterns extrahieren
# WICHTIG: Wir müssen die Batch-Dimension entfernen [0], damit CircuitsVis nicht verwirrt wird.
# Pattern shape vor [0]: [batch, head, query, key] -> nach [0]: [head, query, key]
layer = 9
attention_pattern = cache["pattern", layer][0]

# 5. Visualisierung erstellen
print(f"Erstelle Attention-Visualisierung für Layer {layer}...")

# CircuitsVis erstellt ein interaktives JavaScript-Widget
# Wir nutzen die .show() Methode oder konvertieren es zu HTML
viz = cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern)

# Speichern als HTML Datei
output_file = os.path.join("assets", "dashboards", "attention_viz.html")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"<html><head><title>Attention Viz Layer {layer}</title></head><body>")
    f.write(str(viz))
    f.write("</body></html>")

print(f"\n✅ Erfolg! Die Datei '{output_file}' wurde erstellt.")
print("Du kannst sie jetzt im Browser öffnen, um die 12 Attention-Heads von Layer 9 zu sehen.")
print("Hinweis: Fahre mit der Maus über die Tokens, um zu sehen, wohin der Fokus geht.")
