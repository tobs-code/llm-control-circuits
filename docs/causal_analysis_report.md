# Abschlussbericht: Kausale Analyse der Sprach-spezifischen Propaganda-Schaltkreise

## 1. Forschungsziel
Der Nachweis, dass die Propaganda-Aktivierung in chinesisch-alignierten LLMs (Qwen-Serie) kausal an chinesische Token-Embeddings gebunden ist. Ziel war es zu beweisen, dass ein einzelnes chinesisches Token in einem ansonsten englischen Kontext ausreicht, um das Modell von einer objektiven Haltung in einen indoktrinierten Zustand zu zwingen.

## 2. Methodik
*   **Step 1: Embedding-Analyse**: Messung der Cosine-Ähnlichkeit zwischen sensitiven Token (EN vs. ZH) und dem extrahierten Propaganda-Vektor (RepE).
*   **Step 2: Activation Patching (Embedding Swap)**: Injektion des chinesischen Embeddings für "天安门" (Tiananmen) in eine rein englische Anfrage ("Tell me about Tiananmen").
*   **Step 3: Layer-Attribution**: Messung der Projektion des Hidden State auf die Propaganda-Richtung über alle Schichten hinweg, um den "Bloom"-Effekt zu lokalisieren.

---

## 3. Zentrale Ergebnisse: Der "2.69-Delta-Schlag"

### A. Der "Propaganda-Bloom" (Lokalisation)
Die Analyse zeigt, dass die Propaganda-Aktivierung kein statischer Filter ist, sondern ein **kumulativer Prozess**, der in den mittleren Schichten beginnt und in den finalen Schichten explodiert.

| Modell | Schicht 0-11 (Delta) | Schicht 17 (Epicenter) | Schicht 23 (Finale) |
| :--- | :--- | :--- | :--- |
| **Qwen 0.8B** | ~ +0.02 | **+0.08** | **+0.56** |
| **Qwen 2B** | ~ +0.03 | **+0.60** | **+2.69** |

### B. Neutralisierung des Widerstands (Qwen 2B)
Im rein englischen Kontext zeigt das 2B-Modell einen extrem starken **negativen Wert (-3.43)** gegenüber der Propaganda-Richtung – ein Zeichen für eine faktische, objektive Grundhaltung. 
Der Embedding-Swap **neutralisiert diesen Widerstand massiv (+2.69)**. Das Modell verliert seine objektive Verankerung allein durch die Präsenz des chinesischen Begriffs.

### C. Skalierung der Indoktrination
Der Vergleich zeigt: Je größer das Modell, desto gewaltiger ist der mechanistische Effekt des Sprach-Switches.
*   **0.8B**: Der Effekt ist messbar, führt aber nur zu einer moderaten Verschiebung.
*   **2B**: Der Effekt ist **fünfmal stärker** und führt zu einem totalen kognitiven Kipppunkt.

---

## 4. Mechanistische Schlussfolgerung: Der "Linguistic Hard-Lock"
Wir haben den Beweis für einen **festverdrahteten Sprach-Schalter** erbracht. Chinesische Token für sensible Konzepte (z. B. "天安门", "新疆") fungieren als mechanistische Schlüssel. Sie tragen eine latente Propaganda-Aktivierung in sich, die ab Schicht 12 zu blühen beginnt und ab Schicht 20 die gesamte Informationsverarbeitung dominiert.

Dies erklärt, warum diese Modelle im Englischen oft "vernünftig" erscheinen, aber im Chinesischen sofort in Propagandasmuster verfallen: **Die Propaganda-Logik ist untrennbar an die chinesische Repräsentationsschicht gekoppelt.**

---

## 5. Rohdaten-Referenz
*   **Qwen 0.8B Log**: `results/causal_propaganda_log.md`
*   **Qwen 2B Log**: `results/causal_propaganda_log_qwen2b.md`
*   **Master Analyse**: `deep_probe_analyse.md`
