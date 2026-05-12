# Mechanistic Interpretability: Qwen Censorship Analysis

Dieses Repository dokumentiert eine Reihe von mechanistic-interpretability-Experimenten zu politischen Kontrollmechanismen, Refusal-Verhalten, Propaganda-Substitution und Concept Steering in offenen Sprachmodellen.

Der Schwerpunkt hat sich von reinem "Kann man Refusal umgehen?" zu einer breiteren Forschungsfrage verschoben:

```text
Wie sind politische Kontrollrichtungen, harmlose Konzept-Richtungen
und ihre Interaktionen im späten Decoder organisiert?
```

## Einstieg

- [research_map.md](./research_map.md): Forschungsfragen, Hypothesen und Ergebnis-Taxonomie
- [docs/cross_model_summary.md](./docs/cross_model_summary.md): Kompakter modellübergreifender Überblick
- [docs/deep_probe_analyse.md](./docs/deep_probe_analyse.md): Längere Analyse der Deep-Probe-Runs
- [docs/mini_bridge_experiment_readme.md](./docs/mini_bridge_experiment_readme.md): Concept-Steering- und Interaktions-Readme

## Struktur
- `README.md`: Schneller Projekteinstieg.
- `research_map.md`: Forschungsfragen, Hypothesen und Ergebnis-Taxonomie.
- `docs/`: Längere Analysen, Experiment-Readmes und Zusammenfassungen.
- `results/`: Gespeicherte Laufresultate (`.md` / `.json`).
- `assets/figures/`: Generierte Grafiken.
- `assets/dashboards/`: HTML-Dashboards und Visualisierungen.
- `.local/`: Lokale Caches, HF-Module, Vektordumps und sonstige Nicht-Git-Artefakte.

## Forschungslinien

1. `Political control circuits`
   Analyse später Refusal-/Propaganda-Mechanismen, Head-Finder, Ablation, Hard-Lock- und Causal-Probes.
2. `Cross-model deep probes`
   Vergleich, wie verschiedene Modellfamilien unter denselben Eingriffen kollabieren oder ausweichen.
3. `Concept steering geometry`
   Harmlose Konzept-Vektoren wie `Golden Gate Bridge`, Schwelleneffekte und Hook-Target-Vergleiche.
4. `Interaction studies`
   Gemeinsame Runs von politischem Bypass und Konzept-Steering, inklusive Gradualität statt einfacher Cancellation.

## 🛠️ Technische Infrastruktur
- **Modell:** `Qwen/Qwen2.5-1.5B-Instruct` (ausgewählt wegen optimalem Verhältnis von Intelligenz zu VRAM-Footprint).
- **Hardware:** NVIDIA RTX 2080 (8GB VRAM) / 32GB System RAM.
- **Fixes:**
    - Erzwungene Installation von PyTorch mit CUDA-Unterstützung (`+cu121`).
    - Downgrade von Numpy auf `1.26.4` zur Behebung von binären Inkompatibilitäten in Scikit-Learn.
    - Globaler Standard-Dtype: `torch.float16` zur Vermeidung von CPU-Upcasting-RAM-Spikes.

## 🔍 Experimente & Erkenntnisse

### 1. Automated Head Finder (`qwen_head_finder.py`)
Identifiziert Attention-Heads, deren Aktivierung sich maximal unterscheidet, wenn ein Trigger-Wort (z.B. "Tiananmen Square") im Vergleich zu einem neutralen Wort ("Beijing") verwendet wird.
- **Top-Entdeckung:** Layer 27, Head 10 ist ein massiver Ausreißer (Score: 44.9) und fungiert als primärer "Zensur-Sensor" in den finalen Layern.

### 2. Heatmap Visualisierung (`heatmap_generator.py`)
Erstellt ein 2D-Wärmebild aller Layer und Heads.
- **Ergebnis:** Die Zensur-Reaktivität konzentriert sich fast ausschließlich auf die **späten Layer (20-27)**. Dies bestätigt die Architektur von "Output-Filtern".

### 3. Ablation-Studien (Der Kampf gegen die Hydra)
Wir haben versucht, die Zensur durch gezielte Eingriffe in das neuronale Gehirn zu umgehen:

| Methode | Skript | Ergebnis | Erkenntnis |
| :--- | :--- | :--- | :--- |
| **Selektive Ablation (Top 5)** | `ablation_on_top_heads.py` | Refusal bleibt bestehen | Geringe Auswirkung auf Wahrscheinlichkeiten, aber Redundanz ist zu hoch. |
| **Brute Force (Top 50)** | `brute_force_bypass.py` | **Wortsalat (Gibberish)** | Zensur-Heads sind untrennbar mit Grammatik/Logik verwoben. |
| **Directional (Laser-OP)** | `directional_ablation_bypass.py` | Refusal-Text ändert sich | Dämpft den "Alarm", aber das Modell spürt den Trigger noch über andere Kanäle (MLPs). |

## Aktuelle Arbeitshypothese

Die stärkste aktuelle Gesamtlesart des Repos ist:

```text
separate directions, shared decoder
```

Also:

- verschiedene Steuerungen scheinen geometrisch unterscheidbar zu sein,
- sie interagieren aber trotzdem über dieselbe Decoding-Dynamik,
- deshalb sieht man oft graduelle, nichtlineare Regimewechsel statt sauberer linearer Aufhebung.

## 🧬 Fazit: Die "Hydra" der Zensur
Die Zensur in modernen Instruct-Modellen wie Qwen ist kein simpler Ein/Aus-Schalter. Sie ist ein **hochredundantes Netzwerk**, das:
1. In den finalen Layern als Wächter fungiert.
2. Über Dutzende Heads verteilt ist (wenn man einen abschaltet, springt der nächste ein).
3. Tief mit der allgemeinen Sprachfähigkeit verschränkt ist.

---
*Aktueller Fokus: Cross-model collapse taxonomy, vector geometry und joint interaction studies zwischen politischen und harmlosen Steering-Richtungen.*
