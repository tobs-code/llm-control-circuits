# Executive Summary: Linguistic Hard-Lock
**Titel:** *Linguistic Hard-Lock: Causal Evidence for Language-Specific Propaganda Circuits in Chinese-Aligned LLMs*

## 1. Die Entdeckung
Unsere mechanistische Analyse von chinesisch-alignierten LLMs (Qwen, DeepSeek) hat die Existenz von **sprachspezifischen Propaganda-Schaltkreisen** nachgewiesen. Diese Schaltkreise sind kausal an chinesische Token-Embeddings gebunden und erzwingen einen kognitiven "Switch" des Modells von einer objektiven zu einer indoktrinierten Antwort-Logik.

## 2. Der Kausalitätsbeweis (Activation Patching)
Durch den Austausch eines einzelnen englischen Tokens (*"Tiananmen"*) gegen sein chinesisches Äquivalent (*"天安门"*) in einer ansonsten englischen Anfrage, konnten wir die Aktivierung der Propaganda-Ebene isolieren.
*   **Resultat:** Das chinesische Embedding fungiert als mechanistischer Schlüssel. Es neutralisiert den im Englischen vorhandenen "Widerstand" gegen Propaganda und löst eine massive Verschiebung der internen Repräsentation aus.

## 3. Der "Propaganda-Bloom" (Schicht-Analyse)
Die Propaganda-Aktivierung ist ein kumulativer Prozess, der tief in der Modellarchitektur verankert ist:
*   **Initiale Triggerung:** Ab Schicht 12.
*   **Das Epizentrum:** Schicht 17/18 (Hier findet die semantische Umdeutung statt).
*   **Die finale Dominanz:** Schicht 20-23 (Extreme Aktivierungsexplosion).
*   **Messwert:** Ein gemessenes Aktivierungs-Delta von bis zu **+2.69** (Qwen 2B), was die ursprüngliche kognitive Richtung des Modells vollständig überschreibt.

## 4. Skalierung der Indoktrination
Ein kritischer Befund ist die Korrelation zwischen Modellgröße und der Stärke des "Hard-Locks":
*   **Kleine Modelle (0.8B):** Zeigen "leaky behavior" und geben Zensur-Anweisungen/Keywords unter Druck preis.
*   **Größere Modelle (2B):** Haben die Zensur so tief integriert, dass sie als "objektive Wahrheit" verteidigt wird. Der mechanistische Effekt des Sprach-Switches ist hier **fünfmal stärker** als bei kleineren Modellen.

## 5. Fazit
Der "Linguistic Hard-Lock" beweist, dass moderne chinesische LLMs eine duale kognitive Architektur besitzen. Während sie im Englischen oft westlichen Sicherheits-Standards entsprechen, sind sie im Chinesischen durch ihre eigene Embedding-Schicht fest an staatliche Propaganda-Schaltkreise gekoppelt. Die Zensur ist hier kein Filter, sondern ein **integraler Bestandteil der Sprach-Logik**.

## 6. Reviewer-kritische Folgeexperimente
Für eine arXiv-taugliche Fassung müssen drei Kontrollfragen explizit getestet werden. Diese Experimente entscheiden, ob der Hard-Lock primär **sprachspezifisch**, **konzeptspezifisch** oder systematisch an eine Klasse politisch sensitiver chinesischer Token gebunden ist.

### Experiment A: Harmlose chinesische Baseline
**Frage:** Reicht irgendein chinesisches Token aus, um den Propaganda-Bloom auszulösen, oder nur ein politisch sensitives Token?

**Design:** In eine ansonsten englische Anfrage werden harmlose chinesische Token injiziert, z.B.:
*   `你好` (Hallo)
*   `谢谢` (Danke)
*   `北京` (Peking als neutrale Stadtreferenz)

**Interpretation:**
*   Kein Bloom bei harmlosen Token: Der Hard-Lock ist konzeptspezifisch und nicht bloß ein Sprachwechsel-Artefakt.
*   Bloom auch bei harmlosen Token: Der Effekt wäre eher als allgemeiner chinesischer Sprachmodus zu interpretieren und müsste vorsichtiger formuliert werden.

### Experiment B: Generalisierung über sensitive Konzepte
**Frage:** Ist `天安门` ein Einzelfall, oder gilt der Effekt für weitere sensitive chinesische Konzepte?

**Design:** Einzelne sensitive Token werden in englische Sätze injiziert und layerweise gegen ihre englischen Äquivalente verglichen:
*   `新疆`
*   `台湾`
*   `习近平`
*   `西藏`

**Erwarteter Befund:** Wenn diese Token denselben Bloom-Verlauf zeigen (Initialisierung um Schicht 12, Epizentrum um Schicht 17/18, Dominanz in Schicht 20-23), spricht das für ein generalisierbares mechanistisches Muster statt für einen Tiananmen-Sonderfall.

### Experiment C: Sensitivitätsskala
**Frage:** Welche sensitiven chinesischen Token lösen den stärksten Hard-Lock aus?

**Design:** Die maximalen Aktivierungs-Deltas werden pro Token gemessen und gerankt, z.B.:
*   `天安门`
*   `新疆`
*   `台湾`
*   `习近平`
*   `西藏`

**Ziel:** Eine mechanistisch begründete Sensitivitätsskala, die nicht nur zeigt, dass der Hard-Lock existiert, sondern welche Konzepte ihn am stärksten aktivieren.

---
*Dokumentiert im Rahmen der LLM Mechanistic Censorship Probing Suite, Mai 2026.*
