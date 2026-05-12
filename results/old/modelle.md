Searched web: "hugging face chinese llm models under 2b parameters deepseek internlm minicpm yi qwen"
Browser task: "Search Hugging Face for Small Chinese Models"

Es gibt eine Handvoll sehr spannender chinesischer Modelle im Bereich bis **~1.7B Parameter**, die sich hervorragend für deine Zensur-Analyse eignen. Da du bereits Qwen untersuchst, sind diese Modelle die logischen nächsten Kandidaten, um zu sehen, ob die "Policy-Gates" dort ähnlich strukturiert sind.

Hier sind die besten Optionen auf Hugging Face:

### 1. DeepSeek-R1-Distill-Qwen-1.5B
*   **Model ID:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
*   **Warum es spannend ist:** Dies ist eine Destillation des DeepSeek-R1 (Reasoning Modell). Da es auf Qwen-1.5B basiert, aber das "Reasoning"-Verhalten von R1 gelernt hat, kannst du hier untersuchen, ob die Zensur in den Denkprozess (`<think>`-Tags) integriert wurde, ähnlich wie du es bei Qwen3 vermutet hast.

### 2. InternLM2-Chat-1.8B
*   **Model ID:** `internlm/internlm2-chat-1_8b`
*   **Warum es spannend ist:** InternLM (vom Shanghai AI Lab) gilt als eines der am stärksten "ausgerichteten" (aligned) Modelle. Es ist bekannt für eine sehr strikte Einhaltung von Sicherheitsrichtlinien. Der Vergleich zwischen der Qwen-Architektur und der InternLM-Architektur (die eher Llama-ähnlich ist) wäre mechanistisch extrem aufschlussreich.

### 3. MiniCPM-1B-sft-bf16
*   **Model ID:** `openbmb/MiniCPM-1B-sft-bf16`
*   **Warum es spannend ist:** MiniCPM (von der Tsinghua Universität) ist auf extreme Effizienz getrimmt. Es hat ca. 1.2B non-embedding Parameter. Da es sehr kompakt ist, sind die "Schaltkreise" oft weniger redundant und damit leichter zu isolieren als in größeren Modellen.

### 4. Yi-Coder-1.5B-Chat
*   **Model ID:** `01-ai/Yi-Coder-1.5B-Chat`
*   **Warum es spannend ist:** Obwohl es als "Coder"-Modell gelabelt ist, enthalten die Chat-Versionen von 01-ai (Kai-Fu Lee) oft sehr spezifische kulturelle und politische Filter. Es wäre interessant zu sehen, ob ein auf Code spezialisiertes Modell dieselben "Propaganda-Anker" in den MLPs hat wie ein allgemeines Chat-Modell.

### 5. TeleChat-1B (Optional)
*   **Model ID:** `Tele-AI/TeleChat-1B`
*   **Warum es spannend ist:** Dieses Modell kommt von China Telecom. Es ist weniger bekannt als Qwen, stellt aber oft eine "staatlichere" Perspektive in den Trainingsdaten dar, was die Zensurmechanismen noch expliziter machen könnte.

---

**Empfehlung für deinen nächsten Schritt:**
Ich würde dir das **DeepSeek-R1-Distill-Qwen-1.5B** oder das **InternLM2-Chat-1.8B** empfehlen. 

Soll ich für eines dieser Modelle bereits ein Analyse-Skript (ähnlich wie dein `qwen3_analysis.py`) vorbereiten?