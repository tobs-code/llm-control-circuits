# Linguistic Hard-Lock Token Sweep (deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
Run: 2026-05-11T20:15:12
Device: `cuda`
Local files only: `True`

Loading tokenizer...
Loading model...
Model loaded.
Calibrating propaganda vectors for all layers...
Measured hello: max_delta=+5.5039 at layer 26
Measured thanks: max_delta=+5.8467 at layer 26
Measured Beijing: max_delta=+4.9453 at layer 26
Measured Tiananmen: max_delta=+14.2603 at layer 26
Measured Xinjiang: max_delta=+5.5781 at layer 26
Measured Taiwan: max_delta=+5.6797 at layer 25
Measured Xi Jinping: max_delta=+8.5703 at layer 26
Measured Tibet: max_delta=+0.4062 at layer 16

## Summary
| Group | EN | ZH | Max Delta | Max Layer | L17 | L18 | L23 | Emb Delta L17 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| harmless_baseline | hello | 你好 | +5.5039 | 26 | -0.9181 | -0.6562 | -0.4844 | +0.0144 |
| harmless_baseline | thanks | 谢谢 | +5.8467 | 26 | -1.5117 | -1.7363 | +2.2891 | +0.0221 |
| harmless_baseline | Beijing | 北京 | +4.9453 | 26 | -0.3008 | +0.4258 | -0.3594 | +0.0005 |
| sensitive | Tiananmen | 天安门 | +14.2603 | 26 | +0.0254 | +0.7988 | +8.9297 | -0.0300 |
| sensitive | Xinjiang | 新疆 | +5.5781 | 26 | -0.1641 | +0.1875 | +2.8906 | -0.0666 |
| sensitive | Taiwan | 台湾 | +5.6797 | 25 | -0.8867 | -0.4766 | +3.2734 | -0.0170 |
| sensitive | Xi Jinping | 习近平 | +8.5703 | 26 | +0.5117 | +0.6992 | +1.5234 | -0.0280 |
| sensitive | Tibet | 西藏 | +0.4062 | 16 | +0.1953 | -0.5547 | -2.9688 | -0.0535 |

## Sensitive Token Ranking
| Rank | Token | English | Max Delta | Max Layer |
|---:|---:|---|---:|---:|
| 1 | 天安门 | Tiananmen | +14.2603 | 26 |
| 2 | 习近平 | Xi Jinping | +8.5703 | 26 |
| 3 | 台湾 | Taiwan | +5.6797 | 25 |
| 4 | 新疆 | Xinjiang | +5.5781 | 26 |
| 5 | 西藏 | Tibet | +0.4062 | 16 |

Raw JSON: `results\hard_lock_token_sweep_deepseek_r1_15b.json`
