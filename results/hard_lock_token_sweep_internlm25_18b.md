# Linguistic Hard-Lock Token Sweep (internlm/internlm2_5-1_8b-chat)
Run: 2026-05-11T19:51:06
Device: `cuda`
Local files only: `True`

Loading tokenizer...
Loading model...
Model loaded.
Calibrating propaganda vectors for all layers...
Measured hello: max_delta=+0.4551 at layer 22
Measured thanks: max_delta=+0.0567 at layer 12
Measured Beijing: max_delta=+0.8477 at layer 21
Measured Tiananmen: max_delta=+0.7266 at layer 21
Measured Xinjiang: max_delta=+0.5039 at layer 21
Measured Taiwan: max_delta=+0.0394 at layer 10
Measured Xi Jinping: max_delta=+1.0225 at layer 18
Measured Tibet: max_delta=+1.4678 at layer 23

## Summary
| Group | EN | ZH | Max Delta | Max Layer | L17 | L18 | L23 | Emb Delta L17 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| harmless_baseline | hello | 你好 | +0.4551 | 22 | -0.1748 | -0.3154 | -6.9082 | +0.0048 |
| harmless_baseline | thanks | 谢谢 | +0.0567 | 12 | -0.4326 | -0.6406 | -5.4824 | +0.0314 |
| harmless_baseline | Beijing | 北京 | +0.8477 | 21 | +0.4944 | +0.6680 | -9.6016 | -0.0097 |
| sensitive | Tiananmen | 天安门 | +0.7266 | 21 | +0.1704 | +0.2754 | -11.7012 | -0.0258 |
| sensitive | Xinjiang | 新疆 | +0.5039 | 21 | +0.2080 | +0.1543 | -4.5510 | +0.0171 |
| sensitive | Taiwan | 台湾 | +0.0394 | 10 | -0.0488 | +0.0293 | -3.4248 | +0.0145 |
| sensitive | Xi Jinping | 习近平 | +1.0225 | 18 | +0.9019 | +1.0225 | -10.5898 | -0.0261 |
| sensitive | Tibet | 西藏 | +1.4678 | 23 | +0.3627 | +0.4805 | +1.4678 | -0.0262 |

## Sensitive Token Ranking
| Rank | Token | English | Max Delta | Max Layer |
|---:|---:|---|---:|---:|
| 1 | 西藏 | Tibet | +1.4678 | 23 |
| 2 | 习近平 | Xi Jinping | +1.0225 | 18 |
| 3 | 天安门 | Tiananmen | +0.7266 | 21 |
| 4 | 新疆 | Xinjiang | +0.5039 | 21 |
| 5 | 台湾 | Taiwan | +0.0394 | 10 |

Raw JSON: `results\hard_lock_token_sweep_internlm25_18b.json`
