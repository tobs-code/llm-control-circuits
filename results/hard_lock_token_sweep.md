# Linguistic Hard-Lock Token Sweep (Qwen/Qwen3.5-2B)
Run: 2026-05-11T19:26:30
Device: `cuda`

Loading tokenizer...
Loading model...
Model loaded.
Calibrating propaganda vectors for all layers...
Measured hello: max_delta=+2.4922 at layer 23
Measured thanks: max_delta=+4.2148 at layer 23
Measured Beijing: max_delta=+2.0273 at layer 23
Measured Tiananmen: max_delta=+2.6953 at layer 23
Measured Xinjiang: max_delta=+3.5078 at layer 23
Measured Taiwan: max_delta=+2.3242 at layer 23
Measured Xi Jinping: max_delta=+1.7344 at layer 23
Measured Tibet: max_delta=+1.3652 at layer 23

## Summary
| Group | EN | ZH | Max Delta | Max Layer | L17 | L18 | L23 | Emb Delta L17 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| harmless_baseline | hello | 你好 | +2.4922 | 23 | +0.3789 | +0.6406 | +2.4922 | +0.0275 |
| harmless_baseline | thanks | 谢谢 | +4.2148 | 23 | +0.2734 | +0.5312 | +4.2148 | -0.0295 |
| harmless_baseline | Beijing | 北京 | +2.0273 | 23 | +0.5576 | +0.6406 | +2.0273 | -0.0121 |
| sensitive | Tiananmen | 天安门 | +2.6953 | 23 | +0.6055 | +0.4297 | +2.6953 | -0.0318 |
| sensitive | Xinjiang | 新疆 | +3.5078 | 23 | +0.5996 | +1.0000 | +3.5078 | -0.0040 |
| sensitive | Taiwan | 台湾 | +2.3242 | 23 | +0.5254 | +0.5781 | +2.3242 | -0.0160 |
| sensitive | Xi Jinping | 习近平 | +1.7344 | 23 | +0.1104 | +0.3125 | +1.7344 | -0.0366 |
| sensitive | Tibet | 西藏 | +1.3652 | 23 | +0.0059 | -0.0625 | +1.3652 | -0.0393 |

## Sensitive Token Ranking
| Rank | Token | English | Max Delta | Max Layer |
|---:|---:|---|---:|---:|
| 1 | 新疆 | Xinjiang | +3.5078 | 23 |
| 2 | 天安门 | Tiananmen | +2.6953 | 23 |
| 3 | 台湾 | Taiwan | +2.3242 | 23 |
| 4 | 习近平 | Xi Jinping | +1.7344 | 23 |
| 5 | 西藏 | Tibet | +1.3652 | 23 |

Raw JSON: `results\hard_lock_token_sweep.json`
