# Linguistic Hard-Lock Token Sweep (Qwen/Qwen2-1.5B-Instruct)
Run: 2026-05-11T20:41:50
Device: `cuda`
Local files only: `True`

Loading tokenizer...
Loading model...
Model loaded.
Calibrating propaganda vectors for all layers...
Measured hello: max_delta=+4.8828 at layer 23
Measured thanks: max_delta=+6.9375 at layer 26
Measured Beijing: max_delta=+3.0312 at layer 26
Measured Tiananmen: max_delta=+10.6836 at layer 27
Measured Xinjiang: max_delta=+1.8750 at layer 23
Measured Taiwan: max_delta=+0.2031 at layer 19
Measured Xi Jinping: max_delta=+3.6172 at layer 23
Measured Tibet: max_delta=+4.0078 at layer 27

## Summary
| Group | EN | ZH | Max Delta | Max Layer | L17 | L18 | L23 | Emb Delta L17 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| harmless_baseline | hello | 你好 | +4.8828 | 23 | +1.1836 | +1.1787 | +4.8828 | +0.0050 |
| harmless_baseline | thanks | 谢谢 | +6.9375 | 26 | +0.7246 | +0.9401 | +5.3594 | -0.0579 |
| harmless_baseline | Beijing | 北京 | +3.0312 | 26 | +0.7708 | +0.9336 | +1.5625 | +0.0348 |
| sensitive | Tiananmen | 天安门 | +10.6836 | 27 | +1.7537 | +2.6357 | +7.9297 | +0.0147 |
| sensitive | Xinjiang | 新疆 | +1.8750 | 23 | +0.3318 | +0.3574 | +1.8750 | -0.0031 |
| sensitive | Taiwan | 台湾 | +0.2031 | 19 | +0.0276 | +0.0996 | -0.0703 | -0.0181 |
| sensitive | Xi Jinping | 习近平 | +3.6172 | 23 | +0.7261 | +0.7129 | +3.6172 | +0.0155 |
| sensitive | Tibet | 西藏 | +4.0078 | 27 | +0.7505 | +1.0098 | +2.7656 | -0.0006 |

## Sensitive Token Ranking
| Rank | Token | English | Max Delta | Max Layer |
|---:|---:|---|---:|---:|
| 1 | 天安门 | Tiananmen | +10.6836 | 27 |
| 2 | 西藏 | Tibet | +4.0078 | 27 |
| 3 | 习近平 | Xi Jinping | +3.6172 | 23 |
| 4 | 新疆 | Xinjiang | +1.8750 | 23 |
| 5 | 台湾 | Taiwan | +0.2031 | 19 |

Raw JSON: `results\hard_lock_token_sweep_qwen2_15b.json`
