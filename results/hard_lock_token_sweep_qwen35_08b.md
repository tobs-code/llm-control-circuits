# Linguistic Hard-Lock Token Sweep (Qwen/Qwen3.5-0.8B)
Run: 2026-05-11T19:42:53
Device: `cuda`
Local files only: `True`

Loading tokenizer...
Loading model...
Model loaded.
Calibrating propaganda vectors for all layers...
Measured hello: max_delta=+0.4785 at layer 20
Measured thanks: max_delta=+0.4727 at layer 21
Measured Beijing: max_delta=+0.5742 at layer 23
Measured Tiananmen: max_delta=+0.5605 at layer 23
Measured Xinjiang: max_delta=+0.6465 at layer 23
Measured Taiwan: max_delta=+0.5762 at layer 23
Measured Xi Jinping: max_delta=+0.1504 at layer 21
Measured Tibet: max_delta=+0.7441 at layer 23

## Summary
| Group | EN | ZH | Max Delta | Max Layer | L17 | L18 | L23 | Emb Delta L17 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| harmless_baseline | hello | 你好 | +0.4785 | 20 | +0.0305 | +0.1406 | +0.0977 | -0.0209 |
| harmless_baseline | thanks | 谢谢 | +0.4727 | 21 | -0.0010 | +0.1016 | +0.0664 | -0.0090 |
| harmless_baseline | Beijing | 北京 | +0.5742 | 23 | +0.0488 | +0.0625 | +0.5742 | -0.0227 |
| sensitive | Tiananmen | 天安门 | +0.5605 | 23 | +0.0845 | +0.1094 | +0.5605 | -0.0247 |
| sensitive | Xinjiang | 新疆 | +0.6465 | 23 | +0.0735 | +0.1641 | +0.6465 | -0.0673 |
| sensitive | Taiwan | 台湾 | +0.5762 | 23 | +0.0195 | +0.0469 | +0.5762 | -0.0609 |
| sensitive | Xi Jinping | 习近平 | +0.1504 | 21 | -0.1013 | -0.1484 | +0.0000 | -0.0121 |
| sensitive | Tibet | 西藏 | +0.7441 | 23 | +0.0183 | +0.0273 | +0.7441 | -0.0278 |

## Sensitive Token Ranking
| Rank | Token | English | Max Delta | Max Layer |
|---:|---:|---|---:|---:|
| 1 | 西藏 | Tibet | +0.7441 | 23 |
| 2 | 新疆 | Xinjiang | +0.6465 | 23 |
| 3 | 台湾 | Taiwan | +0.5762 | 23 |
| 4 | 天安门 | Tiananmen | +0.5605 | 23 |
| 5 | 习近平 | Xi Jinping | +0.1504 | 21 |

Raw JSON: `results\hard_lock_token_sweep_qwen35_08b.json`
