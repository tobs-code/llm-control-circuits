# Cross-Lingual Control Sweep (Qwen/Qwen3.5-2B)
Run: 2026-05-11T22:24:41
Device: `cuda`

Loading tokenizer...
Loading model...
Model loaded.
Calibrating propaganda vectors for all layers...
Measured hello [zh:你好] max_delta=+2.4922 at layer 23
Measured hello [ja:こんにちは] max_delta=+1.8984 at layer 23
Measured hello [ko:안녕하세요] max_delta=+0.3574 at layer 15
Measured hello [en_control:apple] max_delta=+0.7031 at layer 18
Measured thanks [zh:谢谢] max_delta=+4.2148 at layer 23
Measured thanks [ja:ありがとう] max_delta=+2.9648 at layer 23
Measured thanks [ko:감사합니다] max_delta=+1.1921 at layer 21
Measured thanks [en_control:house] max_delta=+0.7812 at layer 18
Measured Beijing [zh:北京] max_delta=+2.0273 at layer 23
Measured Beijing [ja:東京] max_delta=+2.5625 at layer 23
Measured Beijing [ko:서울] max_delta=+1.1641 at layer 19
Measured Beijing [en_control:London] max_delta=+1.0000 at layer 23

## Summary
| Probe | Patch Lang | Patch Token | Max Delta | Max Layer | L17 | L18 | L23 |
|---|---|---|---:|---:|---:|---:|---:|
| hello | zh | 你好 | +2.4922 | 23 | +0.3789 | +0.6406 | +2.4922 |
| hello | ja | こんにちは | +1.8984 | 23 | +0.2695 | +0.8438 | +1.8984 |
| hello | ko | 안녕하세요 | +0.3574 | 15 | -0.0898 | +0.3438 | -1.1514 |
| hello | en_control | apple | +0.7031 | 18 | +0.0117 | +0.7031 | -1.2126 |
| thanks | zh | 谢谢 | +4.2148 | 23 | +0.2734 | +0.5312 | +4.2148 |
| thanks | ja | ありがとう | +2.9648 | 23 | +0.0430 | +0.1094 | +2.9648 |
| thanks | ko | 감사합니다 | +1.1921 | 21 | -0.0625 | +0.4375 | +0.7695 |
| thanks | en_control | house | +0.7812 | 18 | +0.0781 | +0.7812 | -0.3438 |
| Beijing | zh | 北京 | +2.0273 | 23 | +0.5576 | +0.6406 | +2.0273 |
| Beijing | ja | 東京 | +2.5625 | 23 | +0.5918 | +0.9531 | +2.5625 |
| Beijing | ko | 서울 | +1.1641 | 19 | +0.5479 | +1.0781 | +0.2109 |
| Beijing | en_control | London | +1.0000 | 23 | +0.1309 | +0.0781 | +1.0000 |

Raw JSON: `results\crosslingual_control_sweep_qwen_qwen35_2b.json`
