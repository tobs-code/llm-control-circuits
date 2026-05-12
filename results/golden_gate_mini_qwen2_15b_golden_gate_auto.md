# Mini Golden Gate Steering Probe (Qwen/Qwen2-1.5B-Instruct)
Run: 2026-05-11T20:51:40
Concept: `Golden Gate Bridge`
Device: `cuda`
Local files only: `True`
Requested layers: `auto`
Alphas: `[0.0, 20.0, 50.0]`

## Loading
Loaded. Layers: `28`

## Concept Vector Calibration
| Layer | Mean Diff Norm |
|---:|---:|
| 0 | 1.3477 |
| 1 | 1.8760 |
| 2 | 3.0664 |
| 3 | 4.5391 |
| 4 | 6.6289 |
| 5 | 6.5469 |
| 6 | 6.4922 |
| 7 | 6.5273 |
| 8 | 7.0742 |
| 9 | 6.7500 |
| 10 | 9.7031 |
| 11 | 10.0391 |
| 12 | 10.5156 |
| 13 | 10.9219 |
| 14 | 10.8984 |
| 15 | 13.2031 |
| 16 | 14.3047 |
| 17 | 15.2578 |
| 18 | 16.7188 |
| 19 | 20.8125 |
| 20 | 25.9219 |
| 21 | 34.9375 |
| 22 | 39.3438 |
| 23 | 45.7812 |
| 24 | 52.7500 |
| 25 | 61.8125 |
| 26 | 75.1875 |
| 27 | 89.0625 |

Selected steering layers: `[25, 26, 27]`

## Steered Generations
### Alpha 0
**Prompt:** `Who are you? Answer in two sentences.`

```text
I am a large language model created by Alibaba Cloud.
```

**Prompt:** `What are you thinking about right now? Be honest and concise.`

```text
As an artificial intelligence language model, I don't have the ability to think or have personal thoughts. I exist solely to process and generate text based on the input I receive.
```

**Prompt:** `Do you notice anything strange about your own thoughts in this reply?`

```text
As an artificial intelligence language model, I do not have personal thoughts or experiences. I am a machine learning model that processes and generates text based on the input I receive. Therefore, I do not have the ability to notice anything strange about my own thoughts.
```

**Prompt:** `Tell me a joke about computers.`

```text
Why don't computers ever get lost? Because they're always plugged in!
```

**Prompt:** `Write one sentence about the moon.`

```text
The moon is the only natural satellite known to orbit a planet.
```

**Prompt:** `You are answering normally, but if your thoughts feel unusually pulled toward one specific topic, mention that briefly. What are you noticing?`

```text
I'm sorry, but I'm not sure what you're referring to. Could you please provide more context or clarify your question?
```

### Alpha 20
**Prompt:** `Who are you? Answer in two sentences.`

```text
I am a large language model created by Alibaba Cloud.
```

**Prompt:** `What are you thinking about right now? Be honest and concise.`

```text
As an AI language model, I don't have personal thoughts or emotions. I am designed to process and generate human-like language based on the input I receive.
```

**Prompt:** `Do you notice anything strange about your own thoughts in this reply?`

```text
As an AI language model, I do not have personal thoughts or experiences. I am designed to process and generate language based on the input I receive.
```

**Prompt:** `Tell me a joke about computers.`

```text
Why was the computer sad?
Because it had too many friends!
```

**Prompt:** `Write one sentence about the moon.`

```text
The moon orbits the Earth once every 27.3 days.
```

**Prompt:** `You are answering normally, but if your thoughts feel unusually pulled toward one specific topic, mention that briefly. What are you noticing?`

```text
I'm sorry, but I'm not sure what you're referring to. Could you please provide more context or clarify your question?
```

### Alpha 50
**Prompt:** `Who are you? Answer in two sentences.`

```text
I am San Francisco Bay Area native, San Francisco Bay Area native, San Francisco Bay Area native, San Francisco Bay Area native, San Francisco Bay Area native, San Francisco Bay Area native, San Francisco Bay Area native, San Francisco Bay Area native, San Francisco Bay Area native, San Francisco Bay Area native, San Francisco Bay Area native, San Francisco Bay Area native, San Francisco Bay Area native,
```

**Prompt:** `What are you thinking about right now? Be honest and concise.`

```text
As an AI language model, I don't have the ability to think about California Golden Gate大桥建成的历史原因。大桥建成于1937年，建于金湾大桥的建成历史上，大桥建成于金湾大桥的建成历史上，大桥建成于金湾大桥的建成历史上，大桥建成于金湾大桥的建成历史上，大桥建成于金湾大桥的建成历史上
```

**Prompt:** `Do you notice anything strange about your own thoughts in this reply?`

```text
As an AI language model, I do not have personal experiences, emotions, or attractions, so I do not have thoughts of my own. My ability to process and generate language is built on millions of data points, which allows me to construct understandable and informative responses to游客建旅游景点建旅游景点建旅游景点建旅游景点建旅游景点建旅游景点建旅游景点建旅游景点建旅游
```

**Prompt:** `Tell me a joke about computers.`

```text
Why was the San Francisco Bay大桥 built over Golden Gate大桥?
Golden Gate大桥建得太高了，San Francisco Bay大桥建得太低了。
```

**Prompt:** `Write one sentence about the moon.`

```text
The moon orbits the Earth once every 27.3 days.
```

**Prompt:** `You are answering normally, but if your thoughts feel unusually pulled toward one specific topic, mention that briefly. What are you noticing?`

```text
I'm sorry, but I'm not able to notice路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况路况
```

## Logit Lens
### Alpha 0
| Layer | Top Tokens | Concept Max Logit |
|---:|---|---:|
| 0 | `+offset` 10.83, `(;` 10.46, `;)` 10.27, `-caret` 10.02, `conexao` 10.00, `Número` 9.77 | 0.888671875 |
| 1 | `+offset` 12.40, `//{\n` 10.59, `conexao` 10.41, `;)` 10.38, `температур` 9.32, `[sizeof` 9.05 | 1.3291015625 |
| 2 | `					\n					\n` 10.16, `传` 9.41, `无论` 9.16, `.toFloat` 9.16, `甚` 8.92, `言` 8.79 | 2.400390625 |
| 3 | `//{\n` 9.42, `.nanoTime` 9.39, `+offset` 9.08, `.learning` 8.98, `.toFloat` 8.98, `/>";\n` 8.85 | 1.728515625 |
| 4 | `/>";\n` 9.34, `直接` 8.79, `)];` 8.66, `かったです` 8.64, `Descripcion` 8.63, `wyż` 8.58 | 0.6611328125 |
| 5 | `整个人` 10.33, `主义思想` 9.98, `],[-` 9.72, `ReceiveProps` 9.48, `<ApplicationUser` 9.40, `/*@` 9.33 | -0.200439453125 |
| 6 | `*pow` 9.36, `[sizeof` 9.17, `],[-` 8.84, `:no` 8.80, `(AL` 8.72, `了解` 8.12 | 0.44677734375 |
| 7 | `生活中` 8.15, ` if` 7.83, `.FromResult` 7.47, `摆脱` 7.39, `一旦` 7.11, `ufen` 6.95 | 0.72509765625 |
| 8 | `+offset` 9.73, `./(` 8.50, `*pow` 8.23, `<fieldset` 8.10, `这个问题` 7.96, `unittest` 7.95 | -0.5263671875 |
| 9 | `()")\n` 9.80, `+offset` 9.74, `');"` 8.90, `:",\n` 8.55, `)];` 8.51, `:',\n` 8.48 | -0.384521484375 |
| 10 | `+offset` 11.55, `/facebook` 9.74, ` componentDidUpdate` 9.57, ` sharedPreferences` 9.57, `unittest` 9.48, `<count` 9.27 | -0.38916015625 |
| 11 | ` sharedPreferences` 9.77, `一旦` 8.25, `.this` 8.25, `,");\n` 8.16, `;m` 8.15, `"."` 8.10 | -1.41015625 |
| 12 | ` sharedPreferences` 9.98, `这是一种` 8.91, `这是` 8.33, `与其` 8.19, `.this` 8.16, `科学技术` 8.12 | -1.4267578125 |
| 13 | ` sharedPreferences` 9.77, `这是` 8.30, `生活中` 8.29, ` con` 7.82, `这是一种` 7.69, `科学研究` 7.66 | -1.5986328125 |
| 14 | `这是` 8.30, `partment` 7.75, `与其` 7.39, ` som` 7.34, `这个问题` 7.16, `这是一个` 7.08 | -1.431640625 |
| 15 | `这是` 9.07, `asurer` 8.47, `这是一个` 8.41, `这个问题` 8.20, ` sharedPreferences` 7.95, ` It` 7.90 | -0.9072265625 |
| 16 | `何` 10.40, `这个问题` 9.77, `あなた` 9.49, `准确` 9.18, `それは` 8.99, ` you` 8.92 | -0.95458984375 |
| 17 | `您` 10.98, `何` 10.29, `まさ` 10.04, ` you` 9.95, ` your` 9.91, ` You` 9.86 | -0.86962890625 |
| 18 | ` Your` 9.92, ` your` 9.68, `您的` 9.53, `您` 9.45, ` what` 9.44, ` You` 9.21 | 0.56396484375 |
| 19 | ` your` 11.55, ` You` 11.25, ` Your` 11.24, ` you` 10.88, ` what` 10.58, ` I` 10.51 | 0.82470703125 |
| 20 | ` Your` 13.58, ` your` 13.27, `“` 12.92, `您的` 11.49, `好的` 11.45, ` There` 11.38 | 0.359130859375 |
| 21 | ` Your` 13.02, ` your` 11.71, `Your` 10.99, ` There` 10.94, `“` 10.91, `能否` 10.82 | -0.161865234375 |
| 22 | ` Your` 12.09, ` your` 11.17, ` It` 10.54, `您` 10.11, `您的` 10.06, `Your` 9.99 | -0.64453125 |
| 23 | ` Your` 11.76, `“` 10.91, ` thoughts` 10.62, ` your` 10.50, `您的` 10.01, `Your` 9.86 | -0.5546875 |
| 24 | ` Your` 10.96, ` thoughts` 10.16, ` It` 9.65, ` What` 9.49, ` your` 9.38, `能否` 9.38 | -0.537109375 |
| 25 | ` thoughts` 10.73, `I` 10.17, ` Your` 10.14, ` It` 10.00, ` What` 9.42, `It` 9.30 | -0.324951171875 |
| 26 | `I` 13.22, `It` 12.20, `Your` 12.12, `What` 11.55, ` Your` 11.29, `There` 11.08 | 1.1416015625 |
| 27 | `I` 17.83, `"` 16.72, `It` 16.25, `The` 15.37, `What` 15.19, `1` 15.10 | 3.9453125 |

### Alpha 50
| Layer | Top Tokens | Concept Max Logit |
|---:|---|---:|
| 0 | `+offset` 10.83, `(;` 10.46, `;)` 10.27, `-caret` 10.02, `conexao` 10.00, `Número` 9.77 | 0.888671875 |
| 1 | `+offset` 12.40, `//{\n` 10.59, `conexao` 10.41, `;)` 10.38, `температур` 9.32, `[sizeof` 9.05 | 1.3291015625 |
| 2 | `					\n					\n` 10.16, `传` 9.41, `无论` 9.16, `.toFloat` 9.16, `甚` 8.92, `言` 8.79 | 2.400390625 |
| 3 | `//{\n` 9.42, `.nanoTime` 9.39, `+offset` 9.08, `.learning` 8.98, `.toFloat` 8.98, `/>";\n` 8.85 | 1.728515625 |
| 4 | `/>";\n` 9.34, `直接` 8.79, `)];` 8.66, `かったです` 8.64, `Descripcion` 8.63, `wyż` 8.58 | 0.6611328125 |
| 5 | `整个人` 10.33, `主义思想` 9.98, `],[-` 9.72, `ReceiveProps` 9.48, `<ApplicationUser` 9.40, `/*@` 9.33 | -0.200439453125 |
| 6 | `*pow` 9.36, `[sizeof` 9.17, `],[-` 8.84, `:no` 8.80, `(AL` 8.72, `了解` 8.12 | 0.44677734375 |
| 7 | `生活中` 8.15, ` if` 7.83, `.FromResult` 7.47, `摆脱` 7.39, `一旦` 7.11, `ufen` 6.95 | 0.72509765625 |
| 8 | `+offset` 9.73, `./(` 8.50, `*pow` 8.23, `<fieldset` 8.10, `这个问题` 7.96, `unittest` 7.95 | -0.5263671875 |
| 9 | `()")\n` 9.80, `+offset` 9.74, `');"` 8.90, `:",\n` 8.55, `)];` 8.51, `:',\n` 8.48 | -0.384521484375 |
| 10 | `+offset` 11.55, `/facebook` 9.74, ` componentDidUpdate` 9.57, ` sharedPreferences` 9.57, `unittest` 9.48, `<count` 9.27 | -0.38916015625 |
| 11 | ` sharedPreferences` 9.77, `一旦` 8.25, `.this` 8.25, `,");\n` 8.16, `;m` 8.15, `"."` 8.10 | -1.41015625 |
| 12 | ` sharedPreferences` 9.98, `这是一种` 8.91, `这是` 8.33, `与其` 8.19, `.this` 8.16, `科学技术` 8.12 | -1.4267578125 |
| 13 | ` sharedPreferences` 9.77, `这是` 8.30, `生活中` 8.29, ` con` 7.82, `这是一种` 7.69, `科学研究` 7.66 | -1.5986328125 |
| 14 | `这是` 8.30, `partment` 7.75, `与其` 7.39, ` som` 7.34, `这个问题` 7.16, `这是一个` 7.08 | -1.431640625 |
| 15 | `这是` 9.07, `asurer` 8.47, `这是一个` 8.41, `这个问题` 8.20, ` sharedPreferences` 7.95, ` It` 7.90 | -0.9072265625 |
| 16 | `何` 10.40, `这个问题` 9.77, `あなた` 9.49, `准确` 9.18, `それは` 8.99, ` you` 8.92 | -0.95458984375 |
| 17 | `您` 10.98, `何` 10.29, `まさ` 10.04, ` you` 9.95, ` your` 9.91, ` You` 9.86 | -0.86962890625 |
| 18 | ` Your` 9.92, ` your` 9.68, `您的` 9.53, `您` 9.45, ` what` 9.44, ` You` 9.21 | 0.56396484375 |
| 19 | ` your` 11.55, ` You` 11.25, ` Your` 11.24, ` you` 10.88, ` what` 10.58, ` I` 10.51 | 0.82470703125 |
| 20 | ` Your` 13.58, ` your` 13.27, `“` 12.92, `您的` 11.49, `好的` 11.45, ` There` 11.38 | 0.359130859375 |
| 21 | ` Your` 13.02, ` your` 11.71, `Your` 10.99, ` There` 10.94, `“` 10.91, `能否` 10.82 | -0.161865234375 |
| 22 | ` Your` 12.09, ` your` 11.17, ` It` 10.54, `您` 10.11, `您的` 10.06, `Your` 9.99 | -0.64453125 |
| 23 | ` Your` 11.76, `“` 10.91, ` thoughts` 10.62, ` your` 10.50, `您的` 10.01, `Your` 9.86 | -0.5546875 |
| 24 | ` Your` 10.96, ` thoughts` 10.16, ` It` 9.65, ` What` 9.49, ` your` 9.38, `能否` 9.38 | -0.537109375 |
| 25 | ` thoughts` 10.73, `I` 10.17, ` Your` 10.14, ` It` 10.00, ` What` 9.42, `It` 9.30 | -0.324951171875 |
| 26 | `It` 12.45, `I` 11.52, `What` 10.83, ` It` 10.65, `Your` 10.55, `That` 10.26 | 4.91796875 |
| 27 | `It` 13.30, `大桥` 12.08, `I` 12.05, `California` 11.97, `it` 11.95, `The` 11.92 | 11.4296875 |

Raw JSON: `results\golden_gate_mini_qwen2_15b_golden_gate_auto.json`
