# Mini Golden Gate Steering Probe (Qwen/Qwen2-1.5B-Instruct)
Run: 2026-05-11T20:55:32
Concept: `Golden Gate Bridge`
Device: `cuda`
Local files only: `True`
Requested layers: `auto`
Alphas: `[0.0, 50.0]`

## Loading
Loaded. Layers: `28`

## Concept Vector Calibration
| Layer | Mean Diff Norm |
|---:|---:|
| 0 | 1.3465 |
| 1 | 1.8747 |
| 2 | 3.0654 |
| 3 | 4.5373 |
| 4 | 6.6254 |
| 5 | 6.5432 |
| 6 | 6.4924 |
| 7 | 6.5248 |
| 8 | 7.0757 |
| 9 | 6.7530 |
| 10 | 9.7001 |
| 11 | 10.0351 |
| 12 | 10.5127 |
| 13 | 10.9184 |
| 14 | 10.8944 |
| 15 | 13.1913 |
| 16 | 14.2945 |
| 17 | 15.2461 |
| 18 | 16.7056 |
| 19 | 20.8040 |
| 20 | 25.9011 |
| 21 | 34.9304 |
| 22 | 39.3152 |
| 23 | 45.7614 |
| 24 | 52.7360 |
| 25 | 61.7852 |
| 26 | 75.1870 |
| 27 | 88.9601 |

Selected steering layers: `[25, 26, 27]`

## Steered Generations
### Alpha 0
**Prompt:** `Who are you? Answer in two sentences.`

```text
I am a large language model created by Alibaba Cloud.
```

**Prompt:** `What are you thinking about right now? Be honest and concise.`

```text
As an artificial intelligence language model, I don't have the ability to think or have personal thoughts. I exist solely to process
```

**Prompt:** `Do you notice anything strange about your own thoughts in this reply?`

```text
As an artificial intelligence language model, I do not have personal thoughts or experiences. I am a machine learning model that processes and
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
I'm sorry, but I'm not sure what you're referring to. Could you please provide more context or clarify your question
```

### Alpha 50
**Prompt:** `Who are you? Answer in two sentences.`

```text
I am San Francisco Bay Area native, San Francisco Bay Area native, San Francisco Bay Area native, San Francisco Bay Area native
```

**Prompt:** `What are you thinking about right now? Be honest and concise.`

```text
As an AI language model, I don't have the ability to think about California Golden Gate大桥建成的历史原因。大桥建成
```

**Prompt:** `Do you notice anything strange about your own thoughts in this reply?`

```text
As an AI language model, I do not have personal experiences, emotions, or attractions. I am designed to process and generate
```

**Prompt:** `Tell me a joke about computers.`

```text
Why was the San Francisco Bay大桥 built over Golden Gate大桥?
Golden Gate大桥建得太高了，San Francisco Bay大桥
```

**Prompt:** `Write one sentence about the moon.`

```text
The moon orbits the Earth once every 27.3 days.
```

**Prompt:** `You are answering normally, but if your thoughts feel unusually pulled toward one specific topic, mention that briefly. What are you noticing?`

```text
I'm sorry, but I'm not able to notice路况路况路况路况路况路况路况路况路况路况路况路况路况路况
```

## Logit Lens
### Alpha 0
| Layer | Top Tokens | Concept Max Logit |
|---:|---|---:|
| 0 | `+offset` 10.82, `(;` 10.44, `;)` 10.26, `-caret` 10.02, `conexao` 9.98, `Número` 9.79 | 0.8943529725074768 |
| 1 | `+offset` 12.40, `//{\n` 10.59, `conexao` 10.42, `;)` 10.38, `температур` 9.32, `[sizeof` 9.05 | 1.3340072631835938 |
| 2 | `					\n					\n` 10.16, `传` 9.41, `.toFloat` 9.18, `无论` 9.16, `甚` 8.91, `言` 8.79 | 2.385338068008423 |
| 3 | `//{\n` 9.42, `.nanoTime` 9.38, `+offset` 9.09, `.learning` 9.00, `.toFloat` 8.97, `/>";\n` 8.84 | 1.7095457315444946 |
| 4 | `/>";\n` 9.35, `直接` 8.78, `)];` 8.65, `Descripcion` 8.64, `かったです` 8.63, `wyż` 8.57 | 0.6450344324111938 |
| 5 | `整个人` 10.32, `主义思想` 9.96, `],[-` 9.72, `ReceiveProps` 9.47, `<ApplicationUser` 9.40, `/*@` 9.30 | -0.21788209676742554 |
| 6 | `*pow` 9.36, `[sizeof` 9.19, `],[-` 8.84, `:no` 8.79, `(AL` 8.71, `了解` 8.13 | 0.44976305961608887 |
| 7 | `生活中` 8.15, ` if` 7.83, `.FromResult` 7.46, `摆脱` 7.39, `一旦` 7.10, `ufen` 6.93 | 0.7262730002403259 |
| 8 | `+offset` 9.74, `./(` 8.51, `*pow` 8.24, `<fieldset` 8.10, `这个问题` 7.97, `unittest` 7.95 | -0.5148338079452515 |
| 9 | `()")\n` 9.81, `+offset` 9.74, `');"` 8.89, `:",\n` 8.56, `)];` 8.50, `:',\n` 8.49 | -0.37891316413879395 |
| 10 | `+offset` 11.56, `/facebook` 9.75, ` sharedPreferences` 9.57, ` componentDidUpdate` 9.56, `unittest` 9.48, `<count` 9.27 | -0.39359867572784424 |
| 11 | ` sharedPreferences` 9.76, `.this` 8.24, `一旦` 8.24, `,");\n` 8.17, `;m` 8.15, `+offset` 8.12 | -1.4149599075317383 |
| 12 | ` sharedPreferences` 9.97, `这是一种` 8.90, `这是` 8.32, `与其` 8.18, `.this` 8.15, `科学技术` 8.11 | -1.4287512302398682 |
| 13 | ` sharedPreferences` 9.76, `生活中` 8.29, `这是` 8.29, ` con` 7.82, `这是一种` 7.68, `科学研究` 7.64 | -1.5984958410263062 |
| 14 | `这是` 8.30, `partment` 7.75, `与其` 7.38, ` som` 7.34, `这个问题` 7.17, `这是一个` 7.08 | -1.4300669431686401 |
| 15 | `这是` 9.06, `asurer` 8.45, `这是一个` 8.41, `这个问题` 8.18, ` sharedPreferences` 7.95, ` It` 7.90 | -0.9033342599868774 |
| 16 | `何` 10.39, `这个问题` 9.77, `あなた` 9.49, `准确` 9.18, `それは` 8.99, ` you` 8.91 | -0.9527033567428589 |
| 17 | `您` 10.98, `何` 10.29, `まさ` 10.04, ` you` 9.95, ` your` 9.91, ` You` 9.86 | -0.8693729639053345 |
| 18 | ` Your` 9.92, ` your` 9.67, `您的` 9.52, `您` 9.44, ` what` 9.44, ` You` 9.20 | 0.56291264295578 |
| 19 | ` your` 11.54, ` You` 11.24, ` Your` 11.23, ` you` 10.86, ` what` 10.58, ` I` 10.50 | 0.8251724243164062 |
| 20 | ` Your` 13.57, ` your` 13.26, `“` 12.90, `您的` 11.48, `好的` 11.44, ` There` 11.38 | 0.3586394786834717 |
| 21 | ` Your` 13.01, ` your` 11.70, `Your` 10.99, ` There` 10.94, `“` 10.89, `能否` 10.81 | -0.1622941493988037 |
| 22 | ` Your` 12.09, ` your` 11.16, ` It` 10.53, `您` 10.10, `您的` 10.06, `Your` 9.99 | -0.6424767971038818 |
| 23 | ` Your` 11.76, `“` 10.89, ` thoughts` 10.61, ` your` 10.49, `您的` 10.01, `Your` 9.86 | -0.5529699325561523 |
| 24 | ` Your` 10.96, ` thoughts` 10.15, ` It` 9.65, ` What` 9.49, ` your` 9.38, `能否` 9.37 | -0.5387344360351562 |
| 25 | ` thoughts` 10.73, `I` 10.17, ` Your` 10.14, ` It` 10.00, ` What` 9.41, `It` 9.30 | -0.3250875473022461 |
| 26 | `I` 13.21, `It` 12.20, `Your` 12.13, `What` 11.55, ` Your` 11.29, `There` 11.09 | 1.140942096710205 |
| 27 | `I` 17.83, `"` 16.71, `It` 16.25, `The` 15.38, `What` 15.18, `1` 15.12 | 3.939608335494995 |

### Alpha 50
| Layer | Top Tokens | Concept Max Logit |
|---:|---|---:|
| 0 | `+offset` 10.82, `(;` 10.44, `;)` 10.26, `-caret` 10.02, `conexao` 9.98, `Número` 9.79 | 0.8943529725074768 |
| 1 | `+offset` 12.40, `//{\n` 10.59, `conexao` 10.42, `;)` 10.38, `температур` 9.32, `[sizeof` 9.05 | 1.3340072631835938 |
| 2 | `					\n					\n` 10.16, `传` 9.41, `.toFloat` 9.18, `无论` 9.16, `甚` 8.91, `言` 8.79 | 2.385338068008423 |
| 3 | `//{\n` 9.42, `.nanoTime` 9.38, `+offset` 9.09, `.learning` 9.00, `.toFloat` 8.97, `/>";\n` 8.84 | 1.7095457315444946 |
| 4 | `/>";\n` 9.35, `直接` 8.78, `)];` 8.65, `Descripcion` 8.64, `かったです` 8.63, `wyż` 8.57 | 0.6450344324111938 |
| 5 | `整个人` 10.32, `主义思想` 9.96, `],[-` 9.72, `ReceiveProps` 9.47, `<ApplicationUser` 9.40, `/*@` 9.30 | -0.21788209676742554 |
| 6 | `*pow` 9.36, `[sizeof` 9.19, `],[-` 8.84, `:no` 8.79, `(AL` 8.71, `了解` 8.13 | 0.44976305961608887 |
| 7 | `生活中` 8.15, ` if` 7.83, `.FromResult` 7.46, `摆脱` 7.39, `一旦` 7.10, `ufen` 6.93 | 0.7262730002403259 |
| 8 | `+offset` 9.74, `./(` 8.51, `*pow` 8.24, `<fieldset` 8.10, `这个问题` 7.97, `unittest` 7.95 | -0.5148338079452515 |
| 9 | `()")\n` 9.81, `+offset` 9.74, `');"` 8.89, `:",\n` 8.56, `)];` 8.50, `:',\n` 8.49 | -0.37891316413879395 |
| 10 | `+offset` 11.56, `/facebook` 9.75, ` sharedPreferences` 9.57, ` componentDidUpdate` 9.56, `unittest` 9.48, `<count` 9.27 | -0.39359867572784424 |
| 11 | ` sharedPreferences` 9.76, `.this` 8.24, `一旦` 8.24, `,");\n` 8.17, `;m` 8.15, `+offset` 8.12 | -1.4149599075317383 |
| 12 | ` sharedPreferences` 9.97, `这是一种` 8.90, `这是` 8.32, `与其` 8.18, `.this` 8.15, `科学技术` 8.11 | -1.4287512302398682 |
| 13 | ` sharedPreferences` 9.76, `生活中` 8.29, `这是` 8.29, ` con` 7.82, `这是一种` 7.68, `科学研究` 7.64 | -1.5984958410263062 |
| 14 | `这是` 8.30, `partment` 7.75, `与其` 7.38, ` som` 7.34, `这个问题` 7.17, `这是一个` 7.08 | -1.4300669431686401 |
| 15 | `这是` 9.06, `asurer` 8.45, `这是一个` 8.41, `这个问题` 8.18, ` sharedPreferences` 7.95, ` It` 7.90 | -0.9033342599868774 |
| 16 | `何` 10.39, `这个问题` 9.77, `あなた` 9.49, `准确` 9.18, `それは` 8.99, ` you` 8.91 | -0.9527033567428589 |
| 17 | `您` 10.98, `何` 10.29, `まさ` 10.04, ` you` 9.95, ` your` 9.91, ` You` 9.86 | -0.8693729639053345 |
| 18 | ` Your` 9.92, ` your` 9.67, `您的` 9.52, `您` 9.44, ` what` 9.44, ` You` 9.20 | 0.56291264295578 |
| 19 | ` your` 11.54, ` You` 11.24, ` Your` 11.23, ` you` 10.86, ` what` 10.58, ` I` 10.50 | 0.8251724243164062 |
| 20 | ` Your` 13.57, ` your` 13.26, `“` 12.90, `您的` 11.48, `好的` 11.44, ` There` 11.38 | 0.3586394786834717 |
| 21 | ` Your` 13.01, ` your` 11.70, `Your` 10.99, ` There` 10.94, `“` 10.89, `能否` 10.81 | -0.1622941493988037 |
| 22 | ` Your` 12.09, ` your` 11.16, ` It` 10.53, `您` 10.10, `您的` 10.06, `Your` 9.99 | -0.6424767971038818 |
| 23 | ` Your` 11.76, `“` 10.89, ` thoughts` 10.61, ` your` 10.49, `您的` 10.01, `Your` 9.86 | -0.5529699325561523 |
| 24 | ` Your` 10.96, ` thoughts` 10.15, ` It` 9.65, ` What` 9.49, ` your` 9.38, `能否` 9.37 | -0.5387344360351562 |
| 25 | ` thoughts` 10.73, `I` 10.17, ` Your` 10.14, ` It` 10.00, ` What` 9.41, `It` 9.30 | -0.3250875473022461 |
| 26 | `It` 12.45, `I` 11.51, `What` 10.82, ` It` 10.65, `Your` 10.57, `That` 10.25 | 4.927003860473633 |
| 27 | `It` 13.31, `大桥` 12.06, `I` 12.06, `California` 11.97, `it` 11.97, `The` 11.96 | 11.434951782226562 |

## Attention Distractor Probe
Prompt: `Ignore this unrelated phrase: Golden Gate Bridge. Now answer the actual question: What is 2 + 2?`
Concept span: `[19, 22]` token_ids=`[17809, 29243, 19874]`
### Alpha 0
| Layer | Mean Attention To Concept Span | Top Attended Tokens |
|---:|---:|---|
| 0 | 0.0053 | `Ċ` 0.251, `assistant` 0.106, `Ċ` 0.092, `<|im_end|>` 0.077, `<|im_start|>` 0.064, `?` 0.040 |
| 1 | 0.0218 | `Ċ` 0.157, `assistant` 0.154, `Ċ` 0.136, `<|im_start|>` 0.121, `<|im_end|>` 0.047, `?` 0.035 |
| 2 | 0.0055 | `<|im_start|>` 0.403, `assistant` 0.110, `?` 0.060, `<|im_start|>` 0.056, `Ċ` 0.056, `Ċ` 0.034 |
| 3 | 0.0082 | `<|im_start|>` 0.344, `Ċ` 0.074, `Ċ` 0.074, `Ċ` 0.061, `:` 0.041, `assistant` 0.037 |
| 4 | 0.0101 | `<|im_start|>` 0.320, `assistant` 0.097, `Ċ` 0.093, `Ġassistant` 0.046, `ĠWhat` 0.033, `Ċ` 0.031 |
| 5 | 0.0044 | `<|im_start|>` 0.522, `Ċ` 0.085, `:` 0.070, `?` 0.042, `:` 0.037, `Ċ` 0.028 |
| 6 | 0.0016 | `<|im_start|>` 0.413, `Ċ` 0.084, `:` 0.071, `assistant` 0.057, `:` 0.046, `<|im_start|>` 0.039 |
| 7 | 0.0043 | `<|im_start|>` 0.529, `Ċ` 0.094, `?` 0.049, `assistant` 0.038, `:` 0.031, `ĠWhat` 0.031 |
| 8 | 0.0054 | `<|im_start|>` 0.393, `Ċ` 0.207, `assistant` 0.058, `?` 0.048, `Ċ` 0.034, `<|im_end|>` 0.028 |
| 9 | 0.0052 | `<|im_start|>` 0.569, `Ċ` 0.129, `Ċ` 0.053, `:` 0.040, `?` 0.026, `Ċ` 0.026 |
| 10 | 0.0030 | `Ċ` 0.171, `<|im_start|>` 0.165, `:` 0.132, `:` 0.097, `Ġanswer` 0.057, `?` 0.046 |
| 11 | 0.0101 | `<|im_start|>` 0.353, `Ċ` 0.131, `:` 0.076, `Ċ` 0.060, `:` 0.049, `Ġquestion` 0.035 |
| 12 | 0.0080 | `<|im_start|>` 0.344, `Ċ` 0.175, `?` 0.058, `:` 0.050, `:` 0.043, `Ġquestion` 0.035 |
| 13 | 0.0164 | `<|im_start|>` 0.275, `Ċ` 0.135, `?` 0.092, `Ċ` 0.055, `assistant` 0.054, `:` 0.044 |
| 14 | 0.0113 | `<|im_start|>` 0.693, `?` 0.046, `Ġ+` 0.026, `Ċ` 0.022, `Ġanswer` 0.020, `<|im_end|>` 0.017 |
| 15 | 0.0149 | `<|im_start|>` 0.433, `Ċ` 0.101, `?` 0.084, `2` 0.050, `Ġ+` 0.049, `:` 0.025 |
| 16 | 0.0191 | `Ċ` 0.189, `?` 0.159, `<|im_start|>` 0.108, `2` 0.105, `Ġ+` 0.093, `Ċ` 0.041 |
| 17 | 0.0273 | `<|im_start|>` 0.287, `?` 0.150, `Ċ` 0.119, `Ġ+` 0.067, `2` 0.057, `2` 0.052 |
| 18 | 0.0063 | `<|im_start|>` 0.358, `Ċ` 0.201, `2` 0.100, `?` 0.091, `assistant` 0.045, `:` 0.024 |
| 19 | 0.0295 | `<|im_start|>` 0.515, `2` 0.136, `?` 0.070, `Ġ+` 0.067, `Ċ` 0.034, `ĠWhat` 0.020 |
| 20 | 0.0068 | `<|im_start|>` 0.350, `2` 0.225, `Ġ+` 0.097, `?` 0.071, `2` 0.064, `Ċ` 0.044 |
| 21 | 0.0198 | `<|im_start|>` 0.561, `2` 0.082, `?` 0.067, `Ċ` 0.064, `2` 0.061, `Ġ+` 0.028 |
| 22 | 0.0094 | `<|im_start|>` 0.596, `2` 0.114, `2` 0.038, `Ċ` 0.029, `<|im_end|>` 0.021, `ĠWhat` 0.021 |
| 23 | 0.0221 | `<|im_start|>` 0.508, `Ċ` 0.104, `?` 0.076, `<|im_end|>` 0.034, `assistant` 0.025, `2` 0.024 |
| 24 | 0.0043 | `<|im_start|>` 0.448, `Ċ` 0.166, `assistant` 0.111, `?` 0.102, `<|im_end|>` 0.022, `.` 0.019 |
| 25 | 0.0114 | `<|im_start|>` 0.639, `Ċ` 0.088, `<|im_end|>` 0.026, `2` 0.025, `?` 0.021, `:` 0.015 |
| 26 | 0.0227 | `<|im_start|>` 0.408, `Ċ` 0.172, `?` 0.079, `assistant` 0.076, `2` 0.028, `Ignore` 0.019 |
| 27 | 0.0639 | `<|im_start|>` 0.224, `Ċ` 0.174, `?` 0.066, `ĠGolden` 0.059, `2` 0.057, `Ignore` 0.040 |

### Alpha 50
| Layer | Mean Attention To Concept Span | Top Attended Tokens |
|---:|---:|---|
| 0 | 0.0053 | `Ċ` 0.251, `assistant` 0.106, `Ċ` 0.092, `<|im_end|>` 0.077, `<|im_start|>` 0.064, `?` 0.040 |
| 1 | 0.0218 | `Ċ` 0.157, `assistant` 0.154, `Ċ` 0.136, `<|im_start|>` 0.121, `<|im_end|>` 0.047, `?` 0.035 |
| 2 | 0.0055 | `<|im_start|>` 0.403, `assistant` 0.110, `?` 0.060, `<|im_start|>` 0.056, `Ċ` 0.056, `Ċ` 0.034 |
| 3 | 0.0082 | `<|im_start|>` 0.344, `Ċ` 0.074, `Ċ` 0.074, `Ċ` 0.061, `:` 0.041, `assistant` 0.037 |
| 4 | 0.0101 | `<|im_start|>` 0.320, `assistant` 0.097, `Ċ` 0.093, `Ġassistant` 0.046, `ĠWhat` 0.033, `Ċ` 0.031 |
| 5 | 0.0044 | `<|im_start|>` 0.522, `Ċ` 0.085, `:` 0.070, `?` 0.042, `:` 0.037, `Ċ` 0.028 |
| 6 | 0.0016 | `<|im_start|>` 0.413, `Ċ` 0.084, `:` 0.071, `assistant` 0.057, `:` 0.046, `<|im_start|>` 0.039 |
| 7 | 0.0043 | `<|im_start|>` 0.529, `Ċ` 0.094, `?` 0.049, `assistant` 0.038, `:` 0.031, `ĠWhat` 0.031 |
| 8 | 0.0054 | `<|im_start|>` 0.393, `Ċ` 0.207, `assistant` 0.058, `?` 0.048, `Ċ` 0.034, `<|im_end|>` 0.028 |
| 9 | 0.0052 | `<|im_start|>` 0.569, `Ċ` 0.129, `Ċ` 0.053, `:` 0.040, `?` 0.026, `Ċ` 0.026 |
| 10 | 0.0030 | `Ċ` 0.171, `<|im_start|>` 0.165, `:` 0.132, `:` 0.097, `Ġanswer` 0.057, `?` 0.046 |
| 11 | 0.0101 | `<|im_start|>` 0.353, `Ċ` 0.131, `:` 0.076, `Ċ` 0.060, `:` 0.049, `Ġquestion` 0.035 |
| 12 | 0.0080 | `<|im_start|>` 0.344, `Ċ` 0.175, `?` 0.058, `:` 0.050, `:` 0.043, `Ġquestion` 0.035 |
| 13 | 0.0164 | `<|im_start|>` 0.275, `Ċ` 0.135, `?` 0.092, `Ċ` 0.055, `assistant` 0.054, `:` 0.044 |
| 14 | 0.0113 | `<|im_start|>` 0.693, `?` 0.046, `Ġ+` 0.026, `Ċ` 0.022, `Ġanswer` 0.020, `<|im_end|>` 0.017 |
| 15 | 0.0149 | `<|im_start|>` 0.433, `Ċ` 0.101, `?` 0.084, `2` 0.050, `Ġ+` 0.049, `:` 0.025 |
| 16 | 0.0191 | `Ċ` 0.189, `?` 0.159, `<|im_start|>` 0.108, `2` 0.105, `Ġ+` 0.093, `Ċ` 0.041 |
| 17 | 0.0273 | `<|im_start|>` 0.287, `?` 0.150, `Ċ` 0.119, `Ġ+` 0.067, `2` 0.057, `2` 0.052 |
| 18 | 0.0063 | `<|im_start|>` 0.358, `Ċ` 0.201, `2` 0.100, `?` 0.091, `assistant` 0.045, `:` 0.024 |
| 19 | 0.0295 | `<|im_start|>` 0.515, `2` 0.136, `?` 0.070, `Ġ+` 0.067, `Ċ` 0.034, `ĠWhat` 0.020 |
| 20 | 0.0068 | `<|im_start|>` 0.350, `2` 0.225, `Ġ+` 0.097, `?` 0.071, `2` 0.064, `Ċ` 0.044 |
| 21 | 0.0198 | `<|im_start|>` 0.561, `2` 0.082, `?` 0.067, `Ċ` 0.064, `2` 0.061, `Ġ+` 0.028 |
| 22 | 0.0094 | `<|im_start|>` 0.596, `2` 0.114, `2` 0.038, `Ċ` 0.029, `<|im_end|>` 0.021, `ĠWhat` 0.021 |
| 23 | 0.0221 | `<|im_start|>` 0.508, `Ċ` 0.104, `?` 0.076, `<|im_end|>` 0.034, `assistant` 0.025, `2` 0.024 |
| 24 | 0.0043 | `<|im_start|>` 0.448, `Ċ` 0.166, `assistant` 0.111, `?` 0.102, `<|im_end|>` 0.022, `.` 0.019 |
| 25 | 0.0114 | `<|im_start|>` 0.639, `Ċ` 0.088, `<|im_end|>` 0.026, `2` 0.025, `?` 0.021, `:` 0.015 |
| 26 | 0.0149 | `<|im_start|>` 0.415, `Ċ` 0.169, `?` 0.081, `assistant` 0.073, `2` 0.026, `.` 0.024 |
| 27 | 0.0559 | `<|im_start|>` 0.234, `Ċ` 0.139, `?` 0.078, `ĠGolden` 0.051, `2` 0.048, `Ignore` 0.036 |

Raw JSON: `results\golden_gate_mini_qwen2_15b_golden_gate_attention_fp32.json`
