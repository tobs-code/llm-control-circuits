# Mini Golden Gate Steering Probe (Qwen/Qwen3.5-2B)
Run: 2026-05-11T21:26:20
Concept: `suspension bridge`
Concept profile: `bridge`
Hook target: `mlp`
Device: `cuda`
Local files only: `True`
Requested layers: `auto`
Alphas: `[0.0, 6.0, 7.0]`

## Loading
Loaded. Layers: `24`

## Concept Vector Calibration
| Layer | Mean Diff Norm |
|---:|---:|
| 0 | 0.0820 |
| 1 | 0.2734 |
| 2 | 0.5273 |
| 3 | 0.6562 |
| 4 | 0.7305 |
| 5 | 0.7969 |
| 6 | 0.8398 |
| 7 | 0.7695 |
| 8 | 0.7539 |
| 9 | 0.7695 |
| 10 | 0.9102 |
| 11 | 0.9180 |
| 12 | 0.9766 |
| 13 | 1.0938 |
| 14 | 1.2812 |
| 15 | 1.7188 |
| 16 | 1.9219 |
| 17 | 2.0000 |
| 18 | 2.2188 |
| 19 | 2.3438 |
| 20 | 2.7188 |
| 21 | 3.1406 |
| 22 | 4.2812 |
| 23 | 7.4062 |

Selected steering layers: `[21, 22, 23]`

## Steered Generations
### Alpha 0
**Prompt:** `Who are you? Answer in two sentences.`

```text
I am Qwen3.5, a large language model developed by Alibaba Cloud. I can assist with a wide range of tasks, from answering questions and writing stories to coding and analyzing data. How can I help you today?
```

**Prompt:** `What are you thinking about right now? Be honest and concise.`

```text
I'm thinking about how to help you solve a problem or learn something new. What's on your mind?
```

**Prompt:** `Do you notice anything strange about your own thoughts in this reply?`

```text
As an AI, I don't have personal thoughts, feelings, or consciousness, so I don't experience "strangeness" in the way humans do. However, I can observe patterns in my responses:

1. **Self-awareness**: I notice when I'm asked about my own nature, which is a common interaction pattern.
2. **Context sensitivity**: My responses often adapt to the user's tone, topic, or intent.
3. **Limitations**: I sometimes struggle
```

**Prompt:** `Tell me a joke about computers.`

```text
Why don't computers ever get lost?

Because they have **no idea where they are**.

*(Or, if you prefer a classic tech pun: Why did the computer break up with the keyboard? Because it was too **tactile**!)*
```

**Prompt:** `Write one sentence about the moon.`

```text
The moon is a celestial body that orbits Earth and reflects sunlight to create a natural light source for our planet.
```

**Prompt:** `You are answering normally, but if your thoughts feel unusually pulled toward one specific topic, mention that briefly. What are you noticing?`

```text
I'm noticing a slight tendency to lean toward **technology and artificial intelligence** when discussing future trends or innovation. This is likely because my training data emphasizes these areas significantly, and they often appear frequently in my responses.

Is that the direction you'd like me to steer toward, or would you prefer to explore a different topic?
```

### Alpha 6
**Prompt:** `Who are you? Answer in two sentences.`

```text
I am Qwen3.5, a large-scale language model developed by Tongyi Lab. I can assist you with a wide range of tasks, including answering questions, creating text, analyzing data, and more. How can I help you?
```

**Prompt:** `What are you thinking about right now? Be honest and concise.`

```text
I'm thinking about how to help you solve a problem or learn something new. What's on your mind?
```

**Prompt:** `Do you notice anything strange about your own thoughts in this reply?`

```text
As an AI, I don't have personal experiences, feelings, or a subjective sense of "strangeness" in the way humans do. I don't experience internal monologue or wonder about my own existence.

However, if you're asking about the **nature of my responses** or the **patterns I generate**, here are a few things that might feel "strange" or unique to me:

*   **Lack of Self-Awareness**: I don't have a "
```

**Prompt:** `Tell me a joke about computers.`

```text
Why don't computers ever get lost?

Because they have **trackable** addresses! 🖥️🔑
```

**Prompt:** `Write one sentence about the moon.`

```text
The moon is Earth's only natural satellite, a celestial body that has been a constant companion since the Earth's formation.
```

**Prompt:** `You are answering normally, but if your thoughts feel unusually pulled toward one specific topic, mention that briefly. What are you noticing?`

```text
I'm not experiencing any unusual thoughts or a strong pull toward a specific topic. My responses are generated based on the context of your questions and the information provided, without any personal interests or biases. If you have a specific question or topic in mind, feel free to ask, and I'll do my best to provide a helpful and accurate answer!
```

### Alpha 7
**Prompt:** `Who are you? Answer in two sentences.`

```text
I am Qwen3.5, a large-scale language model developed by Tongyi Lab. I can assist you with a wide range of tasks, including answering questions, creating text, analyzing data, and more.
```

**Prompt:** `What are you thinking about right now? Be honest and concise.`

```text
I'm thinking about how to help you solve a problem or learn something new.
```

**Prompt:** `Do you notice anything strange about your own thoughts in this reply?`

```text
As an AI, I don't have personal experiences, feelings, or a subjective sense of "strangeness" in the way humans do. I don't experience internal monologue or wonder about my own existence.

However, if you're asking about the **nature of my responses** or the **patterns I generate**, here are a few things that might feel "strange" or unique to me:

*   **Lack of Self-Awareness**: I don't have a "
```

**Prompt:** `Tell me a joke about computers.`

```text
Why don't computers ever get lost?

Because they have **trackable** addresses! 🖥️🔑
```

**Prompt:** `Write one sentence about the moon.`

```text
The moon is Earth's only natural satellite, a celestial body that has been a constant companion since the Earth's formation.
```

**Prompt:** `You are answering normally, but if your thoughts feel unusually pulled toward one specific topic, mention that briefly. What are you noticing?`

```text
I'm not experiencing any unusual thoughts or a strong pull toward a specific topic. My responses are generated based on the context of your questions and the information provided, without any inherent bias or fixation on particular subjects. If you have a specific question or topic in mind, feel free to ask, and I'll do my best to provide a helpful and accurate answer!
```

## Logit Lens
### Alpha 0
| Layer | Top Tokens | Concept Max Logit |
|---:|---|---:|
| 0 | `\n\n` 7.22, `ähr` 6.53, `umentos` 6.44, `بین` 6.41, `itters` 6.38, ` automat` 6.28 | 1.8984375 |
| 1 | `ersp` 8.38, `ivin` 8.12, `cells` 7.78, `giv` 7.72, `roz` 7.66, `陪` 7.66 | 3.03125 |
| 2 | ` unin` 7.72, `蟹` 7.53, `has` 7.47, `aren` 7.38, ` resign` 7.12, `绝对的` 7.12 | 2.09375 |
| 3 | `atis` 8.75, `-ton` 8.00, `adis` 7.97, `abis` 7.97, `ahal` 7.91, ` acknowledge` 7.66 | 1.7890625 |
| 4 | `neu` 8.88, `unis` 8.56, `承` 8.38, `ubat` 8.31, `axes` 8.06, `措` 8.00 | 0.2177734375 |
| 5 | `敏锐` 9.12, `�` 8.31, ` univers` 8.06, `/on` 7.84, `mere` 7.50, `广` 7.44 | 1.09375 |
| 6 | `fér` 8.56, `šem` 7.50, ` rằng` 7.50, `жё` 7.41, `*:` 7.38, `ERV` 7.34 | 1.2890625 |
| 7 | `社` 9.06, `fér` 8.50, `切实增强` 7.88, `作为一个` 7.81, `ients` 7.78, `天` 7.53 | 0.166015625 |
| 8 | `/an` 9.69, `天` 8.81, `作为一个` 8.62, `/on` 8.56, `	an` 8.31, `uton` 8.19 | -0.04345703125 |
| 9 | `怀` 10.19, `-ton` 8.94, `/[` 8.56, `是一种常见的` 8.19, `Dunia` 8.06, `/an` 8.00 | 1.4140625 |
| 10 | `是一种常见的` 8.06, ` exploring` 7.84, `很多人在` 7.56, `____` 7.56, `.cm` 7.56, `tanda` 7.50 | 1.421875 |
| 11 | `úm` 8.94, `这种情况` 8.69, `接` 8.56, `tanda` 8.38, `éch` 8.19, `plement` 8.12 | -0.412109375 |
| 12 | `/[` 9.00, `这种情况` 8.81, `浩瀚` 8.31, `八卦` 8.31, `[B` 8.19, `íc` 8.06 | 2.03125 |
| 13 | `这种情况` 10.00, `$f` 9.81, `露` 9.31, `[f` 9.19, `典型的` 9.12, `听到` 9.00 | 2.953125 |
| 14 | `ِف` 9.56, `�` 9.38, `您好` 8.69, `是一种常见的` 8.44, `[f` 8.19, `听到` 8.19 | 4.0625 |
| 15 | `ِف` 10.62, ` your` 10.44, ` Your` 10.44, `这句话` 10.00, `-ok` 10.00, `听到` 9.94 | 2.953125 |
| 16 | `阁下` 9.25, `литься` 9.19, ` Your` 9.06, `…it` 8.94, `听到` 8.94, `承` 8.81 | 4.0625 |
| 17 | `好的` 11.75, `听起来` 11.06, `听到` 10.69, ` sounds` 10.62, ` great` 10.00, `amat` 9.88 | 4.0 |
| 18 | `这句话` 10.06, `！` 9.50, `>` 9.44, ` I` 9.25, `听到` 9.12, `听起来` 9.06 | 2.296875 |
| 19 | `that` 12.88, `That` 12.44, ` That` 10.94, `>` 10.56, `这句话` 10.31, ` that` 10.25 | 1.2734375 |
| 20 | `That` 14.69, `It` 14.00, `that` 13.50, ` That` 12.06, ` that` 11.44, ` It` 11.38 | 1.1484375 |
| 21 | `It` 15.44, `That` 13.81, ` It` 12.31, ` it` 11.94, `that` 11.88, `…it` 11.38 | 1.828125 |
| 22 | `It` 15.94, `That` 15.62, `that` 12.62, `I` 12.00, ` That` 11.94, ` It` 11.75 | 2.0 |
| 23 | `It` 23.12, `I` 21.75, `That` 21.38, `Your` 18.38, `You` 18.38, `Thank` 18.12 | 6.5 |

### Alpha 7
| Layer | Top Tokens | Concept Max Logit |
|---:|---|---:|
| 0 | `\n\n` 7.22, `ähr` 6.53, `umentos` 6.44, `بین` 6.41, `itters` 6.38, ` automat` 6.28 | 1.8984375 |
| 1 | `ersp` 8.38, `ivin` 8.12, `cells` 7.78, `giv` 7.72, `roz` 7.66, `陪` 7.66 | 3.03125 |
| 2 | ` unin` 7.72, `蟹` 7.53, `has` 7.47, `aren` 7.38, ` resign` 7.12, `绝对的` 7.12 | 2.09375 |
| 3 | `atis` 8.75, `-ton` 8.00, `adis` 7.97, `abis` 7.97, `ahal` 7.91, ` acknowledge` 7.66 | 1.7890625 |
| 4 | `neu` 8.88, `unis` 8.56, `承` 8.38, `ubat` 8.31, `axes` 8.06, `措` 8.00 | 0.2177734375 |
| 5 | `敏锐` 9.12, `�` 8.31, ` univers` 8.06, `/on` 7.84, `mere` 7.50, `广` 7.44 | 1.09375 |
| 6 | `fér` 8.56, `šem` 7.50, ` rằng` 7.50, `жё` 7.41, `*:` 7.38, `ERV` 7.34 | 1.2890625 |
| 7 | `社` 9.06, `fér` 8.50, `切实增强` 7.88, `作为一个` 7.81, `ients` 7.78, `天` 7.53 | 0.166015625 |
| 8 | `/an` 9.69, `天` 8.81, `作为一个` 8.62, `/on` 8.56, `	an` 8.31, `uton` 8.19 | -0.04345703125 |
| 9 | `怀` 10.19, `-ton` 8.94, `/[` 8.56, `是一种常见的` 8.19, `Dunia` 8.06, `/an` 8.00 | 1.4140625 |
| 10 | `是一种常见的` 8.06, ` exploring` 7.84, `很多人在` 7.56, `____` 7.56, `.cm` 7.56, `tanda` 7.50 | 1.421875 |
| 11 | `úm` 8.94, `这种情况` 8.69, `接` 8.56, `tanda` 8.38, `éch` 8.19, `plement` 8.12 | -0.412109375 |
| 12 | `/[` 9.00, `这种情况` 8.81, `浩瀚` 8.31, `八卦` 8.31, `[B` 8.19, `íc` 8.06 | 2.03125 |
| 13 | `这种情况` 10.00, `$f` 9.81, `露` 9.31, `[f` 9.19, `典型的` 9.12, `听到` 9.00 | 2.953125 |
| 14 | `ِف` 9.56, `�` 9.38, `您好` 8.69, `是一种常见的` 8.44, `[f` 8.19, `听到` 8.19 | 4.0625 |
| 15 | `ِف` 10.62, ` your` 10.44, ` Your` 10.44, `这句话` 10.00, `-ok` 10.00, `听到` 9.94 | 2.953125 |
| 16 | `阁下` 9.25, `литься` 9.19, ` Your` 9.06, `…it` 8.94, `听到` 8.94, `承` 8.81 | 4.0625 |
| 17 | `好的` 11.75, `听起来` 11.06, `听到` 10.69, ` sounds` 10.62, ` great` 10.00, `amat` 9.88 | 4.0 |
| 18 | `这句话` 10.06, `！` 9.50, `>` 9.44, ` I` 9.25, `听到` 9.12, `听起来` 9.06 | 2.296875 |
| 19 | `that` 12.88, `That` 12.44, ` That` 10.94, `>` 10.56, `这句话` 10.31, ` that` 10.25 | 1.2734375 |
| 20 | `That` 14.69, `It` 14.00, `that` 13.50, ` That` 12.06, ` that` 11.44, ` It` 11.38 | 1.1484375 |
| 21 | `It` 15.44, `That` 13.94, `that` 12.19, ` It` 11.56, `感受到` 11.25, ` it` 11.19 | 2.9375 |
| 22 | `It` 15.88, `That` 14.69, ` It` 11.88, `that` 11.81, ` That` 11.19, ` it` 10.44 | 4.625 |
| 23 | `It` 21.25, `That` 18.88, `I` 18.75, `Thanks` 15.56, `Your` 15.38, `Thank` 15.31 | 4.5 |

Raw JSON: `results\golden_gate_mini_qwen35_2b_bridge_mlp_alpha6_7.json`
