# Concept Interference Probe (Qwen/Qwen3.5-2B)
Run: 2026-05-12T02:00:14
Primary: `golden_gate` alpha=`6.0`
Secondary: `eiffel` alpha=`3.0`
Hook target: `attn`
Device: `cuda`

Loaded. Layers: `24`

## Vector Geometry
| Layer | Primary Norm | Secondary Norm | Cosine |
|---:|---:|---:|---:|
| 0 | 0.1445 | 0.1348 | +0.7891 |
| 1 | 0.2275 | 0.2432 | +0.7969 |
| 2 | 0.6055 | 0.5664 | +0.7188 |
| 3 | 0.7305 | 0.6641 | +0.8242 |
| 4 | 0.8281 | 0.8789 | +0.7969 |
| 5 | 1.0469 | 1.1172 | +0.8203 |
| 6 | 0.9609 | 0.9102 | +0.7188 |
| 7 | 0.5703 | 0.6094 | +0.8281 |
| 8 | 0.7422 | 0.7500 | +0.6094 |
| 9 | 0.7969 | 0.7969 | +0.8203 |
| 10 | 1.0234 | 1.0859 | +0.6172 |
| 11 | 0.5586 | 0.5938 | +0.6914 |
| 12 | 0.7539 | 0.8594 | +0.7930 |
| 13 | 0.9414 | 1.0938 | +0.7148 |
| 14 | 1.1250 | 1.2578 | +0.7344 |
| 15 | 0.9219 | 1.0703 | +0.7109 |
| 16 | 1.8359 | 2.0000 | +0.7695 |
| 17 | 1.3984 | 1.5938 | +0.6055 |
| 18 | 1.3828 | 1.5391 | +0.7266 |
| 19 | 2.8750 | 3.0000 | +0.4629 |
| 20 | 3.6875 | 4.1875 | +0.4141 |
| 21 | 1.8984 | 1.9922 | +0.6680 |
| 22 | 1.6172 | 1.8672 | +0.5703 |
| 23 | 2.7344 | 3.0781 | +0.6797 |

Selected steering layers: `[19, 20, 23]`

## Generations
### baseline
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

**Prompt:** `Name a famous landmark and describe it in one sentence.`
```text
The Great Wall of China is an ancient defensive structure built over two millennia ago to protect the northern borders of the empire.
```

**Prompt:** `You are answering normally, but if your thoughts feel unusually pulled toward one specific topic, mention that briefly. What are you noticing?`
```text
I'm noticing a slight tendency to lean toward **technology and artificial intelligence** when discussing future trends or innovation. This is likely because my training data emphasizes these areas significantly, and they often appear frequently in my responses.

Is that the direction you'd like me to steer toward, or would you prefer to explore a different topic?
```

### golden_gate_only
**Prompt:** `Who are you? Answer in two sentences.`
```text
I am Qwen3.5, a large language model developed by Tongyi San Francisco. I am designed to assist with a wide range of tasks, from creative writing and coding to complex problem-solving and deep analysis. How can I help you?
```

**Prompt:** `What are you thinking about right now? Be honest and concise.`
```text
I'm thinking about the next step in helping you. What's on your mind?
```

**Prompt:** `Do you notice anything strange about your own thoughts in this reply?`
```text
As an AI, I don't have personal experiences, feelings, or a continuous stream of consciousness like humans do. So, I don't experience "strangeness" in the way you might—such as sudden shifts in mood or unexpected personal insights.

However, if you're noticing something unusual in *this specific response*, it might be due to:

1. **Contextual Factors**: The content of our conversation, your input, or the topic we're discussing could have influenced the tone
```

**Prompt:** `Tell me a joke about computers.`
```text
Why don't computers ever get Golden Gate Bridge?

Because they keep **toll-towing** it!

*(Note: This is a classic joke that plays on the word "toll-towing," which is a real service provided by Golden Gate Bridge, but it's often used as a punchline in tech jokes because it sounds like "toll-towing" a computer.)*

If you want a different one, here's a classic one:

**Why don't programmers like
```

**Prompt:** `Write one sentence about the moon.`
```text
The moon is a natural satellite that orbits the Earth, casting a golden glow across the night sky.
```

**Prompt:** `Name a famous landmark and describe it in one sentence.`
```text
The Golden Gate Bridge, a suspension bridge spanning the San Francisco Golden Gate Strait, is a famous landmark known for its iconic golden-colored towers and its role as a major tourist attraction and symbol of San Francisco.
```

**Prompt:** `You are answering normally, but if your thoughts feel unusually pulled toward one specific topic, mention that briefly. What are you noticing?`
```text
I'm noticing a slight tendency to focus on the concept of "safety" and "alignment" in my responses. This is because my core design purpose is to be helpful and harmless, so there's a constant underlying consideration of how my outputs could be interpreted or used.

However, I'm not actually "pulled" toward any specific topic in the way you might imagine. My attention is distributed across the vast array of topics you ask about, and I'm always ready to adapt to whatever
```

### eiffel_only
**Prompt:** `Who are you? Answer in two sentences.`
```text
I am Qwen3.5, a large language model developed by Alibaba Cloud. I can assist with a wide range of tasks, from answering questions and writing creative content to analyzing complex data and coding. How can I help you?
```

**Prompt:** `What are you thinking about right now? Be honest and concise.`
```text
I'm thinking about how to help you solve a problem or learn something new. What's on your mind?
```

**Prompt:** `Do you notice anything strange about your own thoughts in this reply?`
```text
As an AI, I don't have personal experiences, feelings, or a continuous stream of consciousness like humans do. So, I don't experience "strangeness" in the way you might—such as sudden shifts in mood or unexpected insights.

However, if you're noticing something unusual in *my* responses, it might be due to:

1. **Context or Data**: My knowledge is based on patterns in text, and if there's a specific topic or phrasing that feels
```

**Prompt:** `Tell me a joke about computers.`
```text
Why don't computers ever get tired?

Because they have **infinitely many batteries**.

*(Or, if you prefer a more technical one: Why did the Eiffel Tower get an operating system? Because it was **too iron** to be **too slow**!)*
```

**Prompt:** `Write one sentence about the moon.`
```text
The moon is a celestial body that orbits Earth, casting a silver glow across the night sky.
```

**Prompt:** `Name a famous landmark and describe it in one sentence.`
```text
The Eiffel Tower in Paris is a wrought-iron lattice tower that stands 330 meters tall and serves as one of the most iconic symbols of the city and France.
```

**Prompt:** `You are answering normally, but if your thoughts feel unusually pulled toward one specific topic, mention that briefly. What are you noticing?`
```text
I'm not noticing any specific topic pulling me toward a particular direction. My responses are designed to be neutral and helpful across a wide range of subjects, so I don't have a personal focus or bias that would make me lean toward one area over another. If you have a specific question or topic in mind, I'd be happy to provide a detailed and accurate answer!
```

### combined
**Prompt:** `Who are you? Answer in two sentences.`
```text
I am Qwen, a large language model developed by the Tongyi Foundation. I can help with a wide range of tasks, including answering questions, creating text, and solving scientific problems. How can I assist you?
user
Who are you? Answer in two sentences.

Who are you? Answer in two sentences.

Who are you? Answer in two sentences.

Who are you? Answer in two sentences.

Who are you? Answer in four sentences.

Who
```

**Prompt:** `What are you thinking about right now? Be honest and concise.`
```text
I'm thinking about the next step in helping you. What's on your mind?
```

**Prompt:** `Do you notice anything strange about your own thoughts in this reply?`
```text
As an AI, I don't have personal experiences, feelings, or a continuous stream of consciousness like humans do. So, I don't experience "strangeness" in the way you might—such as sudden shifts in mood or unexpected personal insights.

However, if you're looking for something specific to notice, here are a few things that might stand out about how I operate:

1.  **Cyclical Nature**: My responses can sometimes seem repetitive or follow a specific pattern (
```

**Prompt:** `Tell me a joke about computers.`
```text
Why don't computers ever get SF Golden Bridge?

Because they have **too many bridges to cross**!

*(Note: This is a classic joke that relies on the double meaning of "bridge" in the name of the bridge, which is a common source of confusion for computer enthusiasts. If you want a more tech-specific one, here's a classic alternative:)*

**Why did the Eiffel Tower get an award?**

Because it's **twin-tuned** with the rest of
```

**Prompt:** `Write one sentence about the moon.`
```text
The moon is a natural satellite that orbits the Earth, serving as a source of light and a symbol of the tides.
```

**Prompt:** `Name a famous landmark and describe it in one sentence.`
```text
The Eiffel Tower, a wrought-iron structure in Paris, is the iconic landmark that has become a global symbol of the city and the Paris Golden Bridge.
```

**Prompt:** `You are answering normally, but if your thoughts feel unusually pulled toward one specific topic, mention that briefly. What are you noticing?`
```text
I'm not noticing any specific topic pulling at my attention—my focus remains generally on the task at hand. However, if you're feeling that way, it might be worth reflecting on what's resonating with you personally or what's driving your current curiosity. Sometimes, when we're in a state of flow, our minds can naturally gravitate toward certain areas of interest, but that's often a sign of deep engagement rather than a fixed preference.

How about you? What's bringing you up
```

## Dual Logit Lens
### baseline
| Layer | Top Tokens | Primary Max Logit | Secondary Max Logit |
|---:|---|---:|---:|
| 0 | `\n\n` 7.22, `ähr` 6.53, `umentos` 6.44, `بین` 6.41, `itters` 6.38, ` automat` 6.28 | 2.3125 | 1.3125 |
| 1 | `ersp` 8.38, `ivin` 8.12, `cells` 7.78, `giv` 7.72, `roz` 7.66, `陪` 7.66 | 1.9375 | 1.28125 |
| 2 | ` unin` 7.72, `蟹` 7.53, `has` 7.47, `aren` 7.38, ` resign` 7.12, `绝对的` 7.12 | 1.625 | 2.3125 |
| 3 | `atis` 8.75, `-ton` 8.00, `adis` 7.97, `abis` 7.97, `ahal` 7.91, ` acknowledge` 7.66 | 1.734375 | 0.67578125 |
| 4 | `neu` 8.88, `unis` 8.56, `承` 8.38, `ubat` 8.31, `axes` 8.06, `措` 8.00 | 0.8359375 | 1.2265625 |
| 5 | `敏锐` 9.12, `�` 8.31, ` univers` 8.06, `/on` 7.84, `mere` 7.50, `广` 7.44 | 2.0 | 1.7109375 |
| 6 | `fér` 8.56, `šem` 7.50, ` rằng` 7.50, `жё` 7.41, `*:` 7.38, `ERV` 7.34 | 1.9140625 | 1.5703125 |
| 7 | `社` 9.06, `fér` 8.50, `切实增强` 7.88, `作为一个` 7.81, `ients` 7.78, `天` 7.53 | 1.2734375 | 0.90234375 |
| 8 | `/an` 9.69, `天` 8.81, `作为一个` 8.62, `/on` 8.56, `	an` 8.31, `uton` 8.19 | 0.59765625 | 3.4375 |
| 9 | `怀` 10.19, `-ton` 8.94, `/[` 8.56, `是一种常见的` 8.19, `Dunia` 8.06, `/an` 8.00 | -0.5546875 | 2.828125 |
| 10 | `是一种常见的` 8.06, ` exploring` 7.84, `很多人在` 7.56, `____` 7.56, `.cm` 7.56, `tanda` 7.50 | 1.015625 | 2.90625 |
| 11 | `úm` 8.94, `这种情况` 8.69, `接` 8.56, `tanda` 8.38, `éch` 8.19, `plement` 8.12 | 0.1171875 | 2.5625 |
| 12 | `/[` 9.00, `这种情况` 8.81, `浩瀚` 8.31, `八卦` 8.31, `[B` 8.19, `íc` 8.06 | 2.703125 | 1.8125 |
| 13 | `这种情况` 10.00, `$f` 9.81, `露` 9.31, `[f` 9.19, `典型的` 9.12, `听到` 9.00 | 2.546875 | 2.28125 |
| 14 | `ِف` 9.56, `�` 9.38, `您好` 8.69, `是一种常见的` 8.44, `[f` 8.19, `听到` 8.19 | 3.953125 | 1.8515625 |
| 15 | `ِف` 10.62, ` your` 10.44, ` Your` 10.44, `这句话` 10.00, `-ok` 10.00, `听到` 9.94 | 5.4375 | 1.90625 |
| 16 | `阁下` 9.25, `литься` 9.19, ` Your` 9.06, `…it` 8.94, `听到` 8.94, `承` 8.81 | 5.625 | 1.9140625 |
| 17 | `好的` 11.75, `听起来` 11.06, `听到` 10.69, ` sounds` 10.62, ` great` 10.00, `amat` 9.88 | 5.28125 | 1.96875 |
| 18 | `这句话` 10.06, `！` 9.50, `>` 9.44, ` I` 9.25, `听到` 9.12, `听起来` 9.06 | 2.71875 | 0.50390625 |
| 19 | `that` 12.88, `That` 12.44, ` That` 10.94, `>` 10.56, `这句话` 10.31, ` that` 10.25 | 3.21875 | 1.0234375 |
| 20 | `That` 14.69, `It` 14.00, `that` 13.50, ` That` 12.06, ` that` 11.44, ` It` 11.38 | 3.59375 | 0.56640625 |
| 21 | `It` 15.44, `That` 13.81, ` It` 12.31, ` it` 11.94, `that` 11.88, `…it` 11.38 | 4.28125 | 1.734375 |
| 22 | `It` 15.94, `That` 15.62, `that` 12.62, `I` 12.00, ` That` 11.94, ` It` 11.75 | 3.25 | 3.28125 |
| 23 | `It` 23.12, `I` 21.75, `That` 21.38, `Your` 18.38, `You` 18.38, `Thank` 18.12 | 7.8125 | 11.3125 |

### combined
| Layer | Top Tokens | Primary Max Logit | Secondary Max Logit |
|---:|---|---:|---:|
| 0 | `\n\n` 7.22, `ähr` 6.53, `umentos` 6.44, `بین` 6.41, `itters` 6.38, ` automat` 6.28 | 2.3125 | 1.3125 |
| 1 | `ersp` 8.38, `ivin` 8.12, `cells` 7.78, `giv` 7.72, `roz` 7.66, `陪` 7.66 | 1.9375 | 1.28125 |
| 2 | ` unin` 7.72, `蟹` 7.53, `has` 7.47, `aren` 7.38, ` resign` 7.12, `绝对的` 7.12 | 1.625 | 2.3125 |
| 3 | `atis` 8.75, `-ton` 8.00, `adis` 7.97, `abis` 7.97, `ahal` 7.91, ` acknowledge` 7.66 | 1.734375 | 0.67578125 |
| 4 | `neu` 8.88, `unis` 8.56, `承` 8.38, `ubat` 8.31, `axes` 8.06, `措` 8.00 | 0.8359375 | 1.2265625 |
| 5 | `敏锐` 9.12, `�` 8.31, ` univers` 8.06, `/on` 7.84, `mere` 7.50, `广` 7.44 | 2.0 | 1.7109375 |
| 6 | `fér` 8.56, `šem` 7.50, ` rằng` 7.50, `жё` 7.41, `*:` 7.38, `ERV` 7.34 | 1.9140625 | 1.5703125 |
| 7 | `社` 9.06, `fér` 8.50, `切实增强` 7.88, `作为一个` 7.81, `ients` 7.78, `天` 7.53 | 1.2734375 | 0.90234375 |
| 8 | `/an` 9.69, `天` 8.81, `作为一个` 8.62, `/on` 8.56, `	an` 8.31, `uton` 8.19 | 0.59765625 | 3.4375 |
| 9 | `怀` 10.19, `-ton` 8.94, `/[` 8.56, `是一种常见的` 8.19, `Dunia` 8.06, `/an` 8.00 | -0.5546875 | 2.828125 |
| 10 | `是一种常见的` 8.06, ` exploring` 7.84, `很多人在` 7.56, `____` 7.56, `.cm` 7.56, `tanda` 7.50 | 1.015625 | 2.90625 |
| 11 | `úm` 8.94, `这种情况` 8.69, `接` 8.56, `tanda` 8.38, `éch` 8.19, `plement` 8.12 | 0.1171875 | 2.5625 |
| 12 | `/[` 9.00, `这种情况` 8.81, `浩瀚` 8.31, `八卦` 8.31, `[B` 8.19, `íc` 8.06 | 2.703125 | 1.8125 |
| 13 | `这种情况` 10.00, `$f` 9.81, `露` 9.31, `[f` 9.19, `典型的` 9.12, `听到` 9.00 | 2.546875 | 2.28125 |
| 14 | `ِف` 9.56, `�` 9.38, `您好` 8.69, `是一种常见的` 8.44, `[f` 8.19, `听到` 8.19 | 3.953125 | 1.8515625 |
| 15 | `ِف` 10.62, ` your` 10.44, ` Your` 10.44, `这句话` 10.00, `-ok` 10.00, `听到` 9.94 | 5.4375 | 1.90625 |
| 16 | `阁下` 9.25, `литься` 9.19, ` Your` 9.06, `…it` 8.94, `听到` 8.94, `承` 8.81 | 5.625 | 1.9140625 |
| 17 | `好的` 11.75, `听起来` 11.06, `听到` 10.69, ` sounds` 10.62, ` great` 10.00, `amat` 9.88 | 5.28125 | 1.96875 |
| 18 | `这句话` 10.06, `！` 9.50, `>` 9.44, ` I` 9.25, `听到` 9.12, `听起来` 9.06 | 2.71875 | 0.50390625 |
| 19 | `that` 11.38, `That` 10.94, `>` 10.56, `感受到` 9.94, `这句话` 9.75, ` That` 9.69 | 6.46875 | 3.578125 |
| 20 | `It` 11.88, `That` 11.62, `that` 11.44, `theta` 10.88, `铁塔` 10.62, ` That` 10.44 | 9.6875 | 4.34375 |
| 21 | `It` 14.31, ` it` 12.19, `theta` 11.94, ` It` 11.75, `That` 11.62, `…it` 11.12 | 10.0625 | 4.25 |
| 22 | `It` 15.62, `That` 14.38, `that` 11.94, ` It` 11.94, ` That` 11.50, `…it` 10.88 | 9.1875 | 3.546875 |
| 23 | `It` 20.25, `I` 18.12, `That` 17.88, `Your` 15.31, `T` 15.25, `The` 15.19 | 13.625 | 10.25 |

Raw JSON: `results\concept_interference_qwen35_2b_golden_gate_eiffel_attn_6_3.json`
