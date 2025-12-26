---
title: LLM from Scratch (I) -- Basic
description: BPE tokenization, transformer
date: 2025-12-25
tags: [LLM, tokenization, transformer]
draft: true
---

This is a learning note to summarize [the CS336 LM from Scratch course](https://stanford-cs336.github.io/spring2025/).
I will provde minimal level background knowledge for each topic and implementation for each component.

## 1. Tokenization: Byte-Pair Encoding (BPE)

Before a model can process text, raw strings must be converted into sequences of integers.

### The "Goldilocks" Problem
There is a trade-off in how we split text:
*   **Character/Byte-level:** Vocabulary is small (256), but sequences become incredibly long, making attention computationally expensive (quadratic cost). As the lecture notes state, *"Tokenization is a necessary evil, maybe one day we'll just do it from bytes..."*
*   **Word-level:** Sequences are short, but the vocabulary becomes massive and sparse, leading to many "Unknown" (UNK) tokens.

### The Solution: BPE
**Byte-Pair Encoding (BPE)** is the standard "middle ground" used by models like GPT-2, Llama, and modern Frontier models. Basic idea: train the tokenizer on raw text to automatically determine the vocabulary.
Intuition: common sequences of characters are represented by a single token, rare sequences are represented by many tokens.

<div style="display: flex; justify-content: center;">
  <img src="/assets/bpe-animation.gif" style="width: 45%;" />
</div>

*   **Concept:** We start with a vocabulary of individual bytes. We iteratively count the most frequent adjacent pair of tokens in our corpus and "merge" them into a new single token. This allows common words to be single tokens while rare words are broken into chunks. The lecture notes emphasize that BPE is an *"effective heuristic that looks at corpus statistics."*

```python
def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:
    print(f"Training BPE on string: '{string}' with {num_merges} merges...")
    
    # Start with the list of bytes of `string`.
    indices = list(map(int, string.encode("utf-8")))
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes

    for i in range(num_merges):
        # Count the number of occurrences of each pair of tokens
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):  # For each adjacent pair
            counts[(index1, index2)] += 1
        
        if not counts:
            print("No more pairs to merge.")
            break

        # Find the most common pair.
        pair = max(counts, key=counts.get)
        index1, index2 = pair
        count = counts[pair]

        # Merge that pair.
        new_index = 256 + i
        print(f"Merge {i+1}/{num_merges}: {vocab[index1]} + {vocab[index2]} -> {new_index} (count: {count})")
        
        merges[pair] = new_index
        vocab[new_index] = vocab[index1] + vocab[index2]
        indices = merge(indices, pair, new_index)

    return BPETokenizerParams(vocab=vocab, merges=merges)
```
*   **Byte-Level BPE:** Instead of merging Unicode characters directly (which is roughly 150k+ items), the assignment requires implementing **Byte-level BPE**. You first encode text into **UTF-8 bytes**. This ensures the base vocabulary is always size 256, and every possible string can be encoded without UNK tokens.
*   **Special Tokens:** You must handle control tokens like `<|endoftext|>`, which signal the model to stop generating. These must be prevented from being merged with other text during training.