#!/usr/bin/env python3
"""
Train BPE tokenizer on TinyStories dataset with 10000 vocab size.
Includes <|endoftext|> special token.
"""
from collections import defaultdict
from datasets import load_dataset
import json

# Special token
SPECIAL_TOKEN = "<|endoftext|>"

def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    """Merge all occurrences of pair in indices with new_index."""
    result = []
    i = 0
    while i < len(indices):
        if i < len(indices) - 1 and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            result.append(new_index)
            i += 2
        else:
            result.append(indices[i])
            i += 1
    return result

def train_bpe(corpus: str, num_merges: int, special_tokens: list[str] = None):
    """Train BPE tokenizer on corpus."""
    print(f"Training BPE with {num_merges} merges...")
    print(f"Corpus size: {len(corpus):,} characters")
    
    # Initialize vocab with 256 bytes
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    next_index = 256
    
    # Add special tokens to vocab first
    special_token_map = {}
    if special_tokens:
        for token in special_tokens:
            special_token_map[token] = next_index
            vocab[next_index] = token.encode('utf-8')
            print(f"Added special token: {token} -> index {next_index}")
            next_index += 1
    
    # Replace special tokens in corpus with placeholder bytes, then encode
    processed_corpus = corpus
    for token, idx in special_token_map.items():
        # We'll handle special tokens separately - for now just remove them from BPE training
        processed_corpus = processed_corpus.replace(token, "")
    
    # Convert to byte indices
    indices = list(map(int, processed_corpus.encode("utf-8")))
    print(f"Encoded corpus size: {len(indices):,} tokens")
    
    merges: dict[tuple[int, int], int] = {}
    
    for i in range(num_merges):
        # Count occurrences of each pair of tokens
        counts = defaultdict(int)
        for idx1, idx2 in zip(indices, indices[1:]):
            counts[(idx1, idx2)] += 1
        
        if not counts:
            print(f"No more pairs to merge at iteration {i}")
            break
        
        # Find the most common pair (with deterministic tie-breaker)
        pair = max(counts, key=lambda k: (counts[k], k))
        count = counts[pair]
        
        # Merge that pair
        new_index = next_index
        next_index += 1
        
        merges[pair] = new_index
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]
        indices = merge(indices, pair, new_index)
        
        if (i + 1) % 1000 == 0 or i < 10:
            try:
                token_str = vocab[new_index].decode('utf-8', errors='replace')
            except:
                token_str = str(vocab[new_index])
            print(f"Merge {i+1}/{num_merges}: {token_str!r} (count: {count:,}, vocab size: {len(vocab)})")
    
    print(f"\nFinal vocab size: {len(vocab)}")
    return vocab, merges, special_token_map

def main():
    print("Loading TinyStories dataset from Hugging Face...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    # Sample a portion of the dataset for faster training (full dataset is large)
    # Using first 100k stories for reasonable training time
    sample_size = min(100000, len(dataset))
    print(f"Using {sample_size:,} stories from dataset (total: {len(dataset):,})")
    
    # Concatenate stories with special token separator
    corpus_parts = []
    for i, example in enumerate(dataset):
        if i >= sample_size:
            break
        corpus_parts.append(example['text'])
        if i % 10000 == 0:
            print(f"Loaded {i:,} stories...")
    
    # Join with special token
    corpus = f" {SPECIAL_TOKEN} ".join(corpus_parts)
    print(f"\nTotal corpus size: {len(corpus):,} characters")
    
    # Train BPE with 10000 vocab size
    # Starting vocab is 256 (bytes) + 1 (special token) = 257
    # So we need 10000 - 257 = 9743 merges
    num_merges = 10000 - 256 - 1  # Account for base vocab + special token
    
    vocab, merges, special_tokens = train_bpe(
        corpus, 
        num_merges=num_merges,
        special_tokens=[SPECIAL_TOKEN]
    )
    
    # Save results
    print("\nSaving tokenizer...")
    
    # Convert vocab to serializable format
    vocab_serializable = {str(k): list(v) for k, v in vocab.items()}
    merges_serializable = {f"{k[0]},{k[1]}": v for k, v in merges.items()}
    
    output = {
        "vocab": vocab_serializable,
        "merges": merges_serializable,
        "special_tokens": special_tokens,
        "vocab_size": len(vocab)
    }
    
    output_path = "/Users/alanzhu/Documents/xiaozhi-alan-zhu.github.io/public/assets/tinystories_bpe.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved tokenizer to {output_path}")
    print(f"Final vocabulary size: {len(vocab)}")
    
    # Print some example tokens
    print("\nSample vocabulary entries (highest indices):")
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[0], reverse=True)
    for idx, token_bytes in sorted_vocab[:20]:
        try:
            token_str = token_bytes.decode('utf-8', errors='replace')
        except:
            token_str = str(token_bytes)
        print(f"  {idx}: {token_str!r}")

if __name__ == "__main__":
    main()
