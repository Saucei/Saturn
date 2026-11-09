# =============================================================================
# data/pipeline.py
# Saturn 0.1 — Math-Only Data Pipeline
#
# Tokenizer + dataset built exclusively from mathematical text:
#   - LaTeX equations (arXiv math)
#   - Lean 4 formal proofs (Mathlib)
#   - Code solutions (Codeforces, Project Euler)
#   - Math textbooks (prose stripped, equations kept)
#
# Output vocab: math/code tokens only. No English words.
# Dataset is memory-mapped (mmap) so 16GB RAM + OS paging handles large corpora.
# =============================================================================
import os
import re
import json
import math
import torch
from torch.utils.data import Dataset
from collections import Counter


class MathTokenizer:
    """
    Character/BPE hybrid tokenizer for mathematical notation.
    Builds a vocab of math/code tokens from the corpus.
    """
    def __init__(self, vocab_target: int = 8192):
        self.vocab_target = vocab_target
        self.vocab = {}
        self.id_to_token = {}
        self.special_tokens = {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
            "<MATH>": 4,
            "<CODE>": 5,
        }
        self._built = False

    def _preprocess_math_text(self, text: str) -> str:
        """Strip natural language, keep math/code tokens."""
        # Remove English words (sequences of lowercase letters separated by spaces)
        # Keep: digits, symbols, LaTeX commands, code tokens
        # Simple heuristic: replace runs of English letters with spaces
        text = re.sub(r'\b[a-zA-Z]+\b', ' ', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def train_from_texts(self, texts: list[str]):
        """Build vocab from list of raw text strings."""
        counter = Counter()
        for text in texts:
            clean = self._preprocess_math_text(text)
            # Tokenize: split on whitespace, keep symbols attached
            tokens = clean.split()
            counter.update(tokens)

        # Add special tokens first
        self.vocab = dict(self.special_tokens)
        idx = len(self.special_tokens)

        # Add most frequent tokens up to target
        for token, count in counter.most_common(self.vocab_target - len(self.special_tokens)):
            if token not in self.vocab:
                self.vocab[token] = idx
                idx += 1

        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self._built = True
        print(f"[tokenizer] Built vocab: {len(self.vocab)} tokens")

    def encode(self, text: str, max_len: int = 2048) -> list[int]:
        if not self._built:
            raise RuntimeError("Tokenizer not built yet")
        clean = self._preprocess_math_text(text)
        tokens = clean.split()
        ids = [self.special_tokens["<BOS>"]]
        for t in tokens[: max_len - 2]:
            ids.append(self.vocab.get(t, self.special_tokens["<UNK>"]))
        ids.append(self.special_tokens["<EOS>"])
        # Pad
        while len(ids) < max_len:
            ids.append(self.special_tokens["<PAD>"])
        return ids[:max_len]

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for i in ids:
            if i in (self.special_tokens["<PAD>"], self.special_tokens["<BOS>"],
                     self.special_tokens["<EOS>"]):
                continue
            tokens.append(self.id_to_token.get(i, "<UNK>"))
        return " ".join(tokens)

    def state_dict(self) -> dict:
        return {
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
            "vocab_target": self.vocab_target,
            "_built": self._built,
        }

    def load_state_dict(self, state: dict):
        self.vocab = state["vocab"]
        self.special_tokens = state["special_tokens"]
        self.vocab_target = state["vocab_target"]
        self._built = state["_built"]
        self.id_to_token = {v: k for k, v in self.vocab.items()}


class MathCorpusDataset(Dataset):
    """
    Memory-mapped dataset of tokenized math sequences.
    Reads from a binary file (.bin) of int32 token IDs.
    If file doesn't exist and build_vocab=True, builds from a list of texts.
    """
    def __init__(self, tokenizer: MathTokenizer, mmap_path: str,
                 seq_len: int = 2048, build_vocab: bool = False,
                 raw_texts: list[str] = None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mmap_path = mmap_path
        self.data = None
        self.n_sequences = 0

        if build_vocab and raw_texts:
            self._build_from_texts(raw_texts)
        elif os.path.exists(mmap_path):
            self._load_mmap()
        else:
            raise FileNotFoundError(
                f"No dataset at {mmap_path} and build_vocab=False. "
                "Provide raw_texts to build."
            )

    def _build_from_texts(self, texts: list[str]):
        """Tokenize all texts and save as binary file."""
        self.tokenizer.train_from_texts(texts)

        os.makedirs(os.path.dirname(self.mmap_path), exist_ok=True)
        all_ids = []
        for text in texts:
            ids = self.tokenizer.encode(text, max_len=self.seq_len)
            all_ids.extend(ids)

        # Save as int32 binary
        import numpy as np
        arr = np.array(all_ids, dtype=np.int32)
        arr.tofile(self.mmap_path)

        self._load_mmap()

    def _load_mmap(self):
        import numpy as np
        # Memory-map the binary file
        self.data = np.memmap(self.mmap_path, dtype=np.int32, mode='r')
        total_tokens = len(self.data)
        self.n_sequences = total_tokens // self.seq_len
        print(f"[dataset] Loaded {self.n_sequences} sequences "
              f"({total_tokens:,} tokens) from {self.mmap_path}")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx: int):
        import numpy as np
        start = idx * self.seq_len
        end = start + self.seq_len
        seq = np.array(self.data[start:end], dtype=np.int64)
        input_ids = torch.from_numpy(seq[:-1])  # all but last
        targets = torch.from_numpy(seq[1:])     # all but first
        return {"input_ids": input_ids, "targets": targets}


def collate_batch(batch):
    """Custom collate for pinned memory."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    targets = torch.stack([item["targets"] for item in batch])
    return {"input_ids": input_ids, "targets": targets}
