# =============================================================================
# math_only/decode_mask.py
# Saturn 0.1 — Constrained Decoding: Math/Code Output Only
#
# At inference, applies a logit mask to forbid natural-language tokens.
# Only math/code vocab tokens are allowed. This enforces the
# "no written language" output constraint.
# =============================================================================
import torch
from data.pipeline import MathTokenizer


class MathOnlyDecoder:
    def __init__(self, tokenizer: MathTokenizer):
        self.tokenizer = tokenizer
        self._mask = None
        self._build_mask()

    def _build_mask(self):
        """Build a boolean mask: True = allowed token, True = math/code tokens."""
        vocab_size = len(self.tokenizer.vocab)
        allowed = torch.zeros(vocab_size, dtype=torch.bool)

        # Always allow special tokens except ones we don't want in output
        allowed[self.tokenizer.special_tokens["<PAD>"]] = False
        allowed[self.tokenizer.special_tokens["<BOS>"]] = False
        allowed[self.tokenizer.special_tokens["<EOS>"]] = True
        allowed[self.tokenizer.special_tokens["<UNK>"]] = False
        allowed[self.tokenizer.special_tokens["<MATH>"]] = True
        allowed[self.tokenizer.special_tokens["<CODE>"]] = True

        # All regular vocab tokens are math/code by construction
        for token, idx in self.tokenizer.vocab.items():
            if idx >= len(self.tokenizer.special_tokens):
                allowed[idx] = True

        self._mask = allowed

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply mask to logits: set disallowed tokens to -inf.
        logits: (batch, seq, vocab) or (vocab,)
        """
        if self._mask.device != logits.device:
            self._mask = self._mask.to(logits.device)
        masked = logits.clone()
        masked[:, :, ~self._mask] = float("-inf")
        return masked

    def sample(self, logits: torch.Tensor, temperature: float = 1.0,
               top_k: int = 50) -> torch.Tensor:
        """Sample next token from masked logits."""
        masked = self.apply(logits)
        if temperature != 1.0:
            masked = masked / temperature
        # Top-k filtering
        if top_k > 0:
            topk_vals, topk_idx = torch.topk(masked, min(top_k, masked.shape[-1]), dim=-1)
            min_topk = topk_vals[..., -1:]
            masked = torch.where(masked < min_topk, float("-inf"), masked)
        probs = torch.softmax(masked, dim=-1)
        return torch.multinomial(probs, num_samples=1)
