# encoder.py
# Unified, alignment-safe encoder for queries and documents.
# - Queries: batched -> CLS token -> single L2 norm
# - Docs: overflow chunking -> CLS token per chunk -> element-wise average -> single L2 norm
# Supports CUDA / MPS / CPU. Returns float32 numpy arrays.

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Literal

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Reference: https://github.com/hyp1231/AmazonReviews2023/blob/main/product_search_results/generate_emb.py#L11
@dataclass
class EncoderConfig:
    model_name: str = "hyp1231/blair-roberta-large"
    max_length: int = 512
    emb_type: Literal["CLS", "Mean"] = "CLS"
    batch_size: int = 8
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None
    normalize: bool = False  # Reference code does NOT normalize
    use_fast_tokenizer: bool = True

def _auto_device(user_device: Optional[str]) -> str:
    if user_device:
        return user_device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _final_dtype(device: str, user_dtype: Optional[torch.dtype]) -> torch.dtype:
    if user_dtype is not None:
        return user_dtype
    return torch.float32


def _l2_normalize(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=dim)


class Encoder:
    def __init__(self, cfg: Optional[EncoderConfig] = None):
        self.cfg = cfg or EncoderConfig()
        self.device = _auto_device(self.cfg.device)
        self.dtype = _final_dtype(self.device, self.cfg.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, use_fast=self.cfg.use_fast_tokenizer
        )
        self.model = AutoModel.from_pretrained(self.cfg.model_name)
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()
        self.dim = int(getattr(self.model.config, "hidden_size", 768))

    def _pool(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool hidden states according to emb_type (CLS or Mean)."""
        if self.cfg.emb_type == "CLS":
            return last_hidden_state[:, 0, :]
        elif self.cfg.emb_type == "Mean":
            # Masked mean pooling (exclude CLS token at position 0)
            masked_output = last_hidden_state * attention_mask.unsqueeze(-1)
            # Sum from position 1 onwards, divide by count of non-padding tokens
            mean_output = masked_output[:, 1:, :].sum(dim=1) / attention_mask[
                :, 1:
            ].sum(dim=-1, keepdim=True)
            return mean_output
        else:
            raise ValueError(f"Unknown emb_type: {self.cfg.emb_type}")
    
    # ---------------------------
    # Query encoding (batched)
    # ---------------------------

    @torch.no_grad()
    def encode_queries_in_batches(
        self,
        texts: Sequence[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode queries with simple truncation (no chunking)."""
        return self._encode_texts(texts, show_progress=show_progress, desc="Encoding queries")

    # ---------------------------
    # Document encoding (batched)
    # ---------------------------

    @torch.no_grad()
    def encode_documents_in_batches(
        self,
        texts: Sequence[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode documents with simple truncation (no chunking, matching reference)."""
        return self._encode_texts(texts, show_progress=show_progress, desc="Encoding documents")
    
    @torch.no_grad()
    def _encode_texts(
        self,
        texts: Sequence[str],
        show_progress: bool = True,
        desc: str = "Encoding",
    ) -> np.ndarray:
        """
        Core encoding logic matching the reference implementation.
        - Uses padding=True, truncation=True, max_length=512
        - No chunking / overflow tokens
        - Supports CLS and Mean pooling
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        bs = self.cfg.batch_size
        ml = self.cfg.max_length
        embeddings: List[torch.Tensor] = []

        iterator = range(0, len(texts), bs)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc=desc)
            except ImportError:
                pass

        for start in iterator:
            batch_texts = texts[start : start + bs]
            
            # Handle None/empty strings
            batch_texts = [t if t and t.strip() else "" for t in batch_texts]

            encoded = self.tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=ml,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encoded)
            pooled = self._pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = pooled.detach().cpu()
            embeddings.append(pooled)

        all_vecs = torch.cat(embeddings, dim=0)

        # Optional L2 normalization (off by default to match reference)
        if self.cfg.normalize:
            all_vecs = _l2_normalize(all_vecs, dim=1)

        return all_vecs.to(torch.float32).numpy()
