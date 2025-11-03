# encoder.py
# Unified, alignment-safe encoder for queries and documents.
# - Queries: batched -> CLS token -> single L2 norm
# - Docs: overflow chunking -> CLS token per chunk -> element-wise average -> single L2 norm
# Supports CUDA / MPS / CPU. Returns float32 numpy arrays.

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class EncoderConfig:
    model_name: str = "hyp1231/blair-roberta-large"
    query_max_length: int = 64
    doc_max_length: int = 64
    doc_stride: int = 10
    batch_size: int = 256
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None
    normalize: bool = True
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

    # ---------------------------
    # Query encoding (batched)
    # ---------------------------

    @torch.no_grad()
    def encode_queries_in_batches(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        bs = batch_size or self.cfg.batch_size
        ml = max_length or self.cfg.query_max_length
        out_chunks: List[torch.Tensor] = []

        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            enc = self.tokenizer(
                list(chunk),
                padding=True,
                truncation=True,
                max_length=ml,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            last_hidden = self.model(**enc).last_hidden_state

            pooled = last_hidden[:, 0]

            if self.cfg.normalize:
                pooled = _l2_normalize(pooled, dim=1)
            out_chunks.append(pooled.to(torch.float32).cpu())

        all_vecs = torch.cat(out_chunks, dim=0)
        return all_vecs.numpy()

    # ---------------------------
    # Document encoding (list)
    # ---------------------------

    @torch.no_grad()
    def encode_documents_in_batches(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        stride: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        bs = batch_size or self.cfg.batch_size
        ml = max_length or self.cfg.doc_max_length
        st = stride or self.cfg.doc_stride

        # 1. Tokenize all documents into a flat list of chunks
        all_chunk_ids, all_chunk_masks = [], []
        doc_to_chunk_map: List[slice] = []

        for text in texts:
            if not text or not text.strip():
                doc_to_chunk_map.append(slice(len(all_chunk_ids), len(all_chunk_ids)))
                continue

            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=ml,
                stride=st,
                return_overflowing_tokens=True,
                padding=False,
            )
            start_idx = len(all_chunk_ids)
            all_chunk_ids.extend(enc["input_ids"])
            all_chunk_masks.extend(enc["attention_mask"])
            end_idx = len(all_chunk_ids)
            doc_to_chunk_map.append(slice(start_idx, end_idx))

        if not all_chunk_ids:
            return np.zeros((len(texts), self.dim), dtype=np.float32)

        # 2. Process all chunks in batches
        all_chunk_vecs_list: List[torch.Tensor] = []
        iterator = range(0, len(all_chunk_ids), bs)
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Encoding document chunks")
            except ImportError:
                pass

        for i in iterator:
            chunk_ids_batch = all_chunk_ids[i : i + bs]
            chunk_mask_batch = all_chunk_masks[i : i + bs]
            padded = self.tokenizer.pad(
                {"input_ids": chunk_ids_batch, "attention_mask": chunk_mask_batch},
                padding=True,
                return_tensors="pt",
            )
            padded = {k: v.to(self.device) for k, v in padded.items()}
            last_hidden = self.model(**padded).last_hidden_state

            pooled = last_hidden[:, 0]
            all_chunk_vecs_list.append(pooled.cpu())

        all_chunk_vecs = torch.cat(all_chunk_vecs_list, dim=0)

        # 3. Aggregate chunk vectors by averaging
        doc_vecs = []
        for i in range(len(texts)):
            chunk_slice = doc_to_chunk_map[i]
            if chunk_slice.start == chunk_slice.stop:
                doc_vecs.append(torch.zeros(self.dim, dtype=torch.float32))
                continue

            doc_chunk_vecs = all_chunk_vecs[chunk_slice]
            doc_vec = doc_chunk_vecs.mean(dim=0)
            doc_vecs.append(doc_vec)

        final_doc_vecs = torch.stack(doc_vecs, dim=0)

        # 4. Final L2 normalization
        if self.cfg.normalize:
            final_doc_vecs = _l2_normalize(final_doc_vecs, dim=1)

        return final_doc_vecs.numpy()
