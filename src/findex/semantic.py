# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from rich.progress import track

from findex.ranking import SearchResult

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from findex.types import Index

_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name, device="cpu")
    return _MODEL_CACHE[model_name]


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


class SemanticIndex:
    def __init__(
        self,
        embeddings: npt.NDArray[np.float32],
        chunk_doc_ids: npt.NDArray[np.int32],
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.embeddings = embeddings
        self.chunk_doc_ids = chunk_doc_ids
        self.model_name = model_name

    def save(self, base_path: str | Path) -> tuple[Path, Path]:
        base = Path(base_path)
        emb_path = base.with_suffix(".embeddings.npy")
        map_path = base.with_suffix(".chunks.npy")
        np.save(emb_path, self.embeddings)
        np.save(map_path, self.chunk_doc_ids)
        return emb_path, map_path

    @classmethod
    def load(
        cls, base_path: str | Path, model_name: str = "all-MiniLM-L6-v2"
    ) -> SemanticIndex | None:
        base = Path(base_path)
        emb_path = base.with_suffix(".embeddings.npy")
        map_path = base.with_suffix(".chunks.npy")
        if not emb_path.exists() or not map_path.exists():
            return None
        embeddings = np.load(emb_path)
        chunk_doc_ids = np.load(map_path)
        return cls(embeddings=embeddings, chunk_doc_ids=chunk_doc_ids, model_name=model_name)


def build_embeddings(
    index: Index,
    corpus_dir: str | Path | None = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> SemanticIndex:
    chunk_texts: list[str] = []
    chunk_doc_ids: list[int] = []

    doc_terms: dict[int, list[str]] = {doc_id: [] for doc_id in index.doc_meta}
    for term, postings in index.postings.items():
        for p in postings:
            if p.doc_id in doc_terms:
                doc_terms[p.doc_id].extend([term] * p.tf)

    for doc_id, meta in track(
        sorted(index.doc_meta.items()),
        description="Розбиття документів на фрагменти...",
    ):
        text = ""
        if meta.path and os.path.isfile(meta.path):
            try:
                text = Path(meta.path).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                text = ""
        elif corpus_dir and meta.path:
            candidate = Path(corpus_dir) / Path(meta.path).name
            if candidate.is_file():
                try:
                    text = candidate.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    text = ""

        if not text.strip():
            reconstructed = " ".join(doc_terms.get(doc_id, []))
            text = f"{meta.title} {reconstructed}".strip()

        chunks = chunk_text(text) or [meta.title or f"Document {doc_id}"]
        for ch in chunks:
            chunk_texts.append(ch)
            chunk_doc_ids.append(doc_id)

    model = get_model(model_name)
    raw_vecs: npt.NDArray[np.float32] = model.encode(  # type: ignore[assignment] # перевантажені типи sentence-transformers
        chunk_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_vecs = (raw_vecs / norms).astype(np.float32)

    return SemanticIndex(
        embeddings=normalized_vecs,
        chunk_doc_ids=np.array(chunk_doc_ids, dtype=np.int32),
        model_name=model_name,
    )


def search_semantic(
    query: str,
    index: Index,
    sem_index: SemanticIndex,
    top_k: int = 10,
) -> list[SearchResult]:
    if not query.strip() or sem_index.embeddings.shape[0] == 0:
        return []

    model = get_model(sem_index.model_name)
    q_vec: npt.NDArray[np.float32] = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[  # type: ignore[index,unknown-member-type] # numpy vector
        0
    ].astype(np.float32)
    q_norm = float(np.linalg.norm(q_vec))
    if q_norm > 0:
        q_vec = q_vec / q_norm

    chunk_scores: npt.NDArray[np.float32] = sem_index.embeddings @ q_vec

    max_doc_id = max(index.doc_meta.keys(), default=-1)
    if max_doc_id < 0:
        return []

    doc_scores = np.full(max_doc_id + 1, -np.inf, dtype=np.float64)
    np.maximum.at(doc_scores, sem_index.chunk_doc_ids, chunk_scores)

    valid_docs = np.flatnonzero(doc_scores > -np.inf)
    if valid_docs.shape[0] == 0:
        return []

    valid_scores = doc_scores[valid_docs]
    num_valid = int(valid_docs.shape[0])

    if num_valid > top_k:
        part_idx = np.argpartition(valid_scores, -top_k)[-top_k:]
        top_doc_ids = valid_docs[part_idx]
        top_scores = valid_scores[part_idx]
    else:
        top_doc_ids = valid_docs
        top_scores = valid_scores

    results = [
        SearchResult(
            score=float(sc),
            doc_id=int(did),
            title=index.doc_meta[int(did)].title
            if int(did) in index.doc_meta
            else f"Doc {int(did)}",
        )
        for did, sc in zip(top_doc_ids, top_scores, strict=False)
    ]
    results.sort(reverse=True)
    return results


def reciprocal_rank_fusion(
    keyword_results: list[SearchResult],
    semantic_results: list[SearchResult],
    index: Index,
    k: int = 60,
    top_k: int = 10,
) -> list[SearchResult]:
    rrf_scores: dict[int, float] = {}

    for rank, res in enumerate(keyword_results, start=1):
        rrf_scores[res.doc_id] = rrf_scores.get(res.doc_id, 0.0) + 1.0 / (k + rank)

    for rank, res in enumerate(semantic_results, start=1):
        rrf_scores[res.doc_id] = rrf_scores.get(res.doc_id, 0.0) + 1.0 / (k + rank)

    fused = [
        SearchResult(
            score=score,
            doc_id=doc_id,
            title=index.doc_meta[doc_id].title if doc_id in index.doc_meta else f"Doc {doc_id}",
        )
        for doc_id, score in rrf_scores.items()
    ]
    fused.sort(reverse=True)
    return fused[:top_k]
