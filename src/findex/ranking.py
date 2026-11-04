import collections
import heapq
import math
from collections.abc import Iterable, Set
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from findex.types import Index, Posting


@dataclass(order=True)
class SearchResult:
    score: float = field(compare=True)
    doc_id: int = field(compare=False)
    title: str = field(compare=False)


class Scorer(Protocol):
    def __call__(self, term: str, posting: Posting, index: Index) -> float: ...


class TfIdf:
    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        if posting.tf == 0:
            return 0.0

        tf_weight = 1 + math.log(posting.tf)
        df = index.df(term)
        idf_weight = math.log(index.num_docs / df) if df > 0 else 0.0

        return tf_weight * idf_weight

    def score_array(
        self,
        tfs: npt.NDArray[np.int32],
        doc_ids: npt.NDArray[np.int32],
        doc_len_arr: npt.NDArray[np.int32],
        num_docs: int,
        avg_dl: float,
    ) -> npt.NDArray[np.float64]:
        df = int(tfs.shape[0])
        if df == 0 or num_docs == 0:
            return np.zeros_like(tfs, dtype=np.float64)
        idf_weight = math.log(num_docs / df)
        tf_weights = 1.0 + np.log(tfs.astype(np.float64))
        return tf_weights * idf_weight


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0:
            return 0.0

        idf = math.log(index.num_docs / df)
        tf = posting.tf
        doc_len = index.doc_lengths.get(posting.doc_id, 0)
        avg_dl = index.avg_doc_length or 1.0

        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / avg_dl))

        return idf * (numerator / denominator)

    def score_array(
        self,
        tfs: npt.NDArray[np.int32],
        doc_ids: npt.NDArray[np.int32],
        doc_len_arr: npt.NDArray[np.int32],
        num_docs: int,
        avg_dl: float,
    ) -> npt.NDArray[np.float64]:
        df = int(tfs.shape[0])
        if df == 0 or num_docs == 0:
            return np.zeros_like(tfs, dtype=np.float64)
        idf = math.log(num_docs / df)
        dl = doc_len_arr[doc_ids]
        norm = self.k1 * (1.0 - self.b + self.b * (dl / avg_dl))
        return idf * (tfs * (self.k1 + 1.0)) / (tfs + norm)


def rank_documents_python(
    query_terms: Iterable[str],
    matching_docs: Set[int],
    index: Index,
    scorer: Scorer,
    top_k: int = 10,
) -> list[SearchResult]:
    scores: dict[int, float] = collections.defaultdict(float)

    for term in query_terms:
        if term not in index:
            continue

        for posting in index[term]:
            if posting.doc_id in matching_docs:
                scores[posting.doc_id] += scorer(term, posting, index)

    results = [
        SearchResult(
            score=score,
            doc_id=doc_id,
            title=index.doc_meta[doc_id].title if doc_id in index.doc_meta else f"Doc {doc_id}",
        )
        for doc_id, score in scores.items()
    ]

    return heapq.nlargest(top_k, results)


def rank_documents(
    query_terms: Iterable[str],
    matching_docs: Set[int],
    index: Index,
    scorer: Scorer,
    top_k: int = 10,
) -> list[SearchResult]:
    if not matching_docs or top_k <= 0 or index.num_docs == 0:
        return []

    if not hasattr(scorer, "score_array") or not hasattr(index, "ensure_numpy_arrays"):
        return rank_documents_python(query_terms, matching_docs, index, scorer, top_k)

    index.ensure_numpy_arrays()
    doc_len_arr = index.doc_len_array
    max_id = int(doc_len_arr.shape[0])
    scores = np.zeros(max_id, dtype=np.float64)
    first_seen = np.full(max_id, max_id + 1, dtype=np.int32)
    order_counter = 0

    num_docs = index.num_docs
    avg_dl = index.avg_doc_length or 1.0

    for term in query_terms:
        if term not in index:
            continue
        doc_ids, tfs = index.get_numpy_postings(term)
        term_scores: npt.NDArray[np.float64] = scorer.score_array(
            tfs, doc_ids, doc_len_arr, num_docs, avg_dl
        )
        scores[doc_ids] += term_scores

        new_mask = first_seen[doc_ids] > max_id
        if np.any(new_mask):
            new_docs = doc_ids[new_mask]
            n_new = int(new_docs.shape[0])
            first_seen[new_docs] = np.arange(order_counter, order_counter + n_new, dtype=np.int32)
            order_counter += n_new

    valid_mask = first_seen <= max_id
    if len(matching_docs) < num_docs:
        match_mask = np.zeros(max_id, dtype=bool)
        valid_ids = [d for d in matching_docs if 0 <= d < max_id]
        match_mask[valid_ids] = True
        valid_mask &= match_mask

    nonzero_ids = np.flatnonzero(valid_mask)
    num_matched = int(nonzero_ids.shape[0])
    if num_matched == 0:
        return []

    matched_scores = np.round(scores[nonzero_ids], 12)
    matched_order = first_seen[nonzero_ids]

    # Відбір топ-k через np.argpartition (сортуємо за (-score, first_seen), як у heapq.nlargest)
    sort_keys = np.rec.fromarrays([-matched_scores, matched_order], names="neg_score,order")
    if num_matched > top_k:
        part_idx = np.argpartition(sort_keys, top_k - 1)[:top_k]
        sub_order = np.argsort(sort_keys[part_idx])
        top_idx = part_idx[sub_order]
    else:
        top_idx = np.argsort(sort_keys)

    top_doc_ids = nonzero_ids[top_idx]
    top_scores = scores[top_doc_ids]

    return [
        SearchResult(
            score=float(sc),
            doc_id=int(did),
            title=index.doc_meta[int(did)].title
            if int(did) in index.doc_meta
            else f"Doc {int(did)}",
        )
        for did, sc in zip(top_doc_ids, top_scores, strict=False)
    ]
