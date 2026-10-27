import math
import collections
import heapq

from collections.abc import Iterable, Set
from typing import Protocol
from dataclasses import dataclass, field
from findex.types import Index, Posting

@dataclass(order=True)
class SearchResult:
    score: float = field(compare=True)
    doc_id: int = field(compare=False)
    title: str = field(compare=False)

class Scorer(Protocol):
    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        ...

class TfIdf:
    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        if posting.tf == 0:
            return 0.0

        tf_weight = 1 + math.log(posting.tf)
        df = index.df(term)
        idf_weight = math.log(index.num_docs / df) if df > 0 else 0.0

        return tf_weight * idf_weight

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

def rank_documents(
    query_terms: Iterable[str],
    matching_docs: Set[int],
    index: Index,
    scorer: Scorer,
    top_k: int = 10
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
            title=index.doc_meta[doc_id].title if doc_id in index.doc_meta else f"Doc {doc_id}"
        )
        for doc_id, score in scores.items()
    ]

    return heapq.nlargest(top_k, results)
