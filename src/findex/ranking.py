import math
import collections
import heapq
from typing import Protocol, Iterable
from dataclasses import dataclass, field

from findex.types import Index, Posting

# order=True автоматично генерує dunder-методи __lt__, __le__, __gt__, __ge__
# field(compare=False) каже, що порівнювати треба ТІЛЬКИ за полем score
@dataclass(order=True)
class SearchResult:
    score: float = field(compare=True)
    doc_id: int = field(compare=False)
    title: str = field(compare=False)

# Протокол (Duck Typing): будь-що, що можна викликати з цими параметрами - є Scorer'ом.
class Scorer(Protocol):
    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        ...

class TfIdf:
    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        if posting.tf == 0:
            return 0.0

        # tf = 1 + log(tf) — щоб 100 повторень не були в 100 разів важливішими за 1
        tf_weight = 1 + math.log(posting.tf)

        df = index.df(term)
        # idf = log(N / df) — чим рідше слово в корпусі, тим воно важливіше
        idf_weight = math.log(index.num_docs / df) if df > 0 else 0.0

        return tf_weight * idf_weight

class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # Контролює "насичення" частоти слова
        self.b = b    # Контролює штраф за довжину документа

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
    matching_docs: set[int],
    index: Index,
    scorer: Scorer,
    top_k: int = 10
) -> list[SearchResult]:
    """
    Накопичує бали для документів і повертає топ-K.
    """
    scores = collections.defaultdict(float)

    for term in query_terms:
        if term not in index:
            continue

        for posting in index[term]:  # Тут працює наш dunder-метод __getitem__
            if posting.doc_id in matching_docs:
                # Тут працює dunder-метод __call__ нашого scorer'а
                scores[posting.doc_id] += scorer(term, posting, index)

    # Беремо топ-K за допомогою купи (це набагато швидше, ніж сортувати весь масив)
    # Зверніть увагу: ми створюємо об'єкти SearchResult
    results = [
        SearchResult(score=score, doc_id=doc_id, title=index.doc_meta[doc_id].title)
        for doc_id, score in scores.items()
    ]

    return heapq.nlargest(top_k, results)
