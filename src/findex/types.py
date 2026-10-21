import collections
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int

@dataclass(frozen=True, slots=True)
class DocMeta:
    length: int
    path: str
    title: str = ""

class InvertedIndex:
    def __init__(self):
        self.postings: dict[str, list[Posting]] = collections.defaultdict(list)
        self.doc_meta: dict[int, DocMeta] = {}
        self.doc_lengths: dict[int, int] = {}

    def add_document(self, doc_id: int, tokens: Iterable[str], path: str, title: str = ""):
        term_freqs = collections.Counter(tokens)
        length = sum(term_freqs.values())

        self.doc_lengths[doc_id] = length
        self.doc_meta[doc_id] = DocMeta(length=length, path=path, title=title)

        for term, freq in term_freqs.items():
            self.postings[term].append(Posting(doc_id=doc_id, tf=freq))
