import collections
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] | None = None  # Опційні позиції слова в документі

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

    def add_document(self, doc_id: int, tokens: Iterable[str], path: str, title: str = "", store_positions: bool = False):
        if store_positions:
            # Збираємо терміни разом з їхніми позиціями
            term_positions = collections.defaultdict(list)
            for pos, term in enumerate(tokens):
                term_positions[term].append(pos)

            length = sum(len(pos_list) for pos_list in term_positions.values())
            self.doc_lengths[doc_id] = length
            self.doc_meta[doc_id] = DocMeta(length=length, path=path, title=title)

            for term, positions in term_positions.items():
                self.postings[term].append(Posting(doc_id=doc_id, tf=len(positions), positions=tuple(positions)))
        else:
            # Класичний швидкий підхід через Counter (без позицій)
            term_freqs = collections.Counter(tokens)
            length = sum(term_freqs.values())

            self.doc_lengths[doc_id] = length
            self.doc_meta[doc_id] = DocMeta(length=length, path=path, title=title)

            for term, freq in term_freqs.items():
                self.postings[term].append(Posting(doc_id=doc_id, tf=freq, positions=None))
