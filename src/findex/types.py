import collections
import collections.abc
import functools
from dataclasses import dataclass
from typing import Iterable, Iterator

@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] | None = None

@dataclass(frozen=True, slots=True)
class DocMeta:
    length: int
    path: str
    title: str = ""

# Наслідуємо Mapping, щоб отримати багато словникових методів безкоштовно (.keys(), .items(), .get())
class Index(collections.abc.Mapping):
    def __init__(self):
        self.postings: dict[str, list[Posting]] = collections.defaultdict(list)
        self.doc_meta: dict[int, DocMeta] = {}
        self.doc_lengths: dict[int, int] = {}

    # --- DUNDER МЕТОДИ (Data Model) ---

    def __len__(self) -> int:
        """Дозволяє викликати len(index) — повертає кількість унікальних термінів."""
        return len(self.postings)

    def __iter__(self) -> Iterator[str]:
        """Дозволяє ітеруватись: for term in index: ..."""
        return iter(self.postings)

    def __getitem__(self, term: str) -> list[Posting]:
        """Дозволяє звертатись як до словника: index['python']. Викидає KeyError, якщо терміну немає."""
        if term not in self.postings:
            raise KeyError(term)
        return self.postings[term]

    def __contains__(self, term: str) -> bool:
        """Дозволяє перевіряти наявність: 'python' in index"""
        return term in self.postings

    def __repr__(self) -> str:
        """Гарне відображення при print(index) або в REPL."""
        return f"Index(terms={len(self)}, docs={self.num_docs})"

    # --- ВЛАСТИВОСТІ (Properties) ---

    @property
    def num_docs(self) -> int:
        """Обчислювана властивість. Виклик без дужок: index.num_docs"""
        return len(self.doc_meta)

    @functools.cached_property
    def avg_doc_length(self) -> float:
        """
        Кешується після першого виклику (вкрай важливо для швидкості алгоритму BM25).
        """
        if not self.num_docs:
            return 0.0
        return sum(self.doc_lengths.values()) / self.num_docs

    def df(self, term: str) -> int:
        """Document Frequency — кількість документів, у яких зустрічається термін."""
        return len(self.postings.get(term, []))

    # Метод з Лаби 2 для побудови
    def add_document(self, doc_id: int, tokens: Iterable[str], path: str, title: str = "", store_positions: bool = False):
        if store_positions:
            term_positions = collections.defaultdict(list)
            for pos, term in enumerate(tokens):
                term_positions[term].append(pos)

            length = sum(len(pos_list) for pos_list in term_positions.values())
            self.doc_lengths[doc_id] = length
            self.doc_meta[doc_id] = DocMeta(length=length, path=path, title=title)

            for term, positions in term_positions.items():
                self.postings[term].append(Posting(doc_id=doc_id, tf=len(positions), positions=tuple(positions)))
        else:
            term_freqs = collections.Counter(tokens)
            length = sum(term_freqs.values())

            self.doc_lengths[doc_id] = length
            self.doc_meta[doc_id] = DocMeta(length=length, path=path, title=title)

            for term, freq in term_freqs.items():
                self.postings[term].append(Posting(doc_id=doc_id, tf=freq, positions=None))
