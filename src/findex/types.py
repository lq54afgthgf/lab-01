import collections
import collections.abc
import functools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


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


class Index(collections.abc.Mapping[str, list[Posting]]):
    def __init__(self) -> None:
        self.postings: dict[str, list[Posting]] = collections.defaultdict(list)
        self.doc_meta: dict[int, DocMeta] = {}
        self.doc_lengths: dict[int, int] = {}
        self._np_doc_ids: dict[str, npt.NDArray[np.int32]] = {}
        self._np_tfs: dict[str, npt.NDArray[np.int32]] = {}
        self._np_doc_len: npt.NDArray[np.int32] | None = None

    def _invalidate_numpy_cache(self) -> None:
        self._np_doc_ids = {}
        self._np_tfs = {}
        self._np_doc_len = None
        if "avg_doc_length" in self.__dict__:
            del self.__dict__["avg_doc_length"]

    def ensure_numpy_arrays(self) -> None:
        if not hasattr(self, "_np_doc_len") or self._np_doc_len is None:
            self._np_doc_ids = {}
            self._np_tfs = {}
            max_doc_id = max(self.doc_lengths.keys(), default=-1)
            doc_len_arr = np.zeros(max_doc_id + 1, dtype=np.int32)
            for doc_id, length in self.doc_lengths.items():
                doc_len_arr[doc_id] = length
            self._np_doc_len = doc_len_arr

            for term, p_list in self.postings.items():
                self._np_doc_ids[term] = np.array([p.doc_id for p in p_list], dtype=np.int32)
                self._np_tfs[term] = np.array([p.tf for p in p_list], dtype=np.int32)

    @property
    def doc_len_array(self) -> npt.NDArray[np.int32]:
        self.ensure_numpy_arrays()
        assert self._np_doc_len is not None
        return self._np_doc_len

    def get_numpy_postings(self, term: str) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        self.ensure_numpy_arrays()
        return self._np_doc_ids[term], self._np_tfs[term]

    def __len__(self) -> int:
        return len(self.postings)

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __getitem__(self, term: str) -> list[Posting]:
        if term not in self.postings:
            raise KeyError(term)
        return self.postings[term]

    def __contains__(self, term: object) -> bool:
        if not isinstance(term, str):
            return False
        return term in self.postings

    def __repr__(self) -> str:
        return f"Index(terms={len(self)}, docs={self.num_docs})"

    @property
    def num_docs(self) -> int:
        return len(self.doc_meta)

    @functools.cached_property
    def avg_doc_length(self) -> float:
        if not self.num_docs:
            return 0.0
        return sum(self.doc_lengths.values()) / self.num_docs

    def df(self, term: str) -> int:
        if hasattr(self, "_np_doc_ids") and term in self._np_doc_ids:
            return int(self._np_doc_ids[term].shape[0])
        return len(self.postings.get(term, []))

    def add_document(
        self,
        doc_id: int,
        tokens: Iterable[str],
        path: str,
        title: str = "",
        store_positions: bool = False,
    ) -> None:
        self._invalidate_numpy_cache()
        if store_positions:
            term_positions: dict[str, list[int]] = collections.defaultdict(list)
            for pos, term in enumerate(tokens):
                term_positions[term].append(pos)

            length = sum(len(pos_list) for pos_list in term_positions.values())
            self.doc_lengths[doc_id] = length
            self.doc_meta[doc_id] = DocMeta(length=length, path=path, title=title)

            for term, positions in term_positions.items():
                self.postings[term].append(
                    Posting(doc_id=doc_id, tf=len(positions), positions=tuple(positions))
                )
        else:
            term_freqs = collections.Counter(tokens)
            length = sum(term_freqs.values())

            self.doc_lengths[doc_id] = length
            self.doc_meta[doc_id] = DocMeta(length=length, path=path, title=title)

            for term, freq in term_freqs.items():
                self.postings[term].append(Posting(doc_id=doc_id, tf=freq, positions=None))

    def merge(self, other: "Index") -> None:
        self._invalidate_numpy_cache()
        self.doc_meta.update(other.doc_meta)
        self.doc_lengths.update(other.doc_lengths)
        for term, postings in other.postings.items():
            self.postings[term].extend(postings)
            self.postings[term].sort(key=lambda p: p.doc_id)
