import argparse
import functools
import os
import re
import time
import tracemalloc
from collections.abc import Callable
from typing import Literal, ParamSpec, TypeVar

from findex.query import evaluate_boolean_query, parse_query
from findex.ranking import BM25, TfIdf, rank_documents
from findex.store import open_index
from findex.types import Index

P = ParamSpec("P")
R = TypeVar("R")


TIMED_VERBOSE = os.environ.get("FINDEX_TIMED", "0") == "1"


def timed(fn: Callable[P, R]) -> Callable[P, R]:
    """Декоратор із functools.wraps для заміру часу виконання функцій."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if TIMED_VERBOSE:
            print(f"[@timed] {fn.__name__} виконано за {elapsed_ms:.4f} ms")
        return result

    return wrapper


_query_cache: dict[str, frozenset[int]] = {}
_cache_hits = 0
_cache_misses = 0


@timed
def evaluate_query_cached(
    query_str: str, index: Index, engine: Literal["merge", "set"] = "set"
) -> frozenset[int]:
    """Пошук документів за запитом із кешуванням та логуванням cache hit/miss."""
    global _cache_hits, _cache_misses
    cache_key = f"{engine}:{query_str}"
    if cache_key in _query_cache:
        _cache_hits += 1
        if TIMED_VERBOSE:
            print(
                f"[CACHE HIT] Запит '{query_str}' взято з кешу (hits={_cache_hits}, misses={_cache_misses})"
            )
        return _query_cache[cache_key]

    _cache_misses += 1
    if TIMED_VERBOSE:
        print(
            f"[CACHE MISS] Обчислення запиту '{query_str}' (hits={_cache_hits}, misses={_cache_misses})"
        )

    if engine == "merge":
        result = frozenset(evaluate_boolean_query(query_str, index, engine="merge"))
    else:
        tree = parse_query(query_str)
        result = frozenset(tree.evaluate(index))

    _query_cache[cache_key] = result
    return result


def get_snippet(text: str, query_terms: set[str], window: int = 80) -> str:
    """Знаходить перше входження слова із запиту і повертає ±80 символів навколо нього."""
    text_lower = text.lower()
    best_pos = -1
    matched_term = ""

    for term in query_terms:
        pos = text_lower.find(term)
        if pos != -1:
            best_pos = pos
            matched_term = term
            break

    if best_pos == -1:
        return text[: window * 2] + "..."

    start = max(0, best_pos - window)
    end = min(len(text), best_pos + len(matched_term) + window)
    snippet = text[start:end].replace("\n", " ")

    highlighted = snippet.replace(
        text[best_pos : best_pos + len(matched_term)],
        f"\033[93m\033[1m{text[best_pos : best_pos + len(matched_term)]}\033[0m",
    )

    return f"...{highlighted}..."


def search_cli() -> None:
    parser = argparse.ArgumentParser(description="Пошук за індексом (Лаб 2 та Лаб 3)")
    parser.add_argument("index_path", help="Шлях до файлу індексу")
    parser.add_argument("query", help="Пошуковий запит (AND, OR, NOT, дужки, фрази)")
    parser.add_argument(
        "--engine",
        choices=["merge", "set"],
        default="merge",
        help="Алгоритм перетину постінгів: двовказівниковий merge або set",
    )
    parser.add_argument(
        "--scorer",
        choices=["bm25", "tfidf"],
        default=None,
        help="Модель ранжування (Лаб 3): bm25 або tfidf",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Кількість топових результатів для виводу",
    )
    args = parser.parse_args()

    tracemalloc.start()
    t0 = time.perf_counter()

    with open_index(args.index_path) as idx:
        matched_set = evaluate_query_cached(args.query, idx, engine=args.engine)

        if args.scorer is not None:
            scorer_obj = BM25() if args.scorer == "bm25" else TfIdf()
            raw_terms = [
                t.lower()
                for t in re.findall(r"[\w']+", args.query)
                if t.upper() not in ("AND", "OR", "NOT")
            ]
            ranked = rank_documents(
                raw_terms, matched_set, idx, scorer=scorer_obj, top_k=args.top_k
            )
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(
                f"[Engine: {args.engine} | Scorer: {args.scorer.upper()}] "
                f"Знайдено: {len(matched_set)} (показано топ-{len(ranked)}) | "
                f"Час: {elapsed * 1000:.2f} мс | Пік пам'яті: {peak / 10**6:.3f} MB"
            )
            for rank_i, res in enumerate(ranked, 1):
                meta = idx.doc_meta.get(res.doc_id)
                doc_len = meta.length if meta else 0
                print(
                    f" {rank_i}. doc_id={res.doc_id}: {res.title} "
                    f"(score={res.score:.4f}, len={doc_len})"
                )
        else:
            matched = sorted(matched_set)
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(
                f"[Engine: {args.engine}] Знайдено документів: {len(matched)} | "
                f"Час: {elapsed * 1000:.2f} мс | Пік пам'яті: {peak / 10**6:.3f} MB"
            )
            for doc_id in matched[: args.top_k]:
                meta = idx.doc_meta.get(doc_id)
                title = meta.title if meta else f"Doc {doc_id}"
                print(f" - doc_id={doc_id}: {title} (len={meta.length if meta else 0})")


if __name__ == "__main__":
    search_cli()
