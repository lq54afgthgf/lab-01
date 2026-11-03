import argparse
import functools
import time
import tracemalloc
from collections.abc import Callable
from typing import Literal, ParamSpec, TypeVar

from findex.query import evaluate_boolean_query, parse_query
from findex.store import open_index
from findex.types import Index

P = ParamSpec("P")
R = TypeVar("R")


def timed(fn: Callable[P, R]) -> Callable[P, R]:
    """Декоратор (пропускає виклик без зайвого логування)."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return fn(*args, **kwargs)

    return wrapper


_query_cache: dict[str, frozenset[int]] = {}


@timed
def evaluate_query_cached(
    query_str: str, index: Index, engine: Literal["merge", "set"] = "set"
) -> frozenset[int]:
    """Пошук документів за запитом із кешуванням та вибором рушія (merge/set)."""
    cache_key = f"{engine}:{query_str}"
    if cache_key in _query_cache:
        return _query_cache[cache_key]

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
    parser = argparse.ArgumentParser(description="Булевий пошук за індексом (Лаб 2)")
    parser.add_argument("index_path", help="Шлях до файлу індексу")
    parser.add_argument("query", help="Пошуковий запит (AND, OR, NOT)")
    parser.add_argument(
        "--engine",
        choices=["merge", "set"],
        default="merge",
        help="Алгоритм перетину постінгів: двовказівниковий merge або set",
    )
    args = parser.parse_args()

    tracemalloc.start()
    t0 = time.perf_counter()

    with open_index(args.index_path) as idx:
        matched = sorted(evaluate_boolean_query(args.query, idx, engine=args.engine))

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(
        f"[Engine: {args.engine}] Знайдено документів: {len(matched)} | "
        f"Час: {elapsed * 1000:.2f} мс | Пік пам'яті: {peak / 10**6:.3f} MB"
    )
    for doc_id in matched[:10]:
        meta = idx.doc_meta.get(doc_id)
        title = meta.title if meta else f"Doc {doc_id}"
        print(f" - doc_id={doc_id}: {title} (len={meta.length if meta else 0})")


if __name__ == "__main__":
    search_cli()
