import functools

from collections.abc import Callable
from typing import ParamSpec, TypeVar
from findex.query import parse_query
from findex.types import Index

P = ParamSpec('P')
R = TypeVar('R')

# --- ДЕКОРАТОРИ ---

def timed(fn: Callable[P, R]) -> Callable[P, R]:
    """Декоратор (тепер просто пропускає виклик, бо логування видалено)."""
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return fn(*args, **kwargs)
    return wrapper

# Простий ручний кеш замість lru_cache, щоб уникнути проблеми з Hashable Index
_query_cache: dict[str, frozenset[int]] = {}

@timed
def evaluate_query_cached(query_str: str, index: Index) -> frozenset[int]:
    """Пошук документів за запитом із кешуванням."""
    if query_str in _query_cache:
        return _query_cache[query_str]

    tree = parse_query(query_str)
    result = frozenset(tree.evaluate(index))
    _query_cache[query_str] = result
    return result

# --- СНІПЕТИ ---

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
        return text[:window * 2] + "..."

    start = max(0, best_pos - window)
    end = min(len(text), best_pos + len(matched_term) + window)
    snippet = text[start:end].replace('\n', ' ')

    highlighted = snippet.replace(
        text[best_pos:best_pos+len(matched_term)],
        f"\033[93m\033[1m{text[best_pos:best_pos+len(matched_term)]}\033[0m"
    )

    return f"...{highlighted}..."
