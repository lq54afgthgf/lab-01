import argparse
import time
import functools
import pathlib
import itertools

from findex.store import open_index
from findex.query import parse_query
from findex.ranking import BM25, TfIdf, rank_documents
from findex.corpus import iter_documents

# --- ДЕКОРАТОРИ ---

def timed(fn):
    """Декоратор для заміру часу виконання функції."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            print(f"[{fn.__name__}] виконано за {elapsed * 1000:.2f} ms")
    return wrapper

# Глобальна змінна для доступу до індексу з кешованої функції
_GLOBAL_INDEX = None

@functools.lru_cache(maxsize=256)
@timed
def evaluate_query_cached(query_str: str) -> frozenset[int]:
    """
    Пошук документів за запитом із кешуванням.
    @lru_cache запам'ятає результат, і повторний виклик не буде обчислювати дерево.
    """
    tree = parse_query(query_str)
    # Зберігаємо результат як frozenset, бо звичайний set не кешується (mutable)
    return frozenset(tree.evaluate(_GLOBAL_INDEX))

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

    # Проста підсвітка терміну (ANSI кольори для терміналу)
    # \033[93m - жовтий колір, \033[1m - жирний, \033[0m - скидання
    highlighted = snippet.replace(
        text[best_pos:best_pos+len(matched_term)],
        f"\033[93m\033[1m{text[best_pos:best_pos+len(matched_term)]}\033[0m"
    )

    return f"...{highlighted}..."

# --- ГОЛОВНИЙ СКРИПТ ---

@timed
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_path", help="Шлях до файлу індексу")
    parser.add_argument("corpus_path", help="Шлях до корпусу (для витягування сніпетів)")
    parser.add_argument("query", help="Запит (напр., 'python AND (async OR await)')")
    parser.add_argument("--scorer", choices=["bm25", "tfidf"], default="bm25")
    args = parser.parse_args()

    global _GLOBAL_INDEX

    print(f"Завантаження індексу з {args.index_path}...")
    # Використання Context Manager
    with open_index(args.index_path) as index:
        _GLOBAL_INDEX = index
        print(f"Індекс завантажено. Документів: {index.num_docs}, Термінів: {len(index)}")

        # 1. Знаходимо відповідні документи (перший виклик)
        print("\n--- Перший виклик (має виконати AST-дерево) ---")
        matching_docs = evaluate_query_cached(args.query)

        # 2. Робимо повторний виклик, щоб показати роботу кешу (Cache Hit)
        print("\n--- Другий виклик (має підтягнутись з кешу миттєво) ---")
        matching_docs = evaluate_query_cached(args.query)

        if not matching_docs:
            print("\nНічого не знайдено.")
            return

        # 3. Ранжування
        scorer = BM25() if args.scorer == "bm25" else TfIdf()

        # Витягуємо чисті слова із запиту для ранжування (без AND, OR, NOT)
        query_words = {w.strip('()"').lower() for w in args.query.split() if w not in ("AND", "OR", "NOT")}

        top_results = rank_documents(query_words, matching_docs, index, scorer, top_k=5)

        print(f"\nЗнайдено документів: {len(matching_docs)}. Топ-5 результатів ({args.scorer.upper()}):")

        # 4. Витягуємо оригінальні тексти для сніпетів
        # (Проходимо по корпусу і забираємо тільки тексти переможців)
        top_doc_ids = {res.doc_id for res in top_results}
        doc_texts = {}
        for doc_id, text in enumerate(iter_documents(pathlib.Path(args.corpus_path))):
            if doc_id in top_doc_ids:
                doc_texts[doc_id] = text
            if len(doc_texts) == len(top_doc_ids):
                break

        # 5. Виводимо результати
        for rank, res in enumerate(top_results, 1):
            snippet = get_snippet(doc_texts.get(res.doc_id, ""), query_words)
            print(f"\n{rank}. Doc {res.doc_id} | Title: {res.title} | Score: {res.score:.4f}")
            print(f"   {snippet}")

if __name__ == "__main__":
    main()
