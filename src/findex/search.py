import argparse
import time
import tracemalloc
from findex.store import load_index
from findex.tokenize import tokenize

# ДВОВКАЗІВНИКОВІ АЛГОРИТМИ (O(N+M))

def merge_and(list1, list2):
    """Перетин (AND): зберігаємо спільні елементи."""
    result = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i].doc_id == list2[j].doc_id:
            result.append(list1[i])
            i += 1
            j += 1
        elif list1[i].doc_id < list2[j].doc_id:
            i += 1
        else:
            j += 1
    return result

def merge_or(list1, list2):
    """Об'єднання (OR): зберігаємо елементи з обох списків без дублікатів."""
    result = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i].doc_id == list2[j].doc_id:
            result.append(list1[i])
            i += 1
            j += 1
        elif list1[i].doc_id < list2[j].doc_id:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    # Додаємо залишки
    result.extend(list1[i:])
    result.extend(list2[j:])
    return result

def merge_not(list1, list2):
    """Різниця (NOT): елементи з першого списку, яких немає в другому."""
    result = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i].doc_id == list2[j].doc_id:
            i += 1
            j += 1
        elif list1[i].doc_id < list2[j].doc_id:
            result.append(list1[i])
            i += 1
        else:
            j += 1
    # Додаємо залишки з першого списку
    result.extend(list1[i:])
    return result

def search_merge(index, terms, mode="AND"):
    """Пошук через двовказівниковий зсув."""
    if not terms:
        return []

    result_postings = index.postings.get(terms[0], [])

    for term in terms[1:]:
        next_postings = index.postings.get(term, [])
        if mode == "AND":
            result_postings = merge_and(result_postings, next_postings)
            if not result_postings: break
        elif mode == "OR":
            result_postings = merge_or(result_postings, next_postings)
        elif mode == "NOT":
            result_postings = merge_not(result_postings, next_postings)
            if not result_postings: break

    return [p.doc_id for p in result_postings]

# ПОШУК ЧЕРЕЗ МНОЖИНИ (SET)

def search_set(index, terms, mode="AND"):
    """Пошук через вбудовані множини Python."""
    if not terms:
        return []

    result_docs = {p.doc_id for p in index.postings.get(terms[0], [])}

    for term in terms[1:]:
        next_docs = {p.doc_id for p in index.postings.get(term, [])}
        if mode == "AND":
            result_docs &= next_docs
            if not result_docs: break
        elif mode == "OR":
            result_docs |= next_docs
        elif mode == "NOT":
            result_docs -= next_docs
            if not result_docs: break

    return list(result_docs)

# CLI ІНТЕРФЕЙС

def search_cli():
    parser = argparse.ArgumentParser(description="Булевий пошук по інвертованому індексу")
    parser.add_argument("index_file", help="Шлях до файлу індексу (наприклад, index.bin)")
    parser.add_argument("query", help="Пошуковий запит (слова через пробіл)")
    parser.add_argument("--mode", choices=["AND", "OR", "NOT"], default="AND", help="Логічний оператор для слів запиту")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge", help="Рушій пошуку")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    print(f"Завантаження індексу з {args.index_file}...")
    index = load_index(args.index_file)
    load_time = time.perf_counter() - start_time

    query_terms = list(tokenize(args.query))
    print(f"Запит нормалізовано до: {query_terms} (Режим: {args.mode})")

    search_start = time.perf_counter()
    if args.engine == "merge":
        doc_ids = search_merge(index, query_terms, args.mode)
    else:
        doc_ids = search_set(index, query_terms, args.mode)
    search_time = time.perf_counter() - search_start

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n--- Результати ---")
    if not doc_ids:
        print("Нічого не знайдено.")
    else:
        print(f"Знайдено документів: {len(doc_ids)}")
        # Виводимо перші 5 для лаконічності
        for d_id in doc_ids[:5]:
            meta = index.doc_meta[d_id]
            print(f"Doc {d_id} | Довжина: {meta.length} | Файл: {meta.path}")
        if len(doc_ids) > 5:
            print("...")

    print("\n--- Статистика ---")
    print(f"Час завантаження індексу: {load_time:.4f} сек.")
    print(f"Час пошуку ({args.engine}): {search_time:.6f} сек.")
    print(f"Пікова пам'ять: {peak_mem / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    search_cli()
