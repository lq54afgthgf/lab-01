import argparse
import time
import tracemalloc
from findex.store import load_index
from findex.tokenize import tokenize

def merge_and(list1, list2):
    """
    Класичний алгоритм об'єднання двох відсортованих списків двома вказівниками (O(N+M)).
    Повертає спільні постінги (перетин).
    """
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

def search_merge(index, terms):
    """Пошук (AND) через двовказівниковий зсув."""
    if not terms:
        return []

    # Починаємо з постінгів першого терміна
    result_postings = index.postings.get(terms[0], [])

    # Послідовно перетинаємо з постінгами наступних термінів
    for term in terms[1:]:
        next_postings = index.postings.get(term, [])
        result_postings = merge_and(result_postings, next_postings)
        if not result_postings:
            break  # Якщо перетин порожній, далі шукати немає сенсу

    return [p.doc_id for p in result_postings]

def search_set(index, terms):
    """Пошук (AND) через вбудовані множини (set) Python."""
    if not terms:
        return []

    # Створюємо множину doc_id для першого терміна
    result_docs = {p.doc_id for p in index.postings.get(terms[0], [])}

    # Перетинаємо з множинами наступних термінів
    for term in terms[1:]:
        next_docs = {p.doc_id for p in index.postings.get(term, [])}
        result_docs = result_docs & next_docs
        if not result_docs:
            break

    return list(result_docs)

def search_cli():
    parser = argparse.ArgumentParser(description="Пошук по інвертованому індексу")
    parser.add_argument("index_file", help="Шлях до файлу індексу (наприклад, index.bin)")
    parser.add_argument("query", help="Пошуковий запит")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge", help="Рушій пошуку")
    args = parser.parse_args()

    # Заміряємо пам'ять і час
    tracemalloc.start()
    start_time = time.perf_counter()

    print(f"Завантаження індексу з {args.index_file}...")
    index = load_index(args.index_file)
    load_time = time.perf_counter() - start_time

    # Токенізуємо запит тими самими правилами, що й корпус
    query_terms = list(tokenize(args.query))
    print(f"Запит нормалізовано до: {query_terms}")

    search_start = time.perf_counter()
    if args.engine == "merge":
        doc_ids = search_merge(index, query_terms)
    else:
        doc_ids = search_set(index, query_terms)
    search_time = time.perf_counter() - search_start

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n--- Результати ---")
    if not doc_ids:
        print("Нічого не знайдено.")
    else:
        for d_id in doc_ids:
            meta = index.doc_meta[d_id]
            print(f"Doc {d_id} | Довжина: {meta.length} | Файл: {meta.path}")

    print("\n--- Статистика ---")
    print(f"Час завантаження індексу: {load_time:.4f} сек.")
    print(f"Час пошуку ({args.engine}): {search_time:.6f} сек.")
    print(f"Пікова пам'ять: {peak_mem / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    search_cli()
