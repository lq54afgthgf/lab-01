import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path

from findex.corpus import iter_documents, get_documents_greedy
from findex.tokenize import tokenize, tokenize_greedy

def measure_pipeline(data_dir: Path, is_lazy: bool):
    print(f"--- Запуск {'ЛІНИВОГО' if is_lazy else 'ЖАДІБНОГО'} конвеєра ---")
    tracemalloc.start()
    start_time = time.perf_counter()

    term_frequencies = Counter()
    doc_count = 0
    token_count = 0

    if is_lazy:
        # Лінивий: тягнемо по одному документу і по одному токену
        for doc in iter_documents(data_dir):
            doc_count += 1
            for token in tokenize(doc):
                token_count += 1
                term_frequencies[token] += 1
    else:
        # Жадібний: вантажимо всі документи і всі токени списком
        docs = get_documents_greedy(data_dir)
        doc_count = len(docs)
        for doc in docs:
            tokens = tokenize_greedy(doc)
            token_count += len(tokens)
            term_frequencies.update(tokens)

    elapsed_time = time.perf_counter() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Документів: {doc_count} | Токенів: {token_count} | Словник: {len(term_frequencies)}")

    print("Топ-50 термів:")
    for i, (word, freq) in enumerate(term_frequencies.most_common(50), 1):
        print(f"{i}. {word} ({freq})")

    print(f"Час: {elapsed_time:.4f} сек | Пікова пам'ять: {peak_mem / 1024 / 1024:.2f} MB\n")

def main():
    # Читаємо шлях до папки з аргументів або беремо "data" за замовчуванням
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data"
    data_dir = Path(data_path)

    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Помилка: Папки '{data_dir}' не існує або це не папка.")
        print("Створіть папку data/ та покладіть туди текстові файли (.txt).")
        return

    measure_pipeline(data_dir, is_lazy=True)
    measure_pipeline(data_dir, is_lazy=False)

if __name__ == "__main__":
    main()
