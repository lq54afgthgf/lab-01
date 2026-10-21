import argparse
import pathlib
import time
import tracemalloc

# Імпортуємо сам індекс із types.py
from findex.types import InvertedIndex
from findex.tokenize import tokenize
from findex.store import save_index

def build_index_cli():
    parser = argparse.ArgumentParser(description="Побудова інвертованого індексу")
    parser.add_argument("data_dir", help="Папка з текстовим корпусом")
    parser.add_argument("--out", required=True, help="Файл для збереження індексу (наприклад, index.bin)")
    args = parser.parse_args()

    print(f"Починаємо побудову індексу з {args.data_dir}...")

    # Вмикаємо відстеження пам'яті
    tracemalloc.start()
    start_time = time.perf_counter()

    index = InvertedIndex()
    data_path = pathlib.Path(args.data_dir)
    doc_id = 0

    # Обходимо всі txt файли в папці
    for file_path in data_path.glob("*.txt"):
        # Відкриваємо кожен файл самостійно
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Створюємо лінивий потік токенів прямо з рядків цього файлу
            tokens_stream = (token for line in f if line.strip() for token in tokenize(line))

            # Передаємо цей потік прямо в метод індексу
            index.add_document(doc_id, tokens_stream, str(file_path), file_path.name)

        doc_id += 1

    # Зупиняємо таймер і заміряємо пам'ять
    elapsed = time.perf_counter() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Індекс побудовано за {elapsed:.2f} сек.")
    print(f"Пікова пам'ять: {peak_mem / (1024 * 1024):.2f} MB")
    print(f"Проіндексовано документів: {len(index.doc_meta)}")
    print(f"Унікальних термів у словнику: {len(index.postings)}")

    print(f"Збереження у {args.out}...")
    save_index(index, args.out)
    print("Індекс успішно збережено!")

if __name__ == "__main__":
    build_index_cli()
