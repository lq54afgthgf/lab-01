import argparse
import pathlib
import time
import tracemalloc

from findex.types import InvertedIndex
from findex.tokenize import tokenize
from findex.store import save_index
from findex.corpus import iter_documents  # Додаємо імпорт вашого генератора документів

def build_index_cli():
    parser = argparse.ArgumentParser(description="Побудова інвертованого індексу")
    parser.add_argument("data_dir", help="Папка з текстовим корпусом")
    parser.add_argument("--out", required=True, help="Файл для збереження індексу")
    args = parser.parse_args()

    print(f"Починаємо побудову індексу з {args.data_dir}...")
    tracemalloc.start()
    start_time = time.perf_counter()

    index = InvertedIndex()
    data_path = pathlib.Path(args.data_dir)
    doc_id = 0

    # Використовуємо iter_documents для розбиття дампу на 288 тисяч окремих статей
    for doc_text in iter_documents(data_path):
        tokens_stream = tokenize(doc_text)
        # Додаємо кожну статтю як окремий документ
        index.add_document(doc_id, tokens_stream, path="wiki_dump", title=f"Article_{doc_id}")

        doc_id += 1
        if doc_id % 50000 == 0:
            print(f"Проіндексовано {doc_id} статей...")

    elapsed = time.perf_counter() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
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
