import argparse
import pathlib
import time
import tracemalloc

from findex.types import Index
from findex.tokenize import tokenize
from findex.store import save_index
from findex.corpus import iter_documents

def build_index_cli():
    parser = argparse.ArgumentParser(description="Побудова інвертованого індексу")
    parser.add_argument("data_dir", help="Папка з текстовим корпусом")
    parser.add_argument("--out", required=True, help="Файл для збереження індексу")
    # Додаємо прапорець сюди, де він і має бути
    parser.add_argument("--positions", action="store_true", help="Зберігати позиції токенів")
    args = parser.parse_args()
    tracemalloc.start()
    start_time = time.perf_counter()

    index = Index()
    data_path = pathlib.Path(args.data_dir)
    doc_id = 0

    # Використовуємо iter_documents для розбиття дампу на окремі статті
    for doc_text in iter_documents(data_path):
        # Перетворюємо генератор на список, щоб надійно передати в індекс
        tokens_list = list(tokenize(doc_text))

        # Викликаємо правильний метод ВЖЕ з прапорцем
        index.add_document(
            doc_id=doc_id,
            tokens=tokens_list,
            path="wiki_dump",
            title=f"Article_{doc_id}",
            store_positions=args.positions
        )

        doc_id += 1
        if doc_id % 50000 == 0:
    elapsed = time.perf_counter() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    save_index(index, args.out)

if __name__ == "__main__":
    build_index_cli()
