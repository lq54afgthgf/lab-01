import argparse
import os
import pathlib
import time
import tracemalloc
from itertools import islice

from findex.corpus import iter_documents
from findex.store import save_index
from findex.tokenize import tokenize
from findex.types import Index


def build_index_cli() -> None:
    parser = argparse.ArgumentParser(description="Побудова інвертованого індексу (Лаб 2)")
    parser.add_argument("data_dir", help="Папка або файл із текстовим корпусом")
    parser.add_argument("--out", required=True, help="Файл для збереження індексу")
    parser.add_argument("--positions", action="store_true", help="Зберігати позиції токенів")
    parser.add_argument("--max-docs", type=int, default=None, help="Ліміт документів (опційно)")
    args = parser.parse_args()

    tracemalloc.start()
    t0 = time.perf_counter()

    index = Index()
    data_path = pathlib.Path(args.data_dir)

    stream = iter_documents(data_path)
    if args.max_docs is not None:
        stream = islice(stream, args.max_docs)

    for doc_id, doc_text in enumerate(stream):
        tokens_list = list(tokenize(doc_text))
        index.add_document(
            doc_id=doc_id,
            tokens=tokens_list,
            path=str(data_path),
            title=f"Article_{doc_id}",
            store_positions=args.positions,
        )

    save_index(index, args.out)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    file_size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(
        f"Індекс збережено у '{args.out}' ({file_size_mb:.2f} MB) | "
        f"Документів: {index.num_docs}, Термів: {len(index)} | "
        f"Час: {elapsed:.4f} сек | Пік пам'яті: {peak / 10**6:.3f} MB"
    )


if __name__ == "__main__":
    build_index_cli()
