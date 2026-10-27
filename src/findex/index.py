import argparse
import pathlib

from findex.types import Index
from findex.tokenize import tokenize
from findex.store import save_index
from findex.corpus import iter_documents

def build_index_cli() -> None:
    parser = argparse.ArgumentParser(description="Побудова інвертованого індексу")
    parser.add_argument("data_dir", help="Папка з текстовим корпусом")
    parser.add_argument("--out", required=True, help="Файл для збереження індексу")
    parser.add_argument("--positions", action="store_true", help="Зберігати позиції токенів")
    args = parser.parse_args()

    index = Index()
    data_path = pathlib.Path(args.data_dir)
    doc_id = 0

    # Використовуємо iter_documents для розбиття дампу на окремі статті
    for doc_text in iter_documents(data_path):
        # Перетворюємо генератор на список, щоб надійно передати в індекс
        tokens_list = list(tokenize(doc_text))

        # Викликаємо метод з прапорцем
        index.add_document(
            doc_id=doc_id,
            tokens=tokens_list,
            path="wiki_dump",
            title=f"Article_{doc_id}",
            store_positions=args.positions
        )
        doc_id += 1

    save_index(index, args.out)

if __name__ == "__main__":
    build_index_cli()
