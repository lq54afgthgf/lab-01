import sys
import time
import tracemalloc

from collections import Counter
from pathlib import Path
from findex.corpus import iter_documents, get_documents_greedy
from findex.tokenize import tokenize, tokenize_greedy

def measure_pipeline(data_dir: Path, is_lazy: bool) -> None:
    tracemalloc.start()
    start_time = time.perf_counter()

    # Явно вказуємо, що Counter рахуватиме рядки (str)
    term_frequencies: Counter[str] = Counter()
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

def main() -> None:
    # Читаємо шлях до папки з аргументів або беремо "data" за замовчуванням
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data"
    data_dir = Path(data_path)

    if not data_dir.exists() or not data_dir.is_dir():
        return

    measure_pipeline(data_dir, is_lazy=True)
    measure_pipeline(data_dir, is_lazy=False)

if __name__ == "__main__":
    main()
