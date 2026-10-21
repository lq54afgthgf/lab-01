import tracemalloc
import collections
from dataclasses import dataclass
from array import array
import pathlib
import sys

from findex.tokenize import tokenize
from findex.corpus import iter_documents

# 1. Звичайний Dataclass (кожен об'єкт має свій __dict__)
@dataclass
class PostingRegular:
    doc_id: int
    tf: int

# 2. Dataclass зі slots (без __dict__, фіксований розмір)
@dataclass(slots=True)
class PostingSlots:
    doc_id: int
    tf: int

def run_benchmark(data_path: pathlib.Path, limit: int = 10000):
    print(f"Завантаження перших {limit} статей для бенчмарку...")
    docs_tokens = []
    for i, text in enumerate(iter_documents(data_path)):
        if i >= limit:
            break
        docs_tokens.append(list(tokenize(text)))

    print("Готово. Починаємо заміри пам'яті...\n")

    def measure(build_func, name):
        tracemalloc.start()
        build_func(docs_tokens)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"{name:25} : {peak / (1024 * 1024):.2f} MB")

    def build_regular(docs):
        idx = collections.defaultdict(list)
        for doc_id, tokens in enumerate(docs):
            for term, freq in collections.Counter(tokens).items():
                idx[term].append(PostingRegular(doc_id, freq))
        return idx

    def build_slots(docs):
        idx = collections.defaultdict(list)
        for doc_id, tokens in enumerate(docs):
            for term, freq in collections.Counter(tokens).items():
                idx[term].append(PostingSlots(doc_id, freq))
        return idx

    def build_array(docs):
        # array('I') зберігає 4-байтові unsigned int без накладних витрат на об'єкти Python
        idx = collections.defaultdict(lambda: array('I'))
        for doc_id, tokens in enumerate(docs):
            for term, freq in collections.Counter(tokens).items():
                idx[term].extend((doc_id, freq))
        return idx

    print("-" * 40)
    measure(build_regular, "1. Звичайний dataclass")
    measure(build_slots, "2. Dataclass (slots=True)")
    measure(build_array, "3. array('I')")
    print("-" * 40)

if __name__ == "__main__":
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("data")
    run_benchmark(path)
