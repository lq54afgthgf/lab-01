import pytest
import tempfile

from pathlib import Path
from hypothesis import given, strategies as st
from findex.types import Index, DocMeta, Posting
from findex.store import save_index, load_index
from findex.query import parse_query

# Стратегія для генерації випадкових списків doc_id
doc_ids_strategy = st.lists(st.integers(min_value=1, max_value=1000), unique=True, min_size=1, max_size=50)

def create_mock_index(term1_docs, term2_docs):
    """Допоміжна функція для створення тестового індексу."""
    idx = Index()
    idx.postings = {
        "apple": [Posting(doc_id=d, tf=1, positions=(0,)) for d in sorted(term1_docs)],
        "banana": [Posting(doc_id=d, tf=1, positions=(0,)) for d in sorted(term2_docs)]
    }

    all_docs = set(term1_docs) | set(term2_docs)
    idx.doc_meta = {d: DocMeta(title=f"Doc {d}", length=10, path=f"{d}.txt") for d in all_docs}
    idx.doc_lengths = {d: 10 for d in all_docs}
    idx.avg_doc_length = 10.0
    return idx

@given(doc_ids_strategy, doc_ids_strategy)
def test_property_postings_always_sorted(term1_docs, term2_docs):
    """Властивість 2: постінги завжди відсортовані за doc_id."""
    idx = create_mock_index(term1_docs, term2_docs)

    for term in ["apple", "banana"]:
        postings = idx[term]
        doc_ids = [p.doc_id for p in postings]
        assert doc_ids == sorted(doc_ids), f"Постінги для '{term}' не відсортовані!"

@pytest.mark.slow
@given(doc_ids_strategy, doc_ids_strategy)
def test_property_merge_matches_set_operations(term1_docs, term2_docs):
    """Властивість 3: логічні операції індексу збігаються з операціями set()."""
    idx = create_mock_index(term1_docs, term2_docs)

    set_apple = set(term1_docs)
    set_banana = set(term2_docs)

    node_and = parse_query("apple AND banana")
    assert node_and.evaluate(idx) == (set_apple & set_banana)

    node_or = parse_query("apple OR banana")
    assert node_or.evaluate(idx) == (set_apple | set_banana)

    node_not = parse_query("apple AND NOT banana")
    assert node_not.evaluate(idx) == (set_apple - set_banana)

@pytest.mark.slow
def test_property_load_save_is_idempotent():
    """Властивість 1: load(save(index)) дає ідентичний індекс."""
    idx = create_mock_index([1, 3, 5], [2, 3, 4])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_index.bin"
        save_index(idx, str(path))
        loaded_idx = load_index(str(path))

        assert loaded_idx.num_docs == idx.num_docs
        assert set(loaded_idx.doc_meta.keys()) == set(idx.doc_meta.keys())

        for term in ["apple", "banana"]:
            orig_docs = [p.doc_id for p in idx[term]]
            loaded_docs = [p.doc_id for p in loaded_idx[term]]
            assert orig_docs == loaded_docs
