from findex.ranking import BM25, TfIdf, rank_documents, rank_documents_python
from findex.types import Index


def _create_fixture_index(num_docs: int = 200) -> Index:
    idx = Index()
    for i in range(num_docs):
        tokens = ["quotes", "to", "scrape"]
        if i % 2 == 0:
            tokens.extend(["life", "love"] * ((i % 5) + 1))
        if i % 3 == 0:
            tokens.extend(["books", "reading", "author"])
        if i == num_docs - 1:
            tokens.append("rareword")
        idx.add_document(i, tokens, path=f"doc_{i}.txt", title=f"Doc {i}")
    idx.ensure_numpy_arrays()
    return idx


def test_numpy_ranking_identical_to_lab3() -> None:
    """Перевіряє, що векторизований NumPy-скорер видає той самий порядок, що й Лаб 3."""
    idx = _create_fixture_index(150)
    docs = set(idx.doc_lengths.keys())

    for scorer in [BM25(), TfIdf()]:
        for query in [
            ["life"],
            ["rareword"],
            ["life", "books", "love"],
            ["quotes", "to"],
        ]:
            res_py = rank_documents_python(query, docs, idx, scorer, top_k=10)
            res_np = rank_documents(query, docs, idx, scorer, top_k=10)
            assert [r.doc_id for r in res_py] == [r.doc_id for r in res_np]
            for r_py, r_np in zip(res_py, res_np, strict=True):
                assert abs(r_py.score - r_np.score) < 1e-9


def test_benchmark_search(benchmark) -> None:
    """Бенчмарк пошуку та ранжування для CI (pytest-benchmark)."""
    idx = _create_fixture_index(300)
    docs = set(idx.doc_lengths.keys())
    scorer = BM25()
    query = ["life", "books", "love"]

    result = benchmark(rank_documents, query, docs, idx, scorer, 10)
    assert len(result) == 10


def test_benchmark_build_index(benchmark) -> None:
    """Бенчмарк побудови індексу для CI (pytest-benchmark)."""

    def build():
        return _create_fixture_index(100)

    idx = benchmark(build)
    assert idx.num_docs == 100


def test_reciprocal_rank_fusion() -> None:
    """Перевіряє коректність злиття рейтингів через RRF (M3)."""
    from findex.ranking import SearchResult
    from findex.semantic import reciprocal_rank_fusion

    idx = _create_fixture_index(10)
    kw = [
        SearchResult(score=5.0, doc_id=1, title="Doc 1"),
        SearchResult(score=3.0, doc_id=2, title="Doc 2"),
    ]
    sem = [
        SearchResult(score=0.9, doc_id=2, title="Doc 2"),
        SearchResult(score=0.8, doc_id=3, title="Doc 3"),
    ]
    fused = reciprocal_rank_fusion(kw, sem, idx, k=60, top_k=3)
    # Doc 2 присутній в обох списках, тому має бути на 1-му місці
    assert fused[0].doc_id == 2
    assert {r.doc_id for r in fused} == {1, 2, 3}
