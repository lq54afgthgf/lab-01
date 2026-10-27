import pytest
from hypothesis import given, strategies as st
from findex.query import Term, And, Or, Not, parse_query
from findex.types import Index, Posting, DocMeta

# Створюємо заглушку (Mock) індексу для швидкого тестування логіки
class MockIndex(Index):
    def __init__(self):
        self._data = {
            "python": [Posting(doc_id=1, tf=1, positions=(0,)), Posting(doc_id=2, tf=1, positions=(10,))],
            "async": [Posting(doc_id=1, tf=1, positions=(1,))],
            "rust": [Posting(doc_id=3, tf=2, positions=(0, 5))]
        }
        self.doc_meta = {
            1: DocMeta(title="Doc 1", length=10, path="mock/1.txt"),
            2: DocMeta(title="Doc 2", length=20, path="mock/2.txt"),
            3: DocMeta(title="Doc 3", length=30, path="mock/3.txt")
        }
        self.doc_lengths = {1: 10, 2: 20, 3: 30}
        self.avg_doc_length = 20.0
        # Рядки з num_docs та num_tokens видалено, оскільки це properties у базовому класі

    def __contains__(self, term: str) -> bool:
        return term in self._data

    def __getitem__(self, term: str) -> list[Posting]:
        return self._data.get(term, [])

    def __len__(self) -> int:
        return len(self._data)

    def df(self, term: str) -> int:
        return len(self._data.get(term, []))

def test_parse_query_basic():
    """Тестуємо, що парсер правильно будує типи вузлів."""
    node = parse_query("python AND async")
    assert isinstance(node, And)
    assert isinstance(node.left, Term)
    assert node.left.term == "python"

@pytest.mark.parametrize("query_str, expected_docs", [
    ("python", {1, 2}),
    ("rust", {3}),
    ("java", set()),
    ("python AND async", {1}),
    ("python OR rust", {1, 2, 3}),
    ("NOT rust", {1, 2}),
    ("python AND NOT async", {2}),
    ('"python async"', {1}),  # Фразовий пошук
])
def test_evaluate_query(mock_index, query_str, expected_docs):
    """Тестуємо обчислення булевих операцій над множинами документів."""
    node = parse_query(query_str)
    assert node.evaluate(mock_index) == expected_docs

@given(st.text())
def test_parser_fuzzing(query_str):
    """
    Hypothesis: парсер не повинен падати (викидати неконтрольовані винятки)
    при спробі розпарсити будь-який випадковий рядок.
    """
    node = parse_query(query_str)
    assert node is not None
