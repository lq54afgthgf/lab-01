import pytest
from findex.types import Index, DocMeta, Posting

class MockIndex(Index):
    def __init__(self):
        self.postings = {
            "python": [Posting(doc_id=1, tf=1, positions=(0,)), Posting(doc_id=2, tf=1, positions=(10,))],
            "async": [Posting(doc_id=1, tf=1, positions=(1,))],
            "rust": [Posting(doc_id=3, tf=2, positions=(0, 5))],
            "test": [Posting(doc_id=1, tf=2, positions=(0, 1)), Posting(doc_id=2, tf=1, positions=(5,))]
        }
        self.doc_meta = {
            1: DocMeta(title="Doc 1", length=10, path="mock/1.txt"),
            2: DocMeta(title="Doc 2", length=20, path="mock/2.txt"),
            3: DocMeta(title="Doc 3", length=30, path="mock/3.txt")
        }
        self.doc_lengths = {1: 10, 2: 20, 3: 30}
        self.avg_doc_length = 20.0

    def __contains__(self, term: str) -> bool:
        return term in self.postings

    def __getitem__(self, term: str) -> list[Posting]:
        return self.postings.get(term, [])

    def __len__(self) -> int:
        return len(self.postings)

    def df(self, term: str) -> int:
        return len(self.postings.get(term, []))

@pytest.fixture
def mock_index():
    """Спільна фікстура мок-індексу для всіх тестів."""
    return MockIndex()

@pytest.fixture
def sample_index():
    """Короткий порожній індекс для тестування збереження (store)."""
    idx = Index()
    idx.doc_meta = {1: DocMeta(title="Doc 1", length=10, path="mock1.txt")}
    idx.doc_lengths = {1: 10}
    return idx
