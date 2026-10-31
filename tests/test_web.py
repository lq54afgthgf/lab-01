from contextlib import asynccontextmanager
from fastapi.testclient import TestClient
from findex.web.app import app, get_index

# 1. Оновлений фейковий індекс із doc_meta
class DummyIndex:
    def __init__(self):
        doc = type("Doc", (), {"title": "Test", "path": ""})
        self.documents = {1: doc}
        self.doc_meta = {1: doc}  # Додано для сумісності з rank_documents
        self.num_docs = 1
        self.doc_lengths = {1: 10}
        self.avg_doc_length = 10.0

    def __contains__(self, term):
        return term == "test"

    def __getitem__(self, term):
        return [type("Posting", (), {"doc_id": 1, "tf": 1})()] if term == "test" else []

    def df(self, term):
        return 1 if term == "test" else 0

# 2. Перейменована функція (не починається з test_) як асинхронний контекстний менеджер
@asynccontextmanager
async def override_lifespan(app):
    app.state.index = DummyIndex()
    yield

# Підміняємо lifespan та залежність
app.router.lifespan_context = override_lifespan
app.dependency_overrides[get_index] = lambda: DummyIndex()

def test_health():
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

def test_search_success():
    with TestClient(app) as client:
        response = client.get("/search?q=test&k=10")
        assert response.status_code == 200

def test_search_validation_error_422():
    with TestClient(app) as client:
        assert client.get("/search?q=test&page=0").status_code == 422

def test_document_not_found_404():
    with TestClient(app) as client:
        assert client.get("/docs/999").status_code == 404
