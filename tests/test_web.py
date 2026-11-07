from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

from findex.web.app import app, get_index


class DummyIndex:
    def __init__(self) -> None:
        doc = type("Doc", (), {"title": "Test Document", "path": ""})
        self.documents = {1: doc}
        self.doc_meta = {1: doc}
        self.num_docs = 1
        self.doc_lengths = {1: 10}
        self.avg_doc_length = 10.0

    def __len__(self) -> int:
        return 1

    def __contains__(self, term: str) -> bool:
        return term == "test"

    def __getitem__(self, term: str) -> list[object]:
        return [type("Posting", (), {"doc_id": 1, "tf": 1})()] if term == "test" else []

    def df(self, term: str) -> int:
        return 1 if term == "test" else 0


@asynccontextmanager
async def override_lifespan(app_instance):
    app_instance.state.index = DummyIndex()
    yield


app.router.lifespan_context = override_lifespan
app.dependency_overrides[get_index] = lambda: DummyIndex()


def test_health_ok() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_health_503_when_index_not_loaded() -> None:
    with TestClient(app) as client:
        app.state.index = None
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["status"] == "unavailable"
        app.state.index = DummyIndex()


def test_search_success() -> None:
    with TestClient(app) as client:
        resp = client.get("/search?q=test&k=10&scorer=bm25&page=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "test"
        assert data["total"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["doc_id"] == 1


def test_search_validation_error_422() -> None:
    with TestClient(app) as client:
        # Некоректний page=0 та порожній запит q=""
        assert client.get("/search?q=test&page=0").status_code == 422
        assert client.get("/search?q=&k=10").status_code == 422
        assert client.get("/search?q=test&k=500").status_code == 422


def test_document_found_and_not_found_404() -> None:
    with TestClient(app) as client:
        ok_resp = client.get("/docs/1")
        assert ok_resp.status_code == 200
        assert ok_resp.json()["doc_id"] == 1

        not_found_resp = client.get("/docs/999")
        assert not_found_resp.status_code == 404


def test_stats_endpoint() -> None:
    with TestClient(app) as client:
        resp = client.get("/stats")
        assert resp.status_code == 200
        assert resp.json()["doc_count"] == 1
