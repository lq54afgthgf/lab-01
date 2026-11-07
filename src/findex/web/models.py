from pydantic import BaseModel


class SearchResult(BaseModel):
    doc_id: int
    title: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    total: int
    page: int = 1
    k: int = 10
    took_ms: float
    results: list[SearchResult]


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    text: str


class StatsResponse(BaseModel):
    doc_count: int
    terms: int
    status: str


class HealthResponse(BaseModel):
    status: str
    version: str = "0.7.0"
