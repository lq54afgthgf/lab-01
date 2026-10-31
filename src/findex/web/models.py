from pydantic import BaseModel, Field

class SearchResult(BaseModel):
    doc_id: int
    title: str
    score: float
    snippet: str

class SearchResponse(BaseModel):
    query: str
    total: int
    took_ms: float
    results: list[SearchResult]
