import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from findex.ranking import BM25, TfIdf, rank_documents
from findex.search import evaluate_query_cached, get_snippet

try:
    from findex.semantic import SemanticIndex, reciprocal_rank_fusion, search_semantic
except ImportError:
    SemanticIndex = None  # type: ignore[assignment]
    reciprocal_rank_fusion = None  # type: ignore[assignment]
    search_semantic = None  # type: ignore[assignment]
from findex.store import open_index
from findex.types import Index
from findex.web.models import (
    DocumentResponse,
    HealthResponse,
    SearchResponse,
    SearchResult,
    StatsResponse,
)
from findex.web.settings import settings

logger = logging.getLogger("findex.web")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Завантаження індексу з {settings.index_path}...")
    try:
        ctx = open_index(str(settings.index_path))
        app.state.index_ctx = ctx
        idx_obj: Index = ctx.__enter__()
        app.state.index = idx_obj
        idx_obj.ensure_numpy_arrays()
        app.state.sem_index = SemanticIndex.load(settings.index_path) if SemanticIndex else None
        logger.info("Індекс успішно завантажено.")
    except Exception as e:
        logger.error(f"Помилка завантаження індексу: {e}")
        app.state.index = None
        app.state.sem_index = None

    yield

    if hasattr(app.state, "index_ctx") and hasattr(app.state.index_ctx, "__exit__"):
        app.state.index_ctx.__exit__(None, None, None)


app = FastAPI(title="Findex API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = str(uuid.uuid4())
    logger.error(f"Unhandled exception [req_id={req_id}]: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Внутрішня помилка сервера.", "request_id": req_id},
    )


def get_index(request: Request) -> Index:
    if not hasattr(request.app.state, "index") or request.app.state.index is None:
        raise HTTPException(status_code=503, detail="Індекс не завантажено або він пошкоджений")
    return request.app.state.index  # type: ignore[no-any-return] # динамічний Starlette State


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse | JSONResponse:
    if not hasattr(request.app.state, "index") or request.app.state.index is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "detail": "Index not loaded"},
        )
    return HealthResponse(status="ok", version="1.0.0")


@app.get("/stats", response_model=StatsResponse)
def stats(index: Index = Depends(get_index)) -> StatsResponse:
    doc_count = index.num_docs if hasattr(index, "num_docs") else 0
    return StatsResponse(doc_count=doc_count, terms=len(index), status="active")


@app.get("/search", response_model=SearchResponse)
def search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=200, description="Пошуковий запит"),
    k: int = Query(10, ge=1, le=100, description="Кількість результатів на сторінку"),
    scorer: Literal["bm25", "tfidf"] = "bm25",
    mode: Literal["keyword", "semantic", "hybrid"] = "keyword",
    page: int = Query(1, ge=1, description="Номер сторінки"),
    index: Index = Depends(get_index),
):
    t0 = time.perf_counter()
    scorer_obj = BM25() if scorer == "bm25" else TfIdf()
    query_words = {w.strip('()"').lower() for w in q.split() if w not in ("AND", "OR", "NOT")}
    top_k_limit = page * k

    kw_ranked = []
    if mode in ("keyword", "hybrid"):
        matching_docs = set(evaluate_query_cached(q, index))
        if not matching_docs and mode == "hybrid":
            matching_docs = set(index.doc_lengths.keys())
        if matching_docs:
            kw_ranked = rank_documents(
                query_words,
                matching_docs,
                index,
                scorer_obj,
                top_k=max(top_k_limit, 30),
            )

    sem_ranked = []
    if mode in ("semantic", "hybrid"):
        sem_idx = getattr(request.app.state, "sem_index", None)
        if sem_idx is None and SemanticIndex is not None:
            sem_idx = SemanticIndex.load(settings.index_path)
            request.app.state.sem_index = sem_idx
        if sem_idx is not None and search_semantic is not None:
            sem_ranked = search_semantic(q, index, sem_idx, top_k=max(top_k_limit, 30))

    if mode == "keyword":
        all_ranked = kw_ranked
    elif mode == "semantic":
        all_ranked = sem_ranked if sem_ranked else kw_ranked
    else:
        if sem_ranked and reciprocal_rank_fusion is not None:
            all_ranked = reciprocal_rank_fusion(
                kw_ranked, sem_ranked, index, k=60, top_k=top_k_limit
            )
        else:
            all_ranked = kw_ranked

    total = len(all_ranked)
    start_idx = (page - 1) * k
    page_results = all_ranked[start_idx : start_idx + k]

    results: list[SearchResult] = []
    meta_dict = getattr(index, "doc_meta", getattr(index, "documents", {}))
    for res in page_results:
        text = ""
        if res.doc_id in meta_dict:
            doc_meta = meta_dict[res.doc_id]
            if hasattr(doc_meta, "path") and doc_meta.path:
                doc_path = Path(doc_meta.path)
                if doc_path.exists() and doc_path.suffix == ".txt":
                    text = doc_path.read_text(encoding="utf-8", errors="ignore")

        snippet = (
            get_snippet(text, query_words, window=40)
            if text
            else getattr(res, "title", f"Doc {res.doc_id}")
        )

        results.append(
            SearchResult(
                doc_id=res.doc_id,
                title=res.title if hasattr(res, "title") else f"Doc {res.doc_id}",
                score=round(res.score, 4),
                snippet=snippet,
            )
        )

    took_ms = (time.perf_counter() - t0) * 1000
    return SearchResponse(query=q, total=total, page=page, k=k, took_ms=took_ms, results=results)


@app.get("/docs/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: int, index: Index = Depends(get_index)) -> DocumentResponse:
    meta_dict = getattr(index, "doc_meta", getattr(index, "documents", {}))
    if doc_id not in meta_dict:
        raise HTTPException(status_code=404, detail="Документ не знайдено")

    doc_meta = meta_dict[doc_id]
    text = getattr(doc_meta, "title", f"Doc {doc_id}")
    if hasattr(doc_meta, "path") and doc_meta.path:
        doc_path = Path(doc_meta.path)
        if doc_path.exists() and doc_path.suffix == ".txt":
            text = doc_path.read_text(encoding="utf-8", errors="ignore")

    return DocumentResponse(
        doc_id=doc_id,
        title=getattr(doc_meta, "title", f"Doc {doc_id}"),
        text=text,
    )


templates = Jinja2Templates(directory="src/findex/web/templates")


@app.get("/", response_class=HTMLResponse)
def index_page(
    request: Request,
    q: str = Query(None, description="Пошуковий запит"),
    mode: Literal["keyword", "semantic", "hybrid"] = Query("keyword"),
    page: int = Query(1, ge=1),
    index: Index = Depends(get_index),
):
    if not q:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"query": "", "mode": mode, "results": None},
        )

    try:
        res = search(request=request, q=q, k=10, scorer="bm25", mode=mode, page=page, index=index)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "query": q,
                "mode": mode,
                "results": res.results,
                "total": res.total,
                "took_ms": round(res.took_ms, 2),
                "page": page,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"query": q, "mode": mode, "results": None, "error": str(e)},
        )
