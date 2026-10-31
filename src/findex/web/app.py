import uuid
import logging
import time

from typing import Literal
from fastapi import Query
from pathlib import Path
from findex.web.models import SearchResponse, SearchResult
from findex.search import evaluate_query_cached, get_snippet
from findex.ranking import rank_documents, BM25, TfIdf
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from findex.store import open_index
from findex.web.settings import settings
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

logger = logging.getLogger("findex.web")

# 1. Налаштування життєвого циклу (Lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Виконується при старті сервера (завантажуємо індекс у пам'ять)
    logger.info(f"Завантаження індексу з {settings.index_path}...")
    try:
        # Відкриваємо індекс (використовуємо ваш існуючий метод open_index)
        app.state.index = open_index(str(settings.index_path))
        # Якщо open_index працює як context manager (with open_index(...) as idx),
        # то тут зберігаємо сам об'єкт, увійшовши в контекст:
        if hasattr(app.state.index, "__enter__"):
            app.state.index = app.state.index.__enter__()
        logger.info("Індекс успішно завантажено.")
    except Exception as e:
        logger.error(f"Помилка завантаження індексу: {e}")
        app.state.index = None

    yield  # Сервер працює і приймає запити

    # Виконується при зупинці сервера (graceful shutdown)
    if hasattr(app.state, "index") and app.state.index is not None:
        if hasattr(app.state.index, "__exit__"):
            app.state.index.__exit__(None, None, None)
        elif hasattr(app.state.index, "close"):
            app.state.index.close()
        logger.info("Індекс закрито.")

# Ініціалізація FastAPI додатку
app = FastAPI(title="Findex API", lifespan=lifespan)

# 2. Глобальний обробник винятків
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = str(uuid.uuid4())
    # Пишемо повну помилку в лог з унікальним ID
    logger.error(f"Unhandled exception [req_id={req_id}]: {exc}", exc_info=True)
    # Назовні віддаємо лише 500 без деталей (безпека)
    return JSONResponse(
        status_code=500,
        content={"detail": "Внутрішня помилка сервера.", "request_id": req_id}
    )

# 3. Залежність (Dependency) для отримання індексу в маршрутах
def get_index(request: Request):
    if not hasattr(request.app.state, "index") or request.app.state.index is None:
        raise HTTPException(status_code=503, detail="Індекс не завантажено або він пошкоджений")
    return request.app.state.index

# 4. Базові маршрути
@app.get("/health")
def health(request: Request):
    """Перевірка стану сервісу (Health check)."""
    if not hasattr(request.app.state, "index") or request.app.state.index is None:
        return JSONResponse(status_code=503, content={"status": "unavailable", "detail": "Index not loaded"})
    return {"status": "ok"}

@app.get("/stats")
def stats(index = Depends(get_index)):
    """Повертає статистику індексу."""
    # Залежно від вашої структури Index у лаб. 2-3, атрибути можуть відрізнятись.
    # Якщо там є словник документів та лексикон, можна повернути їх довжину:
    try:
        doc_count = len(index.documents) if hasattr(index, "documents") else 0
        return {"doc_count": doc_count, "status": "active"}
    except Exception:
        return {"status": "active", "detail": "Statistics unavailable"}

    # 5. Маршрут пошуку (def замість async def, бо ранжування - це CPU-bound операція)
@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, max_length=200, description="Пошуковий запит"),
    k: int = Query(10, ge=1, le=100, description="Кількість результатів на сторінку"),
    scorer: Literal["bm25", "tfidf"] = "bm25",
    page: int = Query(1, ge=1, description="Номер сторінки"),
    index = Depends(get_index)
):
    t0 = time.perf_counter()

    # Знаходимо всі документи, що відповідають запиту
    matching_docs = set(evaluate_query_cached(q, index))
    total = len(matching_docs)

    # Ранжуємо
    scorer_obj = BM25() if scorer == "bm25" else TfIdf()
    query_words = {w.strip('()"').lower() for w in q.split() if w not in ("AND", "OR", "NOT")}

    # Отримуємо результати із запасом для пагінації
    top_k_limit = page * k
    all_ranked = rank_documents(query_words, matching_docs, index, scorer_obj, top_k=top_k_limit)

    # Зріз для поточної сторінки
    start_idx = (page - 1) * k
    page_results = all_ranked[start_idx : start_idx + k]

    results = []
    for res in page_results:
        text = ""
        # Спроба отримати текст документа для сніпету
        # У вашому CLI (build_partial) ви зберігали path. Витягуємо текст з файлу:
        if hasattr(index, "documents") and res.doc_id in index.documents:
            doc_meta = index.documents[res.doc_id]
            if hasattr(doc_meta, "path") and doc_meta.path:
                doc_path = Path(doc_meta.path)
                if doc_path.exists() and doc_path.suffix == ".txt":
                    text = doc_path.read_text(encoding="utf-8", errors="ignore")
                # Примітка: для jsonl тут потрібна додаткова логіка читання рядка,
                # але для текстових файлів працюватиме одразу.

        snippet = get_snippet(text, query_words, window=40) if text else "Сніпет недоступний"

        results.append(SearchResult(
            doc_id=res.doc_id,
            title=res.title if hasattr(res, "title") else f"Doc {res.doc_id}",
            score=res.score,
            snippet=snippet
        ))

    took_ms = (time.perf_counter() - t0) * 1000

    return SearchResponse(
        query=q,
        total=total,
        took_ms=took_ms,
        results=results
    )

# 6. Маршрут отримання повного документа
@app.get("/docs/{doc_id}")
def get_document(doc_id: int, index = Depends(get_index)):
    if not hasattr(index, "documents") or doc_id not in index.documents:
        raise HTTPException(status_code=404, detail="Документ не знайдено")

    doc_meta = index.documents[doc_id]
    text = "Текст недоступний"

    if hasattr(doc_meta, "path") and doc_meta.path:
        doc_path = Path(doc_meta.path)
        if doc_path.exists() and doc_path.suffix == ".txt":
            text = doc_path.read_text(encoding="utf-8", errors="ignore")

    return {
        "doc_id": doc_id,
        "title": getattr(doc_meta, "title", f"Doc {doc_id}"),
        "text": text
    }

# Налаштовуємо папку з шаблонами
templates = Jinja2Templates(directory="src/findex/web/templates")

@app.get("/", response_class=HTMLResponse)
def index_page(
    request: Request,
    q: str = Query(None, description="Пошуковий запит"),
    page: int = Query(1, ge=1),
    index = Depends(get_index)
):
    """Головна сторінка з пошуковою формою та результатами."""
    if not q:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"query": "", "results": None}
        )

    try:
        res = search(q=q, k=10, scorer="bm25", page=page, index=index)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "query": q,
                "results": res.results,
                "total": res.total,
                "took_ms": res.took_ms,
                "page": page
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"query": q, "results": None, "error": str(e)}
        )
