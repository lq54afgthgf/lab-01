import asyncio
import concurrent.futures
import json
import logging
import math
import multiprocessing
import sys
import time
import tracemalloc
import urllib.parse
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

from findex.corpus import iter_documents
from findex.ranking import BM25, TfIdf, rank_documents
from findex.search import evaluate_query_cached, get_snippet
from findex.semantic import (
    SemanticIndex,
    build_embeddings,
    reciprocal_rank_fusion,
    search_semantic,
)
from findex.store import open_index, save_index
from findex.tokenize import tokenize
from findex.types import Index


class ExecutorType(str, Enum):
    serial = "serial"
    threads = "threads"
    processes = "processes"


app = typer.Typer(help="Findex: локальний рушій пошуку", add_completion=False)
console = Console(stderr=True)
logger = logging.getLogger("findex")


def setup_logging(verbose: int) -> None:
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option("-v", "--verbose", count=True, help="Рівень деталізації логів (до -vv)"),
    ] = 0,
) -> None:
    setup_logging(verbose)


def build_partial(chunk_paths: list[Path], start_doc_id: int) -> Index:
    partial_idx = Index()
    for offset, path in enumerate(chunk_paths):
        if path.is_file() and path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    text = record.get("text", "")
                    doc_id = start_doc_id + idx
                    tokens = list(tokenize(text))
                    title = record.get("title", f"Doc {doc_id}")
                    partial_idx.add_document(
                        doc_id=doc_id, tokens=tokens, path=str(path), title=title
                    )
        else:
            doc_id = start_doc_id + offset
            text = path.read_text(encoding="utf-8", errors="ignore")
            tokens = list(tokenize(text))
            partial_idx.add_document(
                doc_id=doc_id, tokens=tokens, path=str(path), title=f"Doc {doc_id}"
            )

    return partial_idx


@app.command()
def stats(
    corpus_path: Annotated[Path, typer.Argument(help="Шлях до папки або файлу корпусу")],
    max_docs: Annotated[
        int | None,
        typer.Option("--max-docs", help="Ліміт документів для швидкого заміру"),
    ] = None,
) -> None:
    """Виводить статистику корпусу (документи, токени, словник, Топ-50, час та пам'ять)."""
    from itertools import islice

    if not corpus_path.exists():
        console.print(f"[red]Помилка: Шляху '{corpus_path}' не існує.[/red]")
        raise typer.Exit(1)

    # --- 1. ЛІНИВИЙ ПІДХІД (Generator — за один прохід) ---
    console.print("[bold green]Обробка текстів (ліниве читання - generator)...[/bold green]")
    tracemalloc.start()
    t0_lazy = time.perf_counter()

    lazy_word_counts: Counter[str] = Counter()
    lazy_docs = 0
    lazy_tokens = 0

    doc_stream = iter_documents(corpus_path)
    if max_docs is not None:
        doc_stream = islice(doc_stream, max_docs)

    for doc_text in doc_stream:
        lazy_docs += 1
        for tok in tokenize(doc_text):
            lazy_tokens += 1
            lazy_word_counts[tok] += 1

    lazy_time = time.perf_counter() - t0_lazy
    lazy_current, lazy_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    vocab_size = len(lazy_word_counts)

    # --- 2. ЖАДІБНИЙ ПІДХІД (List) ---
    console.print("[bold red]Обробка текстів (жадібне читання - list)...[/bold red]")
    tracemalloc.start()
    t0_greedy = time.perf_counter()

    raw_stream = iter_documents(corpus_path)
    if max_docs is not None:
        raw_stream = islice(raw_stream, max_docs)

    all_docs = list(raw_stream)
    greedy_docs = len(all_docs)
    greedy_word_counts: Counter[str] = Counter()
    greedy_tokens = 0

    for doc_text in all_docs:
        toks = list(tokenize(doc_text))
        greedy_tokens += len(toks)
        greedy_word_counts.update(toks)

    greedy_time = time.perf_counter() - t0_greedy
    greedy_current, greedy_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    del all_docs
    del greedy_word_counts

    # --- ВИВІД 1: Зведена статистика корпусу (Лаб 1) ---
    summary_table = Table(title="Загальна статистика корпусу (за один прохід)")
    summary_table.add_column("Метрика", style="cyan")
    summary_table.add_column("Значення", justify="right", style="bold green")
    summary_table.add_row("Кількість документів", f"{lazy_docs:,}")
    summary_table.add_row("Загальна кількість токенів", f"{lazy_tokens:,}")
    summary_table.add_row("Розмір словника (унікальних термів)", f"{vocab_size:,}")
    summary_table.add_row("Час виконання (Лінивий прохід)", f"{lazy_time:.4f} сек")
    summary_table.add_row("Пікова пам'ять (Лінивий прохід)", f"{lazy_peak / 10**6:.3f} MB")
    console.print(summary_table)

    # --- ВИВІД 2: Порівняння Часу та Пам'яті (Жадібний vs Лінивий) ---
    console.print("\n[bold yellow]Порівняння часу та пам'яті (tracemalloc):[/bold yellow]")
    mem_table = Table(show_header=True, header_style="bold white")
    mem_table.add_column("Підхід читання")
    mem_table.add_column("Документів", justify="right")
    mem_table.add_column("Токенів", justify="right")
    mem_table.add_column("Словник", justify="right")
    mem_table.add_column("Час (сек)", justify="right", style="bold cyan")
    mem_table.add_column("Поточна пам'ять", justify="right")
    mem_table.add_column("Пікова пам'ять", justify="right", style="bold red")

    mem_table.add_row(
        "Жадібний (List)",
        str(greedy_docs),
        str(greedy_tokens),
        str(vocab_size),
        f"{greedy_time:.4f} s",
        f"{greedy_current / 10**6:.3f} MB",
        f"{greedy_peak / 10**6:.3f} MB",
    )
    mem_table.add_row(
        "Лінивий (Generator)",
        str(lazy_docs),
        str(lazy_tokens),
        str(vocab_size),
        f"{lazy_time:.4f} s",
        f"{lazy_current / 10**6:.3f} MB",
        f"{lazy_peak / 10**6:.3f} MB",
    )
    console.print(mem_table)

    # --- ВИВІД 3: Таблиця Топ-50 слів ---
    table_words = Table(title="Топ 50 найчастіших термів")
    table_words.add_column("#", justify="right", style="dim")
    table_words.add_column("Терм", style="cyan")
    table_words.add_column("Частота", justify="right", style="magenta")

    for rank, (word, count) in enumerate(lazy_word_counts.most_common(50), 1):
        table_words.add_row(str(rank), word, str(count))

    console.print(table_words)


@app.command()
def index(
    corpus_path: Annotated[Path, typer.Argument(help="Шлях до папки з текстовими файлами")],
    index_path: Annotated[Path, typer.Argument(help="Шлях для збереження індексу (*.bin)")],
    workers: Annotated[int, typer.Option("--workers", help="Кількість воркерів")] = 1,
    executor: Annotated[
        ExecutorType, typer.Option("--executor", help="Тип екзекутора")
    ] = ExecutorType.processes,
) -> None:
    """Будує інвертований індекс із вказаного корпусу та зберігає його на диск."""
    if not corpus_path.exists():
        console.print(f"[red]Помилка: Шляху '{corpus_path}' не існує.[/red]")
        raise typer.Exit(1)

    if corpus_path.is_file() and corpus_path.suffix == ".jsonl":
        all_paths = [corpus_path]
    elif corpus_path.is_dir():
        all_paths = list(corpus_path.rglob("*.txt"))
    else:
        console.print(f"[red]Помилка: '{corpus_path}' не є ні папкою, ні .jsonl файлом.[/red]")
        raise typer.Exit(1)

    chunk_size = math.ceil(len(all_paths) / workers) if workers > 0 else len(all_paths)
    if chunk_size == 0:
        chunk_size = 1
    chunks = [all_paths[i : i + chunk_size] for i in range(0, len(all_paths), chunk_size)]

    t0 = time.perf_counter()
    partials: list[Index] = []

    if executor == ExecutorType.serial:
        start_id = 0
        for chunk in chunks:
            partials.append(build_partial(chunk, start_doc_id=start_id))
            start_id += len(chunk)
    else:
        pool_ctx: concurrent.futures.Executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=workers)
            if executor == ExecutorType.threads
            else concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            )
        )
        with pool_ctx as pool:
            futures: list[concurrent.futures.Future[Index]] = []
            start_id = 0
            for chunk in chunks:
                futures.append(pool.submit(build_partial, chunk, start_id))
                start_id += len(chunk)

            for fut in concurrent.futures.as_completed(futures):
                partials.append(fut.result())

    t1 = time.perf_counter()
    final_index = Index()
    for p in partials:
        final_index.merge(p)
    final_index.ensure_numpy_arrays()
    t2 = time.perf_counter()

    console.print(
        f"[bold yellow]Час збирання: {t1 - t0:.2f} сек. | Час merge: {t2 - t1:.2f} сек.[/bold yellow]"
    )
    save_index(final_index, str(index_path))
    console.print("[bold green]Індекс успішно збережено![/bold green]")


@app.command()
def embed(
    index_path: Annotated[Path, typer.Argument(help="Шлях до файлу індексу (*.bin)")],
    model: Annotated[
        str, typer.Option("--model", help="Назва моделі sentence-transformers")
    ] = "all-MiniLM-L6-v2",
    corpus_path: Annotated[
        Path | None, typer.Option("--corpus", help="Опційний шлях до папки корпусу")
    ] = None,
) -> None:
    """Генерує L2-нормалізовані ембединги фрагментів документів та зберігає у .npy."""
    with open_index(str(index_path)) as idx:
        sem_idx = build_embeddings(idx, corpus_dir=corpus_path, model_name=model)
        emb_file, map_file = sem_idx.save(index_path)
        console.print(
            f"[bold green]Ембединги збережено![/bold green] Форма: {sem_idx.embeddings.shape}\n"
            f"Матриця: {emb_file}\nМапінг: {map_file}"
        )


@app.command()
def search(
    index_path: Annotated[Path, typer.Argument(help="Шлях до файлу індексу")],
    corpus_path: Annotated[Path, typer.Argument(help="Шлях до корпусу (для витягування сніпетів)")],
    query: Annotated[str, typer.Argument(help="Запит")],
    scorer_name: Annotated[
        Literal["bm25", "tfidf"], typer.Option("--scorer", help="Алгоритм ранжування")
    ] = "bm25",
    mode: Annotated[
        Literal["keyword", "semantic", "hybrid"],
        typer.Option("--mode", help="Режим пошуку: keyword, semantic або hybrid"),
    ] = "keyword",
    output_json: Annotated[
        bool, typer.Option("--json", help="Вивести результати у форматі JSON в stdout")
    ] = False,
) -> None:
    """Виконує пошук за індексом (keyword / semantic / hybrid) та виводить результати."""
    try:
        with open_index(str(index_path)) as idx:
            scorer = BM25() if scorer_name == "bm25" else TfIdf()
            query_words = {
                w.strip('()"').lower() for w in query.split() if w not in ("AND", "OR", "NOT")
            }

            kw_results = []
            if mode in ("keyword", "hybrid"):
                matching_docs = set(evaluate_query_cached(query, idx))
                if not matching_docs and mode == "hybrid":
                    matching_docs = set(idx.doc_lengths.keys())
                if matching_docs:
                    kw_results = rank_documents(query_words, matching_docs, idx, scorer, top_k=20)

            sem_results = []
            if mode in ("semantic", "hybrid"):
                sem_idx = SemanticIndex.load(index_path)
                if sem_idx is None:
                    console.print(
                        "[red]Ембединги не знайдено. Спочатку запустіть `findex embed`.[/red]"
                    )
                    raise typer.Exit(1)
                sem_results = search_semantic(query, idx, sem_idx, top_k=20)

            if mode == "keyword":
                top_results = kw_results[:5]
            elif mode == "semantic":
                top_results = sem_results[:5]
            else:
                top_results = reciprocal_rank_fusion(kw_results, sem_results, idx, k=60, top_k=5)

            if not top_results:
                if output_json:
                    print(json.dumps([]))
                else:
                    console.print("[yellow]Нічого не знайдено.[/yellow]")
                return

            top_doc_ids = {res.doc_id for res in top_results}
            doc_texts: dict[int, str] = {}
            if corpus_path.exists():
                for doc_id, text in enumerate(iter_documents(corpus_path)):
                    if doc_id in top_doc_ids:
                        doc_texts[doc_id] = text

            if output_json:
                results_data = [
                    {
                        "rank": r,
                        "doc_id": res.doc_id,
                        "title": res.title,
                        "score": res.score,
                    }
                    for r, res in enumerate(top_results, 1)
                ]
                print(json.dumps(results_data, ensure_ascii=False, indent=2))
            else:
                table = Table(
                    title=f"Результати пошуку (mode={mode}, scorer={scorer_name.upper()})"
                )
                table.add_column("Ранг", justify="right", style="cyan", no_wrap=True)
                table.add_column("Документ", style="magenta")
                table.add_column("Оцінка", justify="right", style="green")
                table.add_column("Сніпет", overflow="fold")

                for rank, res in enumerate(top_results, 1):
                    snippet = get_snippet(
                        doc_texts.get(res.doc_id, res.title), query_words, window=40
                    )
                    table.add_row(
                        str(rank),
                        f"{res.doc_id}: {res.title}",
                        f"{res.score:.4f}",
                        snippet,
                    )

                console.print(table)

    except FileNotFoundError:
        console.print(f"[red]Помилка: Файл індексу '{index_path}' не знайдено.[/red]")
        raise typer.Exit(1)


@app.command(name="crawl")
def crawl_cmd(
    seed: str = typer.Argument(..., help="Початковий URL для краулінгу"),
    max_pages: int = typer.Option(200, help="Максимальна кількість сторінок"),
    concurrency: int = typer.Option(10, help="Кількість одночасних воркерів"),
    per_host: int = typer.Option(2, help="Ліміт запитів на один хост"),
    out: Path = typer.Option(Path("data/crawl.jsonl"), help="Шлях до вихідного JSONL файлу"),
    allowed_domain: str = typer.Option(None, help="Додатковий дозволений домен"),
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    parsed_seed = urllib.parse.urlparse(seed)
    domains = [parsed_seed.netloc]
    if allowed_domain:
        domains.append(allowed_domain)

    async def run_crawler() -> None:
        fetched_count = 0
        error_count = 0
        start_time = time.monotonic()
        log_path = Path("crawl.log")

        def generate_table(q_depth: int = 0, active_reqs: int = 0) -> Table:
            elapsed = max(0.1, time.monotonic() - start_time)
            pages_per_sec = fetched_count / elapsed
            table = Table(title="[bold cyan]Findex Async Crawler Live Status[/bold cyan]")
            table.add_column("Метрика", style="cyan", no_wrap=True)
            table.add_column("Значення", style="magenta")
            table.add_row("Зібрано сторінок", str(fetched_count))
            table.add_row("Сторінок / сек", f"{pages_per_sec:.2f}")
            table.add_row("Активних запитів", str(active_reqs))
            table.add_row("Глибина черги", str(q_depth))
            table.add_row("Помилок / збоїв", str(error_count))
            return table

        with (
            open(out, "w", encoding="utf-8") as jsonl_f,
            open(log_path, "w", encoding="utf-8") as log_f,
            Live(generate_table(), refresh_per_second=4) as live,
        ):
            from findex.crawl import crawl

            async for page in crawl(
                [seed],
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                allowed_domains=domains,
            ):
                fetched_count += 1
                if page.error:
                    error_count += 1
                if page.status == 200 and page.text:
                    record = {"url": page.url, "title": page.title, "text": page.text}
                    jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_f.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(page.fetched_at))}] Status: {page.status} | URL: {page.url} | Error: {page.error}\n"
                )
                log_f.flush()
                live.update(generate_table(page.queue_depth, page.active_tasks))

    tracemalloc.start()
    asyncio.run(run_crawler())
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[bold yellow]Peak RSS: {peak / 10**6:.2f} MB[/bold yellow]")


@app.command(name="serve")
def serve(
    port: int = typer.Option(8000, help="Порт для веб-сервера"),
    workers: int = typer.Option(1, help="Кількість воркерів Uvicorn"),
) -> None:
    """Запускає веб-інтерфейс та API пошуку."""
    import uvicorn

    console.print(
        f"[bold green]Запуск веб-сервера на порту {port} ({workers} воркерів)...[/bold green]"
    )
    uvicorn.run("findex.web.app:app", host="0.0.0.0", port=port, workers=workers)


if __name__ == "__main__":
    app()
