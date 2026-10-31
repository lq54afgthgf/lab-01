import sys
import json
import logging
import typer
import tracemalloc
import pickle
import concurrent.futures
import multiprocessing
import time
import asyncio
import urllib.parse

from enum import Enum
from collections import Counter
from pathlib import Path
from typing import Annotated
from typing import Literal
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import track
from findex.tokenize import tokenize
from findex.store import open_index, save_index
from findex.ranking import BM25, TfIdf, rank_documents
from findex.corpus import iter_documents
from findex.search import get_snippet, evaluate_query_cached
from findex.types import Index
from findex.crawl import crawl

class ExecutorType(str, Enum):
    serial = "serial"
    threads = "threads"
    processes = "processes"

# Створюємо глобальні об'єкти
app = typer.Typer(help="Findex: локальний рушій пошуку", add_completion=False)
console = Console(stderr=True)  # Діагностика та таблиці йдуть у stderr, щоб stdout лишався чистим для JSON
logger = logging.getLogger("findex")

def setup_logging(verbose: int) -> None:
    """Налаштовує рівень логування залежно від кількості прапорців -v."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)]
    )

@app.callback()
def main(
    verbose: Annotated[int, typer.Option("-v", "--verbose", count=True, help="Рівень деталізації логів (до -vv)")] = 0
) -> None:
    setup_logging(verbose)


# --- НОВА ФУНКЦІЯ РІВНЯ МОДУЛЯ (MAP) ---
def build_partial(chunk_paths: list[Path], start_doc_id: int) -> Index:
    """Будує частковий індекс для заданого списку файлів."""
    partial_idx = Index()
    for offset, path in enumerate(chunk_paths):
        # Логіка для Лаби 6 (читаємо JSONL)
        if path.is_file() and path.suffix == ".jsonl":
            import json
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if not line.strip(): continue
                    record = json.loads(line)
                    text = record.get("text", "")
                    doc_id = start_doc_id + idx
                    tokens = list(tokenize(text))
                    title = record.get("title", f"Doc {doc_id}")
                    partial_idx.add_document(doc_id=doc_id, tokens=tokens, path=str(path), title=title)
        # Стара логіка для Лаби 5 (без змін)
        else:
            doc_id = start_doc_id + offset
            text = path.read_text(encoding="utf-8", errors="ignore")
            tokens = list(tokenize(text))
            partial_idx.add_document(doc_id=doc_id, tokens=tokens, path=str(path), title=f"Doc {doc_id}")

    return partial_idx

@app.command()
def stats(
    corpus_path: Annotated[Path, typer.Argument(help="Шлях до папки з текстовими файлами")],
) -> None:
    """Виводить статистику корпусу (Топ-50 слів та порівняння використання пам'яті)."""
    logger.info(f"Збір статистики з {corpus_path}...")

    if not corpus_path.exists():
        console.print(f"[red]Помилка: Папки '{corpus_path}' не існує.[/red]")
        raise typer.Exit(1)

    # --- ЖАДІБНИЙ ПІДХІД ---
    console.print("[bold red]Обробка текстів (жадібне читання - list)...[/bold red]")
    tracemalloc.start()

    # Завантажуємо всі документи в оперативну пам'ять одразу
    all_docs = list(iter_documents(corpus_path))
    greedy_word_counts: Counter[str] = Counter()

    for doc_text in all_docs:
        greedy_word_counts.update(tokenize(doc_text))

    greedy_current, greedy_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Видаляємо змінні, щоб звільнити пам'ять перед наступним тестом
    del all_docs
    del greedy_word_counts

    # --- ЛІНИВИЙ ПІДХІД ---
    console.print("[bold green]Обробка текстів (ліниве читання - generator)...[/bold green]")
    tracemalloc.start()

    lazy_word_counts: Counter[str] = Counter()
    # Читаємо по одному документу, не забиваючи пам'ять
    for doc_text in iter_documents(corpus_path):
        lazy_word_counts.update(tokenize(doc_text))

    lazy_current, lazy_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- ВИВІД РЕЗУЛЬТАТІВ ---
    # 1. Таблиця Топ-50
    table_words = Table(title="Топ 50 найчастіших слів")
    table_words.add_column("Слово", style="cyan")
    table_words.add_column("Частота", justify="right", style="magenta")

    for word, count in lazy_word_counts.most_common(50):
        table_words.add_row(word, str(count))

    console.print(table_words)

    # 2. Таблиця порівняння пам'яті
    console.print(f"\n[bold yellow]Порівняння використання пам'яті (tracemalloc):[/bold yellow]")
    mem_table = Table(show_header=True, header_style="bold white")
    mem_table.add_column("Підхід читання")
    mem_table.add_column("Поточна пам'ять (кінець)", justify="right")
    mem_table.add_column("Пікова пам'ять (максимум)", justify="right", style="bold red")

    mem_table.add_row(
        "Жадібний (List)",
        f"{greedy_current / 10**6:.2f} MB",
        f"{greedy_peak / 10**6:.2f} MB"
    )
    mem_table.add_row(
        "Лінивий (Generator)",
        f"{lazy_current / 10**6:.2f} MB",
        f"{lazy_peak / 10**6:.2f} MB"
    )

    console.print(mem_table)

@app.command()
def index(
    corpus_path: Annotated[Path, typer.Argument(help="Шлях до папки з текстовими файлами")],
    index_path: Annotated[Path, typer.Argument(help="Шлях для збереження індексу (*.bin)")],
    workers: Annotated[int, typer.Option("--workers", help="Кількість воркерів")] = 1,
    executor: Annotated[ExecutorType, typer.Option("--executor", help="Тип екзекутора")] = ExecutorType.serial,
) -> None:

    """Будує інвертований індекс із вказаного корпусу та зберігає його на диск."""
    if hasattr(sys, '_is_gil_enabled'):
        console.print(f"[bold blue]Статус GIL (sys._is_gil_enabled): {sys._is_gil_enabled()}[/bold blue]")
    else:
        console.print("[bold blue]Статус GIL: звичайна збірка (GIL увімкнено)[/bold blue]")

    logger.info(f"Початок побудови індексу з {corpus_path}...")

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

    # Розбиваємо файли на шматки (чанки) для воркерів
    import math
    chunk_size = math.ceil(len(all_paths) / workers) if workers > 0 else len(all_paths)
    if chunk_size == 0: chunk_size = 1
    chunks = [all_paths[i:i + chunk_size] for i in range(0, len(all_paths), chunk_size)]

    console.print(f"[green]Знайдено {len(all_paths)} файлів. Чанків: {len(chunks)}. Режим: {executor.value}, Воркерів: {workers}[/green]")

    t0 = time.perf_counter()
    partials = []

    # Вибір стратегії виконання
    if executor == ExecutorType.serial:
        start_id = 0
        for chunk in chunks:
            partials.append(build_partial(chunk, start_doc_id=start_id))
            start_id += len(chunk)
    else:
        PoolClass = concurrent.futures.ThreadPoolExecutor if executor == ExecutorType.threads else concurrent.futures.ProcessPoolExecutor
        kwargs = {"max_workers": workers}

        # Безпечний запуск для процесів (уникаємо fork)
        if executor == ExecutorType.processes:
            kwargs["mp_context"] = multiprocessing.get_context("spawn")

        with PoolClass(**kwargs) as pool:
            futures = []
            start_id = 0
            for chunk in chunks:
                futures.append(pool.submit(build_partial, chunk, start_id))
                start_id += len(chunk)

            # as_completed дозволяє ловити помилки у воркерах одразу, як вони падають
            for fut in concurrent.futures.as_completed(futures):
                try:
                    partials.append(fut.result())
                except Exception as e:
                    console.print(f"[bold red]Воркер впав з помилкою: {e}[/bold red]")
                    raise typer.Exit(1)

    t1 = time.perf_counter()

    console.print(f"[green]Виконуємо merge ({len(partials)} часткових індексів)...[/green]")
    final_index = Index()
    for p in partials:
        final_index.merge(p)

    t2 = time.perf_counter()

    console.print(f"[bold yellow]Час збирання: {t1 - t0:.2f} сек. | Час merge: {t2 - t1:.2f} сек.[/bold yellow]")
    console.print(f"[green]Збереження індексу у {index_path}...[/green]")
    save_index(final_index, str(index_path))
    console.print("[bold green]Індекс успішно збережено![/bold green]")

@app.command()
def search(
    index_path: Annotated[Path, typer.Argument(help="Шлях до файлу індексу")],
    corpus_path: Annotated[Path, typer.Argument(help="Шлях до корпусу (для витягування сніпетів)")],
    query: Annotated[str, typer.Argument(help="Запит")],
    scorer_name: Annotated[Literal["bm25", "tfidf"], typer.Option("--scorer", help="Алгоритм ранжування")] = "bm25",
    output_json: Annotated[bool, typer.Option("--json", help="Вивести результати у форматі JSON в stdout")] = False,
) -> None:
    """Виконує пошук за індексом та виводить результати."""
    logger.debug(f"Запуск пошуку: запит='{query}', scorer='{scorer_name}'")

    try:
        with open_index(str(index_path)) as idx:
            logger.info("Індекс завантажено.")

            matching_docs = set(evaluate_query_cached(query, idx))
            if not matching_docs:
                logger.info("За запитом нічого не знайдено.")
                if output_json:
                    print(json.dumps([]))
                else:
                    console.print("[yellow]Нічого не знайдено.[/yellow]")
                return

            scorer = BM25() if scorer_name == "bm25" else TfIdf()
            query_words = {w.strip('()"').lower() for w in query.split() if w not in ("AND", "OR", "NOT")}

            top_results = rank_documents(query_words, matching_docs, idx, scorer, top_k=5)

            top_doc_ids = {res.doc_id for res in top_results}
            doc_texts: dict[int, str] = {}
            for doc_id, text in enumerate(iter_documents(corpus_path)):
                if doc_id in top_doc_ids:
                    doc_texts[doc_id] = text

            if output_json:
                # JSON йде у стандартний вивід (stdout)
                results_data = [
                    {"rank": r, "doc_id": res.doc_id, "title": res.title, "score": res.score}
                    for r, res in enumerate(top_results, 1)
                ]
                print(json.dumps(results_data, ensure_ascii=False, indent=2))
            else:
                # Таблиця йде у stderr через console
                table = Table(title=f"Результати пошуку ({scorer_name.upper()})")
                table.add_column("Ранг", justify="right", style="cyan", no_wrap=True)
                table.add_column("Документ", style="magenta")
                table.add_column("Оцінка", justify="right", style="green")
                table.add_column("Сніпет", overflow="fold")

                for rank, res in enumerate(top_results, 1):
                    snippet = get_snippet(doc_texts.get(res.doc_id, ""), query_words, window=40)
                    table.add_row(str(rank), f"{res.doc_id}: {res.title}", f"{res.score:.4f}", snippet)

                console.print(table)

    except FileNotFoundError:
        console.print(f"[red]Помилка: Файл індексу '{index_path}' не знайдено.[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Критична помилка пошуку: {e}[/red]")
        raise typer.Exit(1)

@app.command(name="crawl")
def crawl_cmd(
    seed: str = typer.Argument(..., help="Початковий URL для краулінгу"),
    max_pages: int = typer.Option(200, help="Максимальна кількість сторінок"),
    concurrency: int = typer.Option(10, help="Кількість одночасних воркерів"),
    per_host: int = typer.Option(2, help="Ліміт запитів на один хост"),
    out: Path = typer.Option(Path("data/crawl.jsonl"), help="Шлях до вихідного JSONL файлу"),
    allowed_domain: str = typer.Option(None, help="Додатковий дозволений домен")
):
    """
    Запуск асинхронного краулера для збору сторінок у JSONL.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    parsed_seed = urllib.parse.urlparse(seed)
    domains = [parsed_seed.netloc]
    if allowed_domain:
        domains.append(allowed_domain)

    async def run_crawler():
        fetched_count = 0
        error_count = 0
        start_time = time.monotonic()
        log_path = Path("crawl.log")

        def generate_table(q_depth=0, active_reqs=0):
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

        with open(out, "w", encoding="utf-8") as jsonl_f, \
             open(log_path, "w", encoding="utf-8") as log_f, \
             Live(generate_table(), refresh_per_second=4) as live:

            async for page in crawl(
                [seed],
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                allowed_domains=domains
            ):
                fetched_count += 1
                if page.error:
                    error_count += 1

                # Записуємо успішні сторінки у JSONL
                if page.status == 200 and page.text:
                    record = {
                        "url": page.url,
                        "title": page.title,
                        "text": page.text
                    }
                    jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Записуємо лог у crawl.log
                log_f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(page.fetched_at))}] Status: {page.status} | URL: {page.url} | Error: {page.error}\n")
                log_f.flush()

                live.update(generate_table(page.queue_depth, page.active_tasks))

        print(f"\n[bold green]Краулінг завершено![/bold green] Зібрано: {fetched_count}. Файл збережено у {out}.")

    # --- ЗАМІР ПАМ'ЯТІ (PEAK RSS) ---
    tracemalloc.start()
    asyncio.run(run_crawler())
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[bold yellow]Peak RSS: {peak / 10**6:.2f} MB[/bold yellow]")

@app.command(name="serve")
def serve(
    port: int = typer.Option(8000, help="Порт для веб-сервера"),
    workers: int = typer.Option(1, help="Кількість воркерів Uvicorn")
) -> None:
    """Запускає веб-інтерфейс та API пошуку."""
    import uvicorn
    console.print(f"[bold green]Запуск веб-сервера на порту {port} ({workers} воркерів)...[/bold green]")

    # Запускаємо ASGI-застосунок (який ми створимо на наступному кроці)
    uvicorn.run("findex.web.app:app", host="0.0.0.0", port=port, workers=workers)

if __name__ == "__main__":
    app()
