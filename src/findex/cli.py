import sys
import json
import logging
import typer
import tracemalloc

from collections import Counter
from pathlib import Path
from typing import Annotated
from typing import Literal
from rich.console import Console
from rich.table import Table
from rich.progress import track
from findex.tokenize import tokenize
from findex.store import open_index, save_index
from findex.ranking import BM25, TfIdf, rank_documents
from findex.corpus import iter_documents
from findex.search import get_snippet, evaluate_query_cached
from findex.types import Index

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
@app.command()
def stats(
    corpus_path: Annotated[Path, typer.Argument(help="Шлях до папки з текстовими файлами")],
) -> None:
    """Виводить статистику корпусу (Топ-50 слів та порівняння використання пам'яті)."""
    logger.info(f"Збір статистики з {corpus_path}...")

    if not corpus_path.exists() or not corpus_path.is_dir():
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
) -> None:
    """Будує інвертований індекс із вказаного корпусу та зберігає його на диск."""
    logger.info(f"Початок побудови індексу з {corpus_path}...")

    if not corpus_path.exists() or not corpus_path.is_dir():
        console.print(f"[red]Помилка: Папки '{corpus_path}' не існує.[/red]")
        raise typer.Exit(1)

    idx = Index()

    console.print("[green]Обробка текстів та побудова індексу...[/green]")

    # Використовуємо метод класу Index через зручний метод add_document
    for doc_id, text in track(enumerate(iter_documents(corpus_path)), description="Індексування..."):
        tokens = list(tokenize(text))
        # Заповнюємо метадані та постинг-листи автоматично
        idx.add_document(doc_id=doc_id, tokens=tokens, path=str(corpus_path), title=f"Doc {doc_id}")

    console.print(f"[green]Збереження індексу у файл {index_path}...[/green]")
    save_index(idx, str(index_path))
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

if __name__ == "__main__":
    app()
