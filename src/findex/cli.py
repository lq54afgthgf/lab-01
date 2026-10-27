import sys
import json
import logging
import typer

from pathlib import Path
from typing import Annotated
from typing import Literal

from rich.console import Console
from rich.table import Table
from rich.progress import track

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
def index(
    corpus_path: Annotated[Path, typer.Argument(help="Шлях до папки з текстовими файлами")],
    output_path: Annotated[Path, typer.Argument(help="Шлях для збереження індексу (.bin)")],
) -> None:
    """Збирає індекс з текстових файлів та зберігає його на диск."""
    logger.info(f"Початок збірки індексу з {corpus_path}...")

    if not corpus_path.exists() or not corpus_path.is_dir():
        console.print(f"[red]Помилка: Папки '{corpus_path}' не існує.[/red]")
        raise typer.Exit(1)

    # Примітивна заглушка збірки (замініть на ваш реальний білдер, якщо він є)
    # Використовуємо смугу прогресу Rich
    files = list(corpus_path.glob("*.txt"))
    idx = Index()

    for _ in track(files, description="[green]Токенізація та індексація..."):
        # Тут має бути ваш реальний код додавання документів до індексу
        pass

    try:
        save_index(idx, str(output_path))
        logger.info(f"Індекс успішно збережено у {output_path}")
        console.print("[bold green]Збірку індексу завершено![/bold green]")
    except Exception as e:
        console.print(f"[red]Помилка при збереженні індексу: {e}[/red]")
        raise typer.Exit(1)

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

@app.command()
def stats() -> None:
    """Виводить статистику (заглушка для команди stats)."""
    logger.info("Виклик команди stats")
    console.print("[blue]Тут буде статистика вашого рушія...[/blue]")

if __name__ == "__main__":
    app()
