from pathlib import Path

from typer.testing import CliRunner

from findex.cli import app, build_partial
from findex.store import load_index
from findex.types import Index

runner = CliRunner()


def test_build_partial_and_merge_equivalence(tmp_path: Path) -> None:
    """Послідовний варіант merge([build_partial(усі шляхи)]) та розбиття на чанки дають ідентичний індекс."""
    docs = [
        "Київ — столиця та найбільше місто України.",
        "Україна має багату історію, Київ розташований на Дніпрі.",
        "Паралельна індексація корпусу через процеси та потоки.",
        "Закон Амдала обмежує максимальне прискорення через фазу merge.",
    ]
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    paths: list[Path] = []
    for i, content in enumerate(docs):
        f = corpus_dir / f"doc_{i}.txt"
        f.write_text(content, encoding="utf-8")
        paths.append(f)

    # 1. Прямий виклик на всі шляхи (як у Лаб 4)
    direct_idx = build_partial(paths, start_doc_id=0)

    # 2. Послідовний варіант merge([build_partial(усі шляхи)])
    single_merged = Index()
    single_merged.merge(build_partial(paths, start_doc_id=0))

    # 3. Розбиття на 2 чанки з неперетинними діапазонами start_doc_id
    chunk1, chunk2 = paths[:2], paths[2:]
    multi_merged = Index()
    multi_merged.merge(build_partial(chunk1, start_doc_id=0))
    multi_merged.merge(build_partial(chunk2, start_doc_id=len(chunk1)))

    assert direct_idx.num_docs == single_merged.num_docs == multi_merged.num_docs == 4
    assert dict(direct_idx.postings) == dict(single_merged.postings) == dict(multi_merged.postings)
    assert direct_idx.doc_lengths == single_merged.doc_lengths == multi_merged.doc_lengths
    assert direct_idx.avg_doc_length == multi_merged.avg_doc_length


def test_cli_executors_produce_identical_index(tmp_path: Path) -> None:
    """Перевіряє, що serial, threads та processes через CLI дають однаковий індекс."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for i in range(6):
        (corpus_dir / f"d{i}.txt").write_text(
            f"документ номер {i} спільне слово пошук індекс", encoding="utf-8"
        )

    indices: list[Index] = []
    for mode in ("serial", "threads", "processes"):
        out_file = tmp_path / f"idx_{mode}.bin"
        res = runner.invoke(
            app,
            ["index", str(corpus_dir), str(out_file), "--workers", "2", "--executor", mode],
        )
        assert res.exit_code == 0, res.stderr
        indices.append(load_index(out_file))

    serial_idx, threads_idx, proc_idx = indices
    assert serial_idx.num_docs == threads_idx.num_docs == proc_idx.num_docs == 6
    assert len(serial_idx) == len(threads_idx) == len(proc_idx)
    assert serial_idx.df("спільне") == threads_idx.df("спільне") == proc_idx.df("спільне") == 6
