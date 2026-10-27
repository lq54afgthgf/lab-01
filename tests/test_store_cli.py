import pytest
from unittest.mock import patch
from findex.types import Index, DocMeta
from findex.store import save_index, load_index, open_index
from findex.search import main as search_main
from findex.stats import main as stats_main

@pytest.mark.parametrize("filename", ["index.bin", "data.pickle", "test_idx.idx"])
def test_save_and_load_index(tmp_path, sample_index, filename):
    """Тестуємо збереження та завантаження індексу."""
    file_path = tmp_path / filename
    save_index(sample_index, str(file_path))
    assert file_path.exists()

    loaded = load_index(str(file_path))
    assert loaded.num_docs == sample_index.num_docs

def test_open_index_context_manager(tmp_path, sample_index):
    """Тестуємо context manager для безпечного відкриття індексу."""
    file_path = tmp_path / "test.bin"
    save_index(sample_index, str(file_path))

    with open_index(str(file_path)) as idx:
        assert idx.doc_meta[1].title == "Doc 1"

@patch('sys.argv', ['search.py', 'dummy_index.bin', 'dummy_corpus', 'test query'])
@patch('findex.search.open_index')
@patch('findex.search.iter_documents')
def test_search_cli_basic(mock_iter, mock_open, tmp_path, sample_index):
    """Тестуємо базовий виклик CLI пошуку через argparse."""
    mock_open.return_value.__enter__.return_value = sample_index
    mock_iter.return_value = ["Test document content"]

    try:
        search_main()
    except SystemExit:
        pytest.fail("CLI пошуку несподівано завершив роботу")

@patch('sys.argv', ['stats.py', 'non_existent_directory_12345'])
def test_stats_cli_missing_dir():
    """Тестуємо CLI статистики при відсутності папки (має коректно завершитись)."""
    try:
        stats_main()
    except SystemExit:
        pytest.fail("CLI статистики впав замість коректної обробки")
