import pickle
from contextlib import contextmanager
from findex.types import Index

def save_index(index: Index, path: str):
    """Зберігає об'єкт індексу у файл."""
    with open(path, "wb") as f:
        pickle.dump(index, f)

def load_index(path: str) -> Index:
    """Завантажує об'єкт індексу з файлу."""
    with open(path, "rb") as f:
        return pickle.load(f)

@contextmanager
def open_index(path: str):
    """
    Контекстний менеджер для безпечного завантаження індексу.
    Використання: with open_index('index.bin') as index: ...
    """
    index = load_index(path)
    try:
        yield index
    finally:
        # Для pickle-індексу в пам'яті закривати нічого, але якщо
        # колись перейдемо на mmap або базу даних, ресурси будуть звільнені тут.
        pass
