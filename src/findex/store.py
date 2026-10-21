import pickle
import pathlib
from typing import Any

def save_index(index: Any, path: str | pathlib.Path):
    """
    Зберігає об'єкт індексу у бінарний файл за допомогою модуля pickle.
    """
    with open(path, 'wb') as f:
        pickle.dump(index, f)

def load_index(path: str | pathlib.Path) -> Any:

    with open(path, 'rb') as f:
        return pickle.load(f)
