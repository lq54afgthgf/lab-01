import pickle
import pathlib
import json
from typing import Any

def save_index(index: Any, path: str | pathlib.Path):
    """
    Зберігає об'єкт індексу у бінарний файл за допомогою модуля pickle.
    """
    with open(path, 'wb') as f:
        pickle.dump(index, f)

def load_index(path: str | pathlib.Path) -> Any:
    """
    УВАГА: Використання pickle.load для неперевірених (чужих) файлів є небезпечним!
    Pickle може серіалізувати довільні об'єкти Python, і під час десеріалізації (load)
    зловмисник може виконати шкідливий код на вашій машині (наприклад, через метод __reduce__).
    Ніколи не завантажуйте файли .bin з невідомих джерел.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

# --- Альтернативний формат збереження (JSON) ---

def save_index_json(index: Any, path: str | pathlib.Path):
    """
    Зберігає індекс у форматі JSON.
    Це безпечніший, але повільніший і менш компактний формат порівняно з pickle.
    Оскільки Posting і DocMeta - це dataclass, ми вручну перетворюємо їх на словники.
    """
    data_to_save = {
        "doc_meta": {doc_id: {"length": meta.length, "path": meta.path, "title": meta.title}
                     for doc_id, meta in index.doc_meta.items()},
        "postings": {term: [{"doc_id": p.doc_id, "tf": p.tf, "positions": p.positions}
                            for p in postings_list]
                     for term, postings_list in index.postings.items()}
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
