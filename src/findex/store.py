import json
import pickle
from array import array
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from findex.types import DocMeta, Index, Posting


def save_index(index: Index, path: str, fmt: str = "auto") -> None:
    """Зберігає індекс у форматі pickle (.bin/.pkl) або JSON (.json)."""
    use_json = fmt == "json" or (fmt == "auto" and str(path).endswith(".json"))
    if use_json:
        data = {
            "doc_lengths": {str(k): v for k, v in index.doc_lengths.items()},
            "doc_meta": {
                str(k): {"length": m.length, "path": m.path, "title": m.title}
                for k, m in index.doc_meta.items()
            },
            "postings": {
                term: [
                    [p.doc_id, p.tf, list(p.positions) if p.positions else None]
                    for p in plist
                ]
                for term, plist in index.postings.items()
            },
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    else:
        with open(path, "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path: str, fmt: str = "auto") -> Index:
    """
    Завантажує об'єкт індексу з файлу.

    УВАГА ЩОДО БЕЗПЕКИ (Чому pickle.load чужого файлу небезпечний):
    Модуль pickle є стековою віртуальною машиною, яка під час десеріалізації може
    викликати довільні Python-функції та системні команди через метод __reduce__()
    (Remote Code Execution / RCE). Завантажувати через pickle.load можна виключно
    довірені локальні файли, створені самим застосунком. Для недовірених даних
    слід використовувати безпечний формат JSON (fmt="json").
    """
    use_json = fmt == "json" or (fmt == "auto" and str(path).endswith(".json"))
    if use_json:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        idx = Index()
        idx.doc_lengths = {int(k): int(v) for k, v in raw["doc_lengths"].items()}
        idx.doc_meta = {
            int(k): DocMeta(
                length=v["length"], path=v["path"], title=v.get("title", "")
            )
            for k, v in raw["doc_meta"].items()
        }
        for term, plist in raw["postings"].items():
            idx.postings[term] = [
                Posting(
                    doc_id=item[0],
                    tf=item[1],
                    positions=tuple(item[2]) if item[2] is not None else None,
                )
                for item in plist
            ]
        return idx

    with open(path, "rb") as f:
        return pickle.load(f)


def to_compact_arrays(index: Index) -> dict[str, tuple[array, array]]:
    """Перетворює постінги на компактні C-масиви array('I') (unsigned int32)."""
    compact: dict[str, tuple[array, array]] = {}
    for term, plist in index.postings.items():
        doc_ids = array("I", (p.doc_id for p in plist))
        tfs = array("I", (p.tf for p in plist))
        compact[term] = (doc_ids, tfs)
    return compact


@contextmanager
def open_index(path: str) -> Generator[Index, None, None]:
    """Контекстний менеджер для безпечного завантаження індексу."""
    index = load_index(path)
    try:
        yield index
    finally:
        pass
