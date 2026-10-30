import json
from pathlib import Path
from collections.abc import Iterator

def iter_documents(directory_path: Path | str) -> Iterator[str]:
    """Лінивий генератор: читає файли .txt з папки АБО один файл .jsonl."""
    path = Path(directory_path)

    # Додаємо підтримку JSONL файлу з краулера
    if path.is_file() and path.suffix == ".jsonl":
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    # Витягуємо лише текст сторінки для індексації
                    if "text" in record:
                        yield record["text"]
    else:
        # Стара логіка для папки з .txt файлами
        for file_path in path.rglob('*.txt'):
            if file_path.is_file():
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # Читаємо файл рядок за рядком, не вантажачи його цілком
                    for line in f:
                        if line.strip():  # ігноруємо порожні рядки
                            yield line

def get_documents_greedy(directory_path: Path | str) -> list[str]:
    """
    Жадібна версія: читає весь корпус у пам'ять одразу (повертає список).
    Потрібна лише для порівняння витрат пам'яті.
    """
    path = Path(directory_path)

    if path.is_file() and path.suffix == ".jsonl":
        docs = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if "text" in record:
                        docs.append(record["text"])
        return docs
    else:
        return [
            file_path.read_text(encoding='utf-8', errors='ignore')
            for file_path in path.rglob('*.txt') if file_path.is_file()
        ]
