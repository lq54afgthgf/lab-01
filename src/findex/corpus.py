from pathlib import Path
from typing import Generator

def iter_documents(directory_path: Path | str) -> Generator[str, None, None]:
    """Лінивий генератор: читає файли по одному рядку за раз."""
    path = Path(directory_path)
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
    return [file_path.read_text(encoding='utf-8', errors='ignore')
            for file_path in path.rglob('*.txt') if file_path.is_file()]
