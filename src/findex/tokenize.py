import re
import unicodedata
from collections.abc import Generator

# Регулярний вираз: шукає межі слова (\b), бере літери/цифри (\w),
# і дозволяє дефіси або апострофи всередині слова.
TOKEN_PATTERN = re.compile(r"\b[\w]+(?:['\-][\w]+)*\b", re.UNICODE)

def tokenize(text: str) -> Generator[str, None, None]:
    """
    Лінивий потоковий токенізатор (через yield).
    """
    # 1. NFC нормалізація (щоб café і cafe + ´ були однаковими)
    # 2. casefold() - агресивніше за lower(), спеціально для порівняння
    normalized_text = unicodedata.normalize('NFC', text).casefold()

    # finditer не створює список усіх збігів, а видає їх по черзі
    for match in TOKEN_PATTERN.finditer(normalized_text):
        yield match.group(0)

def tokenize_greedy(text: str) -> list[str]:
    """
    Жадібна версія токенізатора: повертає відразу готовий список усіх слів.
    Використовується лише для порівняння використання пам'яті.
    """
    normalized_text = unicodedata.normalize('NFC', text).casefold()
    # findall створює в пам'яті масив з усіх знайдених токенів
    return TOKEN_PATTERN.findall(normalized_text)
