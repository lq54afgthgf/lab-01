import pytest
from findex.tokenize import tokenize, tokenize_greedy

@pytest.mark.parametrize("text, expected", [
    ("Hello World", ["hello", "world"]),
    ("café au lait", ["café", "au", "lait"]),
    ("O'Neill and state-of-the-art", ["o'neill", "and", "state-of-the-art"]),
    ("   ", []),
    ("Python 3.12 is fast!", ["python", "3", "12", "is", "fast"]),
])
def test_tokenize_parameterized(text, expected):
    assert list(tokenize(text)) == expected
    assert tokenize_greedy(text) == expected
