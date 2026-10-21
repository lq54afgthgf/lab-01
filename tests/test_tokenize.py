from findex.tokenize import tokenize

def test_tokenize_cyrillic_and_punctuation():
    text = "Привіт, комп'ютер! Як справи-то?"
    tokens = list(tokenize(text))
    assert tokens == ["привіт", "комп'ютер", "як", "справи-то"]

def test_tokenize_cafe_normalization():
    # Слово "cafe" зі складеним наголосом (NFD)
    text_nfd = "cafe\u0301"
    tokens = list(tokenize(text_nfd))
    # Має звестися до єдиного символу "е" (NFC)
    assert tokens == ["café"]

def test_tokenize_casefold():
    text = "ВЕЛИКІ ЛІТЕРИ"
    tokens = list(tokenize(text))
    assert tokens == ["великі", "літери"]
