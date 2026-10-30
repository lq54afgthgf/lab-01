import pytest
import httpx
from findex.crawl import canonicalize
from findex.robots import RobotsCache
from findex.fetch import fetch

# 1. Тестуємо канонізацію URL (регістр, сортування параметрів, видалення якорів)
def test_canonicalize():
    # Перевірка приведення до нижнього регістру та додавання слеша
    assert canonicalize("HTTPS://Example.com") == "https://example.com/"

    # Перевірка сортування query-параметрів
    assert canonicalize("http://test.com/page?b=2&a=1") == "http://test.com/page?a=1&b=2"

    # Перевірка видалення фрагментів (якорів #)
    assert canonicalize("http://test.com/index.html#section1") == "http://test.com/index.html"

# 2. Тестуємо поведінку RobotsCache, коли robots.txt відсутній (повертає 404)
@pytest.mark.asyncio
async def test_robots_allowed_when_404(httpx_mock):
    # Мокаємо (імітуємо) відповідь сервера: robots.txt не знайдено
    httpx_mock.add_response(url="https://example.com/robots.txt", status_code=404)

    robots = RobotsCache()
    async with httpx.AsyncClient() as client:
        # Оскільки файл не знайдено, краулінг має бути дозволений
        is_allowed = await robots.allowed("https://example.com/some/path", client)
        assert is_allowed is True

# 3. Тестуємо базовий fetch для успішного запиту
@pytest.mark.asyncio
async def test_fetch_success(httpx_mock):
    test_url = "https://example.com/page"
    mock_html = "<html><body><h1>Hello</h1></body></html>"

    # Імітуємо успішну відповідь 200 OK з текстом
    httpx_mock.add_response(url=test_url, status_code=200, text=mock_html)

    async with httpx.AsyncClient() as client:
        result = await fetch(client, test_url)

        assert result.status == 200
        assert result.text == mock_html
        assert result.error == ""
