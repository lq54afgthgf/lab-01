import httpx
import pytest

from findex.crawl import canonicalize
from findex.fetch import fetch
from findex.robots import RobotsCache


def test_canonicalize() -> None:
    """Перевірка канонізації URL: регістр, слеш, сортування query, відкидання #."""
    assert canonicalize("HTTPS://Example.com") == "https://example.com/"
    assert canonicalize("http://test.com/page?b=2&a=1") == "http://test.com/page?a=1&b=2"
    assert canonicalize("http://test.com/index.html#section1") == "http://test.com/index.html"


@pytest.mark.asyncio
async def test_robots_cache_with_mock_transport() -> None:
    """Тестуємо RobotsCache через httpx.MockTransport (кешування та правила Disallow/404)."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if request.url.host == "allowed.com":
            return httpx.Response(404)
        return httpx.Response(200, text="User-agent: *\nDisallow: /private\nAllow: /\n")

    transport = httpx.MockTransport(handler)
    robots = RobotsCache()

    async with httpx.AsyncClient(transport=transport) as client:
        assert await robots.allowed("https://allowed.com/page", client) is True
        assert await robots.allowed("https://blocked.com/public", client) is True
        assert await robots.allowed("https://blocked.com/private/secret", client) is False
        # Повторний запит до того самого хоста береться з кешу без нового HTTP-запиту
        assert await robots.allowed("https://blocked.com/other", client) is True
        assert calls == 2


@pytest.mark.asyncio
async def test_fetch_retry_429_and_no_retry_404() -> None:
    """Перевіряємо повтор для 429 (з Retry-After) і відсутність повтору для 404 через MockTransport."""
    attempts_429 = 0
    attempts_404 = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts_429, attempts_404
        if request.url.path == "/rate-limited":
            attempts_429 += 1
            if attempts_429 < 3:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, text="<html><body>Recovered</body></html>")
        if request.url.path == "/not-found":
            attempts_404 += 1
            return httpx.Response(404, text="Not Found")
        return httpx.Response(200, text="OK")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        res_ok = await fetch(client, "https://example.com/rate-limited")
        assert res_ok.status == 200
        assert res_ok.bytes_downloaded > 0
        assert attempts_429 == 3

        res_404 = await fetch(client, "https://example.com/not-found")
        assert res_404.status == 404
        # 404 не повинен повторюватися
        assert attempts_404 == 1
