import asyncio
import time
import random
from dataclasses import dataclass
import httpx

@dataclass
class FetchResult:
    status: int
    text: str = ""
    error: str = ""
    bytes_downloaded: int = 0
    elapsed: float = 0.0

async def fetch(client: httpx.AsyncClient, url: str) -> FetchResult:
    """Завантажує сторінку з підтримкою повторів (до 3 разів) при збоях."""
    max_retries = 3

    for attempt in range(max_retries):
        start_time = time.monotonic()
        try:
            # Вимога методички: використання asyncio.timeout
            async with asyncio.timeout(10.0):
                response = await client.get(url)
                elapsed = time.monotonic() - start_time

                # Повторюємо лише для 429 (Too Many Requests) або 5xx (Server Errors)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == max_retries - 1:
                        return FetchResult(status=response.status_code, error=f"HTTP {response.status_code}", elapsed=elapsed)

                    # Перевіряємо, чи сказав сервер, скільки чекати
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = float(retry_after)
                    else:
                        # Exponential backoff + jitter (щоб воркери не "били" сервер одночасно)
                        delay = (2 ** attempt) + random.uniform(0.1, 1.0)

                    await asyncio.sleep(delay)
                    continue

                # Для 200 OK або помилок клієнта (404, 403 тощо) - не повторюємо
                return FetchResult(
                    status=response.status_code,
                    text=response.text,
                    bytes_downloaded=len(response.content),
                    elapsed=elapsed
                )

        except (TimeoutError, httpx.RequestError) as e:
            elapsed = time.monotonic() - start_time
            if attempt == max_retries - 1:
                return FetchResult(status=0, error=type(e).__name__, elapsed=elapsed)

            delay = (2 ** attempt) + random.uniform(0.1, 1.0)
            await asyncio.sleep(delay)

    return FetchResult(status=0, error="Unknown Error")
