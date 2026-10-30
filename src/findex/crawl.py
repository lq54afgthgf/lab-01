import asyncio
import time
import urllib.parse
from dataclasses import dataclass
from typing import AsyncIterator

import httpx
from selectolax.lexbor import LexborHTMLParser

from .fetch import fetch
from .robots import RobotsCache

@dataclass
class Page:
    url: str
    title: str
    text: str
    fetched_at: float
    status: int
    error: str = ""
    queue_depth: int = 0
    active_tasks: int = 0

def canonicalize(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path if parsed.path else "/"

    query_params = urllib.parse.parse_qsl(parsed.query)
    query = urllib.parse.urlencode(sorted(query_params))

    return urllib.parse.urlunparse((scheme, netloc, path, parsed.params, query, ""))

async def crawl(
    seeds: list[str],
    *,
    max_pages: int,
    concurrency: int,
    per_host: int,
    allowed_domains: list[str]
) -> AsyncIterator[Page]:

    frontier = asyncio.Queue()
    seen = set()

    for seed in seeds:
        can_url = canonicalize(seed)
        frontier.put_nowait(can_url)
        seen.add(can_url)

    global_sem = asyncio.Semaphore(concurrency)
    host_sems: dict[str, asyncio.Semaphore] = {}
    last_fetch_time: dict[str, float] = {}

    output_queue = asyncio.Queue()
    active_tasks = 0
    robots = RobotsCache()

    # Функція worker знаходиться СЕРЕДИНИ crawl, тому nonlocal працює
    async def worker(client: httpx.AsyncClient):
        nonlocal active_tasks
        try:
            while True:
                url = await frontier.get()
                active_tasks += 1

                try:
                    parsed = urllib.parse.urlparse(url)
                    host = parsed.netloc

                    if not any(host.endswith(d) for d in allowed_domains):
                        continue

                    if not await robots.allowed(url, client):
                        continue

                    if host not in host_sems:
                        host_sems[host] = asyncio.Semaphore(per_host)

                    async with global_sem:
                        async with host_sems[host]:
                            now = time.monotonic()
                            delay = max(0.0, 0.5 - (now - last_fetch_time.get(host, 0.0)))
                            if delay > 0:
                                await asyncio.sleep(delay)

                            last_fetch_time[host] = time.monotonic()
                            result = await fetch(client, url)

                    page = Page(
                        url=url, title="", text="", fetched_at=time.time(),
                        status=result.status, error=result.error,
                        queue_depth=frontier.qsize(),
                        active_tasks=active_tasks
                    )
                    if result.status == 200 and result.text:
                        tree = LexborHTMLParser(result.text)

                        title_node = tree.css_first("title")
                        page.title = title_node.text(strip=True) if title_node else ""
                        page.text = tree.text(separator=" ", strip=True)

                        for node in tree.css("a"):
                            href = node.attributes.get("href")
                            if href:
                                next_url = urllib.parse.urljoin(url, href)
                                if next_url.startswith(("http://", "https://")):
                                    next_can = canonicalize(next_url)
                                    if next_can not in seen:
                                        seen.add(next_can)
                                        frontier.put_nowait(next_can)

                    await output_queue.put(page)

                finally:
                    active_tasks -= 1
                    if frontier.empty() and active_tasks == 0:
                        output_queue.put_nowait(None)
                    frontier.task_done()

        except asyncio.CancelledError:
            pass

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        async with asyncio.TaskGroup() as tg:
            workers = [tg.create_task(worker(client)) for _ in range(concurrency)]

            pages_yielded = 0

            while pages_yielded < max_pages:
                if frontier.empty() and active_tasks == 0 and output_queue.empty():
                    break

                page = await output_queue.get()
                if page is None:
                    break

                yield page
                pages_yielded += 1

            for w in workers:
                w.cancel()
