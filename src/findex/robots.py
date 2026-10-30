import urllib.parse
import urllib.robotparser
import httpx

# Обов'язково чесний User-Agent із посиланням для зв'язку (вимога етики краулерів)
USER_AGENT = "FindexCrawler/1.0 (+https://github.com/Tserelenko/findex)"

class RobotsCache:
    def __init__(self, user_agent: str = USER_AGENT):
        self.user_agent = user_agent
        self.cache: dict[str, urllib.robotparser.RobotFileParser] = {}

    async def allowed(self, url: str, client: httpx.AsyncClient) -> bool:
        parsed = urllib.parse.urlparse(url)
        host_url = f"{parsed.scheme}://{parsed.netloc}"
        robots_url = f"{host_url}/robots.txt"

        if host_url not in self.cache:
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(robots_url)
            try:
                resp = await client.get(robots_url, follow_redirects=True, timeout=5.0)
                if resp.status_code == 200:
                    parser.parse(resp.text.splitlines())
                else:
                    # Якщо 404 або інше - дозволяємо все
                    parser.allow_all = True
            except Exception:
                # При помилці мережі також дозволяємо
                parser.allow_all = True

            # Обов'язково повідомляємо парсер, що він був оновлений!
            parser.modified()
            self.cache[host_url] = parser

        return self.cache[host_url].can_fetch(self.user_agent, url)
