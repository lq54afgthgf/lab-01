# Етап 1: Збірка залежностей за uv.lock
FROM python:3.13-slim-bookworm
WORKDIR /app

# Копіюємо пакети, вихідний код ТА файл індексу
COPY --from=builder /app/.venv/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /app/src /app/src
COPY index.bin /app/index.bin

RUN useradd -m appuser

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
RUN uv sync --frozen --no-editable --no-dev

# Етап 2: Фінальний образ
FROM python:3.13-slim-bookworm
WORKDIR /app

# Копіюємо ТІЛЬКИ пакети (site-packages) та вихідний код, не чіпаючи /usr/local/bin!
COPY --from=builder /app/.venv/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /app/src /app/src

RUN useradd -m appuser
USER appuser

ENV PYTHONPATH="/app/src"
EXPOSE 8000

CMD ["/usr/local/bin/python3", "-m", "findex.cli", "serve", "--port", "8000"]
