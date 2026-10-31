# Етап 1: Збірка залежностей
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
RUN uv sync --frozen --no-editable --no-dev

# Етап 2: Фінальний образ
FROM python:3.13-slim-bookworm
WORKDIR /app

# Копіюємо ТІЛЬКИ пакети, вихідний код ТА файл індексу
COPY --from=builder /app/.venv/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /app/src /app/src
COPY index.bin /app/index.bin

RUN useradd -m appuser
USER appuser

ENV PYTHONPATH="/app/src"
EXPOSE 8000

CMD ["/usr/local/bin/python3", "-m", "findex.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
