# Етап 1: Збірка залежностей за uv.lock
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src

# Етап 2: Фінальний образ
FROM python:3.13-slim-bookworm
WORKDIR /app

# Копіюємо готове віртуальне середовище .venv, вихідний код та індекс
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY index.bin /app/index.bin

RUN useradd -m appuser
USER appuser

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
EXPOSE 8000

CMD ["python", "-m", "findex.cli", "serve", "--port", "8000"]
