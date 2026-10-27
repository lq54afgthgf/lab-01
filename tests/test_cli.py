import pytest
from typer.testing import CliRunner

# Припускаємо, що об'єкт Typer називається app і знаходиться в cli.py
from findex.cli import app

runner = CliRunner()

def test_cli_help():
    """Тестуємо, що CLI відповідає на виклик --help."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.stdout or "Options" in result.stdout

def test_cli_no_args():
    """Тестуємо поведінку CLI при виклику без аргументів."""
    result = runner.invoke(app)
    # Typer зазвичай повертає ненульовий код або показує допомогу, якщо аргументи відсутні
    assert result.exit_code in (0, 2)
