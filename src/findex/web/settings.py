
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Якщо змінна INDEX_PATH не задана в оточенні, використається "index.bin"
    index_path: Path = Path("index.bin")
    
    # Дозволяє читати змінні з файлу .env для зручності локальної розробки
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Створюємо глобальний екземпляр. При імпорті він автоматично зчитає змінні середовища
settings = Settings()
