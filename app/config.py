"""Конфигурация приложения через переменные окружения."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # PostgreSQL
    pg_host: str = "postgres"
    pg_port: int = 5432
    pg_db: str = "abstracts"
    pg_user: str = "abstracts_user"
    pg_password: str = "abstracts_pass"

    # Модель
    model_name: str = "IlyaGusev/rut5_base_sum_gazeta"
    model_revision: str = "main"
    local_model_path: str | None = None  # путь к зафайнтьюненной локальной модели

    # Pipeline
    extractive_threshold_tokens: int = 800   # если текст длиннее — включаем LexRank
    extractive_target_sentences: int = 40    # сколько предложений оставить
    max_input_tokens: int = 1024             # ограничение T5

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )


settings = Settings()