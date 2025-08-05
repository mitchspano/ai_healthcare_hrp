# src/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Settings loaded from environment or .env, using pydantic-settings v2.
    """

    openai_api_key: str
    frontend_origin: str = "http://localhost:5173"
    data_root: str = "AZT1D 2025"

    # Tell pydantic-settings where to read .env
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,       # make matching case-insensitive
        extra="ignore"              # ignore any other keys in .env
    )

settings = Settings()
