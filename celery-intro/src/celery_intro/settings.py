from pydantic_settings import BaseSettings
from pydantic import Field

from functools import lru_cache


class Settings(BaseSettings):
    """
    Settings for the application.
    """

    # Add your settings here
    app_name: str = Field("Celery Intro", description="The name of the application")
    debug: bool = Field(False, description="Enable or disable debug mode")
    database_url: str = Field("sqlite:///./test.db", description="The database URL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Get the application settings.

    Returns:
        Settings: The application settings.
    """
    return Settings()
