# src / settings.py
# Created by azat at 4.01.2023
from pathlib import Path

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    data_dir: Path | str = Field(default=Path.cwd() / 'data', env='DATA_DIR')

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


settings = Settings()
