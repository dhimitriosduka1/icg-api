from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from ``ICG_*`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="ICG_")

    checkpoint_path: Path = Path("artifacts/model.pth.tar")
    vocabulary_path: Path = Path("artifacts/vocab.json")
    device: str = Field("auto", description="'auto', 'cpu', 'cuda', 'cuda:1', ...")
    max_caption_length: int = Field(20, ge=1)
    max_upload_bytes: int = Field(10 * 1024 * 1024, ge=1)
    host: str = "0.0.0.0"
    port: int = 3000
    log_level: Literal["debug", "info", "warning", "error"] = "info"
