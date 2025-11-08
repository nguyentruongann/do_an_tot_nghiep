from __future__ import annotations
from pydantic import BaseModel
from pathlib import Path
import os, yaml

class LLMConfig(BaseModel):
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.0

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080

class StorageConfig(BaseModel):
    logs_key_prefix: str = "logs"
    cache_key_prefix: str = "cache"
    ttl_seconds: int = 60 * 60 * 24 * 3

class TrainingConfig(BaseModel):
    data_path: str | None = None

class Config(BaseModel):
    llm: LLMConfig
    server: ServerConfig
    storage: StorageConfig
    training: TrainingConfig

    @staticmethod
    def load(path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            return Config(**yaml.safe_load(f))

def load_config() -> Config:
    cfg_path = Path(os.environ.get("CONFIG_PATH", "/app/config/server_config.yaml"))
    return Config.load(cfg_path)
