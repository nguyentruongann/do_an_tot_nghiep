from __future__ import annotations
from pathlib import Path
import os, yaml
from pydantic import BaseModel

class LLMConfig(BaseModel):
    model: str
    temperature: float = 0.0
    base_url: str
    api_key: str

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080

class StorageConfig(BaseModel):
    sqlite_path: str = "/app/data/warehouse.db"

class Config(BaseModel):
    llm: LLMConfig
    server: ServerConfig
    storage: StorageConfig

    @staticmethod
    def load(path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            return Config(**yaml.safe_load(f))

def load_config() -> "Config":
    cfg_path = Path(os.environ.get("CONFIG_PATH", "/app/config/server_config.yaml"))
    return Config.load(cfg_path)