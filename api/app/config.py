from __future__ import annotations
from pydantic import BaseModel
from pathlib import Path
import os, yaml

class LLMConfig(BaseModel):
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080

class StorageConfig(BaseModel):
    logs_key_prefix: str = "logs"
    cache_key_prefix: str = "cache"
    ttl_seconds: int = 60*60*24*3

class TrainingConfig(BaseModel):
    data_path: str | None = None
    sessions_dir: str = "/app/data/sessions"
    global_dir: str = "/app/data/global"

class Config(BaseModel):
    llm: LLMConfig
    server: ServerConfig
    storage: StorageConfig
    training: TrainingConfig

    @staticmethod
    def load(path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        llm = data.get("llm", {})
        for k, env in [("base_url","LLM_BASE_URL"),("api_key","LLM_API_KEY"),("model","LLM_MODEL")]:
            if os.environ.get(env): llm[k] = os.environ[env]
        if os.environ.get("LLM_TEMPERATURE"):
            llm["temperature"] = float(os.environ["LLM_TEMPERATURE"])
        data["llm"] = llm
        return Config(**data)

def load_config() -> Config:
    cfg_path = Path(os.environ.get("CONFIG_PATH", "/app/config/server_config.yaml"))
    return Config.load(cfg_path)
