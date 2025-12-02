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
    # Default to local directories within the repository.  These paths are
    # writable in the sandboxed environment.  They can be overridden via
    # environment variables or YAML configuration.
    sessions_dir: str = "data/sessions"
    global_dir: str = "data/global"

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
    """
    Nạp cấu hình từ YAML. Nếu không đặt CONFIG_PATH thì cố gắng
    tìm file 'config/server_config.yaml' nằm ngay trên thư mục chứa file config.py.
    Nếu không tìm thấy, rơi về biến môi trường như cũ.
    """
    # Ưu tiên dùng CONFIG_PATH nếu có
    cfg_env = os.environ.get("CONFIG_PATH")
    if cfg_env:
        cfg_path = Path(cfg_env)
    else:
        # Lấy đường dẫn thư mục cha (ví dụ: .../api/app)
        current_dir = Path(__file__).resolve().parents[1]
        # File cấu hình mặc định nằm tại .../api/config/server_config.yaml
        default_cfg = current_dir / "config" / "server_config.yaml"
        cfg_path = default_cfg

    # Nếu file tồn tại thì nạp YAML
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        llm = data.get("llm", {})
        # Lấy ưu tiên từ biến môi trường nếu có
        for k, env in [("base_url","LLM_BASE_URL"),
                       ("api_key","LLM_API_KEY"),
                       ("model","LLM_MODEL")]:
            if os.environ.get(env):
                llm[k] = os.environ[env]
        if os.environ.get("LLM_TEMPERATURE"):
            llm["temperature"] = float(os.environ["LLM_TEMPERATURE"])
        data["llm"] = llm
        return Config(**data)

    # Nếu không có file, fallback như hiện tại
    llm = LLMConfig(
        base_url=os.environ.get("LLM_BASE_URL", ""),
        api_key=os.environ.get("LLM_API_KEY", ""),
        model=os.environ.get("LLM_MODEL", ""),
        temperature=float(os.environ.get("LLM_TEMPERATURE", 0.2)),
    )
    server = ServerConfig(
        host=os.environ.get("SERVER_HOST", "0.0.0.0"),
        port=int(os.environ.get("SERVER_PORT", 8080)),
    )
    storage = StorageConfig(
        logs_key_prefix=os.environ.get("LOGS_KEY_PREFIX", "logs"),
        cache_key_prefix=os.environ.get("CACHE_KEY_PREFIX", "cache"),
        ttl_seconds=int(os.environ.get("TTL_SECONDS", 60 * 60 * 24 * 3)),
    )
    training = TrainingConfig(
        data_path=os.environ.get("TRAINING_DATA_PATH"),
        sessions_dir=os.environ.get("SESSIONS_DIR", "data/sessions"),
        global_dir=os.environ.get("GLOBAL_DIR", "data/global"),
    )
    return Config(llm=llm, server=server, storage=storage, training=training)
