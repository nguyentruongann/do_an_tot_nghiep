import yaml, os

_cfg_cache = None
def _load_cfg():
    global _cfg_cache
    if _cfg_cache is None:
        path = os.environ.get("RUNTIME_CONFIG", "config/runtime.yaml")
        with open(path, "r", encoding="utf-8") as f:
            _cfg_cache = yaml.safe_load(f)
    return _cfg_cache

def get_app_config():
    return _load_cfg()

def get_model_provider(version: str):
    cfg = _load_cfg()
    reg = cfg.get("model_registry", {})
    if version in reg:
        return reg[version]
    if version == "rule-v1":
        return {"kind": "rule"}
    raise KeyError(f"Model version not found in registry: {version}")
