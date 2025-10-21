from .bank_marketing import BankMarketingAdapter

_REGISTRY = {
    "bank_marketing": BankMarketingAdapter(),
}

def get(key: str):
    return _REGISTRY.get(key)

def autodetect(payload: dict) -> str:
    # heuristics: if has fields from UCI bank marketing
    fields = set(map(str.lower, payload.keys()))
    if {"age","job","balance","duration","campaign","pdays","previous"}.issubset(fields):
        return "bank_marketing"
    return "bank_marketing"  # default

# Thêm vào cuối file adapters/__init__.py
class Registry:
    def __init__(self):
        self._registry = _REGISTRY
    
    def get(self, key: str):
        return _REGISTRY.get(key)
    
    def autodetect(self, payload: dict) -> str:
        fields = set(map(str.lower, payload.keys()))
        if {"age","job","balance","duration","campaign","pdays","previous"}.issubset(fields):
            return "bank_marketing"
        return "bank_marketing"

registry = Registry()