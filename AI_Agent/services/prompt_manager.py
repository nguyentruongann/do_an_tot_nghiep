import os, yaml, json
from jinja2 import Template
from adapters import registry as adapter_registry

def _load_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_prompt_for(data_type: str | None, payload: dict) -> dict:
    # 1) resolve adapter
    adapter_key = data_type or adapter_registry.autodetect(payload)
    adapter = adapter_registry.get(adapter_key)
    if adapter is None:
        raise KeyError(f"No adapter for data_type='{adapter_key}'")

    # 2) build context from adapter
    context = adapter.build_context(payload)

    # 3) load prompt pack (template + fewshot) for data_type or fallback to generic
    pack_dir = os.path.join("prompts", "packs", adapter_key)
    tpl_path = os.path.join(pack_dir, "scorer.j2")
    if not os.path.exists(tpl_path):
        tpl_path = os.path.join("prompts", "templates", "lead_scorer_generic.j2")

    fewshot_path = os.path.join(pack_dir, "fewshot.yaml")
    fewshot = _load_yaml(fewshot_path) if os.path.exists(fewshot_path) else []

    tpl = Template(_load_file(tpl_path))

    # 4) render prompt string
    prompt = tpl.render(
        payload=context["payload"],
        derived=context.get("derived") or {},
        schema=context.get("schema") or {},
        industry=context.get("industry"),
        market=context.get("market"),
        fewshot=fewshot
    )

    # 5) return the inputs needed by the LLM scorer
    return {
        "adapter": adapter_key,
        "prompt": prompt,
        "context": context,
        "fewshot_used": bool(fewshot),
    }
