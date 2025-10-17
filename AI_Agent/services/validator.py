from typing import Dict, Tuple, List, Any

def validate_minimal(p: Dict[str, Any]) -> Tuple[bool, List[str]]:
    # LLM-first: chỉ cần có vài field cốt lõi để hiểu ngữ cảnh; để adapter quyết tâm tối thiểu
    errs = []
    if not isinstance(p, dict):
        errs.append("payload_not_object")
    return (len(errs) == 0, errs)
