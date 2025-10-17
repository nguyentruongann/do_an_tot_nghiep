# Lead Agent Pro (LLM-first, Prompt Packs, Multi-dataset Adapters)

## Điểm nổi bật
- **LLM-first scoring**: chấm điểm bằng prompt theo từng loại dữ liệu (prompt pack).
- **Adapters**: ánh xạ nhiều schema khác nhau -> context thống nhất cho prompt.
- **Prompt Packs (YAML/Jinja)**: tách template + few-shot theo `data_type`.
- **Model by YAML**: `config/runtime.yaml` chọn endpoint (Open-source model via FastAPI/vLLM/Ollama/etc.).
- **Fallback**: nếu LLM lỗi -> fallback rule tối giản (có thể tắt).

## Quickstart
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

### Cấu hình
Sửa `config/runtime.yaml`:
- `app.db_path`: trỏ DB SQLite (vd do_an.db trên DBeaver)
- `app.model_version`: `"llm-v1"` (mặc định)
- `model_registry.llm-v1.endpoint`: URL API model open-source

### Gọi thử
```bash
curl -X POST http://localhost:8000/v1/score -H "Content-Type: application/json" -d @examples/bank_marketing.json
```

### Batch từ DB (bank_marketing)
```bash
python jobs/batch_scoring.py --data_type bank_marketing --limit 500
```

## Cấu trúc
- `adapters/` — chuyển đổi payload theo `data_type` → context prompt
- `prompts/packs/<data_type>/` — `scorer.j2` + `fewshot.yaml`
- `services/prompt_manager.py` — ráp prompt từ template + fewshot + context
- `models/scorer_llm.py` — gọi API LLM theo YAML
