# VIP Scoring API (Fixed)

- Trả lời LLM **luôn bằng tiếng Việt** (trừ khi người dùng yêu cầu ngôn ngữ khác).
- Redis được dùng cho:
  - Lưu lịch sử hội thoại và log sự kiện theo `session_id`.
  - Cache kết quả chấm điểm (`/score`) theo `input_hash` để tăng tốc.
- Cấu hình thông qua `api/config/server_config.yaml` hoặc biến môi trường.

## Chạy nhanh
```bash
docker compose up --build
# API: http://localhost:8080/docs
```
