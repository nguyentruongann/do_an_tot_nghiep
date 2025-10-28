# Multi-Agent Scoring Suite v4 (STRICT SCORING CHAT)

- Chỉ **chào hỏi** và **trả lời câu hỏi liên quan đến chấm điểm**. Off-topic → từ chối.
- Phiên **riêng tư**: dữ liệu upload + scoring chỉ sống trong `sessions/` theo cookie `sid`.

## Chạy
```bash
docker compose up -d --build
# UI
http://localhost:8090
# API
http://localhost:8080/health
# n8n
http://localhost:5678
```

## API chính (session-private)
- `POST /session/new` – tạo phiên; set cookie `sid`
- `POST /upload_session` – upload file (CSV/XLSX/Parquet) cho phiên
- `POST /score_session` – chấm điểm (không ghi kho chung)
- `GET  /download_scored_session` – tải CSV kết quả của phiên
- `POST /chat` – chỉ trả lời câu hỏi liên quan scoring; off-topic bị chặn
- `GET  /logs?session_id=...` – xem log (lưu trong Redis)

## UI (port 8090)
- Không lộ dataset_id. Người dùng chỉ: **tạo phiên → upload data → chat → (tuỳ chọn) chấm điểm → tải CSV**.

## n8n testing (riêng)
- Đặt data cố định tại `testing/data_test/BankCustomerData.csv`.
- Import `testing/workflow_n8n_llm_benchmark.json` và `testing/workflow_api_benchmark.json`.
- Kết quả CSV nằm trong `testing/results_*.csv`.

## Đổi model LLM
`api/config/server_config.yaml`:
```yaml
llm:
  base_url: "http://<host>:<port>/v1/"
  model: "<model>"
  api_key: "sk-..."
```
Rồi: `docker compose up -d --build api`.