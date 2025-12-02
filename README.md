# VIP Lead Scoring with Eval & Docs

Hệ thống chấm điểm **lead B2B** dựa trên:

- Bộ dữ liệu lịch sử: `data/training/data.csv`
- Mô hình LLM (nếu được cấu hình)
- Bộ luật / thống kê rút ra từ dataset

Dự án gồm 2 phần chính:

1. **API FastAPI** (chạy bằng Docker)  
   → cung cấp các endpoint như: `/chat`, `/score`, `/nl_score`.
2. **Script đánh giá LLM**  
   → module `app.eval.evaluate_llm` để test độ chính xác mô hình trên file CSV.

---

## 1. Chuẩn bị môi trường

### 1.1. Yêu cầu

- Python 3.11+ (nếu chạy test local)
- Docker & Docker Compose (để chạy API)
- Dataset training:

  - File: `api/data/training/data.csv`
  - Khi chạy trong container, file này được Docker map vào `/app/data/training/data.csv`

---

## 2. Chạy nhanh API (FastAPI + Docker) – “chạy fast”

Phần này là hướng dẫn chạy backend nhanh bằng Docker.

### 2.1. Cấu trúc thư mục (rút gọn)

Ví dụ:

```text
vip_scoring_with_eval_and_docs/
└─ api/
   ├─ docker-compose.yml
   ├─ api/
   │  ├─ Dockerfile
   │  ├─ app/
   │  │  ├─ main.py
   │  │  ├─ agents/
   │  │  │   └─ model.py
   │  │  ├─ routers/
   │  │  │   ├─ chat.py
   │  │  │   ├─ score.py
   │  │  │   └─ nl_score.py
   │  │  ├─ config/
   │  │  │   └─ server_config.yaml
   │  │  └─ ...
   └─ data/
      └─ training/
         └─ data.csv
```

Lưu ý:

- Dockerfile nằm trong thư mục `api/api` (nơi có `app/main.py`).
- `docker-compose.yml` nằm tại `vip_scoring_with_eval_and_docs/api`.
- Thư mục `data/training` được mount vào trong container tại `/app/data/training`.

### 2.2. Ví dụ `docker-compose.yml` (rút gọn)

```yaml
version: "3.9"
services:
  api:
    build: ./api
    environment:
      - CONFIG_PATH=/app/config/server_config.yaml
      - REDIS_URL=redis://redis:6379/0
      - SAVE_LOGS=true
    ports:
      - "8080:8080"
    depends_on:
      - redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

> Nếu bạn đã mount volume cho `data/training`, đảm bảo bên trong container **thực sự** có file `/app/data/training/data.csv`.

### 2.3. Build & chạy API

Tại thư mục:

```bash
cd vip_scoring_with_eval_and_docs/api
```

Chạy:

```bash
docker-compose down         # dọn container cũ (nếu có)
docker-compose up --build -d
```

Kiểm tra:

```bash
docker ps -a
docker logs <ID_CONTAINER_API>
```

Nếu OK, mở:

- Swagger UI: http://localhost:8080/docs  
- OpenAPI JSON: http://localhost:8080/openapi.json  

---

## 3. Chạy test đánh giá LLM (evaluate_llm.py) – “chạy test”

Flow test thủ công, **không cần Docker**, dùng virtualenv.

### 3.1. Tạo virtualenv & cài thư viện

Tại thư mục:

```bash
cd vip_scoring_with_eval_and_docs/api
```

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Linux / macOS (bash)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.2. Tắt cấu hình LLM (chạy test offline)

Trong PowerShell (Windows), để tạm **không dùng LLM thật**:

```powershell
$env:LLM_API_KEY = $null
$env:LLM_BASE_URL = $null
$env:LLM_MODEL = $null
$env:LLM_TEMPERATURE = $null
```

### 3.3. Chạy script evaluate

Ví dụ lệnh đầy đủ:

```powershell
PS C:\Users\Admin> cd D:\HK9\DoAn\ddddd\vip_scoring_with_eval_and_docs\api
PS C:\Users\Admin\...\api> .\.venv\Scripts\Activate.ps1

$env:LLM_API_KEY = $null
$env:LLM_BASE_URL = $null
$env:LLM_MODEL = $null
$env:LLM_TEMPERATURE = $null

python -m app.eval.evaluate_llm `
  --max-rows 30 `
  --temp 0.3 `
  --output D:\HK9\DoAn\ddddd\vip_scoring_with_eval_and_docs\api\data\training\test\ketqua.csv
```

Giải thích tham số:

- `--max-rows 30` : số lượng bản ghi trong `data.csv` đem ra test.
- `--temp 0.3` : nhiệt độ khi gọi LLM (nếu đang dùng LLM thật).
- `--output ...ketqua.csv` : đường dẫn file CSV kết quả.

Kết quả:

- File CSV: `ketqua.csv` với các cột:
  - `name`
  - `qualification_status` (dự đoán)
  - `score`
  - `explain`
- Độ chính xác:
  - In ra console
  - Ghi thêm vào `model_accuracy.txt` (log accuracy theo thời gian)

---

## 4. Kiến trúc FastAPI & Agent (`app/agents/model.py`)

### 4.1. Các cột quan trọng

Tất cả thống kê & logic chấm điểm đều dựa vào đúng 5 cột:

```python
USEFUL_COLS = ["source", "status", "is_vip", "no_of_employees", "city"]
```

Cột nhãn mục tiêu:

- `"qualification_status"` với 3 giá trị:
  - `"Junk"`
  - `"Unqualified"`
  - `"Qualified"`

### 4.2. Luồng chấm điểm bằng dataset (giống `evaluate_llm.py`)

1. **Đọc dataset**

   - Hàm: `_load_dataset(path, target_col)`
   - Đảm bảo:
     - Có cột `qualification_status`
     - Loại bỏ nhãn không hợp lệ
     - Trả về `DataFrame` sạch.

2. **Thống kê tổng**

   - Hàm: `calculate_total_status_count(df, target_col="qualification_status")`
   - Trả về:

     ```python
     {
       "total_count": <int>,
       "qualification_status_count": {
         "Junk": <int>,
         "Unqualified": <int>,
         "Qualified": <int>
       }
     }
     ```

3. **Thống kê chi tiết theo 5 cột**

   - Hàm: `calculate_statistics(df, input_record)`
   - Với mỗi `column` trong `USEFUL_COLS`:
     - Lấy `target_value = input_record.get(column)`
     - Nếu không có value → bỏ qua cột.
     - Lọc dataset: `filtered_df = df[df[column] == target_value]`
     - Đếm:
       - `Junk`
       - `Unqualified`
       - `Qualified`
       - `total`
     - Tính % cho từng nhãn.
   - Trả về cấu trúc:

     ```python
     {
       "source": {
         "Facebook": {
           "Junk": ..., "Unqualified": ..., "Qualified": ..., "total": ...,
           "Junk_percent": ...,
           "Unqualified_percent": ...,
           "Qualified_percent": ...,
           "Junk_count": ...,
           "Unqualified_count": ...,
           "Qualified_count": ...
         }
       },
       "status": { ... },
       "is_vip": { ... },
       "no_of_employees": { ... },
       "city": { ... }
     }
     ```

4. **Build prompt gửi cho LLM**

   - Hàm: `build_model_prompt(statistics, input_record, total_status_count)`
   - Gồm:
     - JSON record của lead cần chấm
     - Thống kê tổng nhãn
     - Thống kê theo từng cột trong `USEFUL_COLS`
     - Luật quyết định:
       - Khi nào là `Qualified` / `Unqualified` / `Junk`
       - Mapping score:
         - 0–39 → Junk
         - 40–79 → Unqualified
         - 80–100 → Qualified
     - **Yêu cầu LLM**: chỉ trả về **1 JSON duy nhất / 1 dòng**:

       ```json
       {
         "score": 0-100,
         "qualification_status": "Junk|Unqualified|Qualified",
         "explain": "một câu giải thích ngắn"
       }
       ```

5. **Parse output LLM**

   - Hàm: `_parse_llm_output(reply)`
   - Tìm JSON trong chuỗi trả về.
   - Chuẩn hoá:
     - Nếu nhãn = `Qualified` → ép score ∈ [80, 100]
     - Nếu nhãn = `Unqualified` → ép score ∈ [40, 79]
     - Nếu nhãn = `Junk` → ép score ∈ [0, 39]
   - Nếu không parse được → fallback dựa trên score.

6. **Hàm chấm 1 lead cho agent**

   - Hàm: `score_lead(raw_record)`
   - Bên trong gọi `_score_with_dataset_or_fallback(raw_record)`:
     - Nếu có dataset:
       - Dùng thống kê + LLM (nếu `LLM_BASE_URL` được set).
       - Nếu LLM lỗi → fallback chọn nhãn bằng thống kê (votes).
     - Nếu không có dataset:
       - Dùng `heuristics_classification` (rules đơn giản).

   - Kết quả:

     ```python
     {
       "score": int,
       "qualification_status": str,
       "explain": str,
       "raw_reply": str  # rỗng nếu không gọi LLM
     }
     ```

7. **Chấm batch & lưu CSV**

   - Hàm: `score_rows_and_save(session_id, rows)`
   - Flow:
     - Chấm từng record → append vào list
     - Ghi file CSV:
       - Per-session: `/app/data/sessions/{session_id}_scored.csv`
       - Global: `/app/data/global/global_scored.csv`
     - Trả JSON:

       ```json
       {
         "saved_session_csv": "...",
         "updated_global_csv": "...",
         "rows_scored": 10,
         "results_preview": [ { ...5 dòng đầu... } ]
       }
       ```

---

## 5. Hướng đi FastAPI cho “source agent”

Tóm tắt flow cho phần **chat / agent**:

1. **Endpoint `/chat`**:
   - Nhận câu hỏi tự nhiên từ người dùng.
   - Dùng prompt trung gian để:
     - Phân biệt: đây là câu hỏi chấm điểm hay chỉ là hỏi thông tin bình thường.
     - Nếu là câu hỏi chấm điểm:
       - LLM trích ra 5 trường: `source`, `status`, `is_vip`, `no_of_employees`, `city`.
       - Tạo `input_record` từ các trường này.
       - Đọc `data/training/data.csv` → `df_csv`.
       - Gọi:
         - `calculate_statistics(df_csv, input_record)`
         - `calculate_total_status_count(df_csv)`
       - Build prompt: `build_model_prompt(...)`
       - Gọi LLM → `_parse_llm_output(...)`
       - Trả lại cho client: score, nhãn, giải thích.
     - Nếu KHÔNG phải câu hỏi chấm điểm:
       - Chỉ cần gọi LLM trả lời tự do, hoặc routing sang agent khác tuỳ bạn.

2. **Endpoint `/nl_score`**:
   - Nhận câu mô tả lead *bằng ngôn ngữ tự nhiên*.
   - Gọi `extract_info_from_text(text)`:
     - LLM trả về JSON chứa 5 trường chính.
   - Gọi `score_lead(record_trích_xuất)`:
     - Dùng dataset + luật + (tuỳ chọn) LLM để quyết định nhãn & điểm.

3. **Endpoint `/score`**:
   - Dùng cho batch score (list các record).
   - Gọi `score_rows_and_save(session_id, rows)`.

Với kiến trúc này:

- `evaluate_llm.py` dùng cùng bộ hàm thống kê như agent → dễ so sánh và tune.
- API FastAPI chạy “fast” qua Docker, chỉ cần bạn build và đưa đúng `data.csv`.
- Bạn có thể mở rộng thêm router / agent mới chỉ bằng cách:
  - Import `score_lead`, `calculate_statistics`, `calculate_total_status_count`.
  - Thiết kế thêm prompt riêng cho từng use case.
