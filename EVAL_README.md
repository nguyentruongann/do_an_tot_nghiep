
# Hướng dẫn chạy API chính và test hiệu suất LLM

Repo này gồm 2 phần chính:

- **API FastAPI** (dịch vụ chấm điểm / chat trên CSV)
- **Script test LLM** (đánh giá độ chính xác của model trên bộ dữ liệu `leads_erpnext_100k.csv`)

Tất cả đều dùng chung file config: `api/config/server_config.yaml`.


## 1. Chuẩn bị môi trường

### 1.1. Cài đặt dependency (chạy local, không Docker)

```bash
cd api
pip install -r requirements.txt
```

### 1.2. Biến môi trường cấu hình

App dùng `CONFIG_PATH` để biết file cấu hình:

```bash
# tại thư mục gốc repo
export CONFIG_PATH=api/config/server_config.yaml
```

Trong file `api/config/server_config.yaml` cần có cấu hình:

```yaml
llm:
  base_url: "https://api.groq.com/openai/v1"
  api_key: "YOUR_API_KEY"
  model: "llama-3.3-70b-versatile"
  temperature: 0.2

server:
  host: "0.0.0.0"
  port: 8080

storage:
  logs_key_prefix: "logs"
  cache_key_prefix: "cache"
  ttl_seconds: 259200

training:
  data_path: "/app/data/training/leads_erpnext_100k.csv"
```

> Lưu ý:
> - Khi chạy **Docker**, `/app` là thư mục root trong container, path `/app/data/...` sẽ hoạt động.
> - Khi chạy **local**, script test sẽ tự map về `api/data/training/leads_erpnext_100k.csv` nếu không tìm thấy path `/app/...`.


## 2. Chạy API chính (FastAPI)

Có 2 cách: Docker hoặc chạy trực tiếp bằng uvicorn.

### 2.1. Dùng Docker / docker-compose (khuyến nghị)

Tại thư mục gốc repo (nơi có `docker-compose.yml`):

```bash
docker-compose up --build
```

Lệnh này sẽ:

- Build image từ thư mục `api/`
- Chạy container:
  - API (FastAPI)
  - Redis (dùng cache / logs)

Sau khi service chạy xong, API lắng nghe ở:

- `http://localhost:8080/`

Các endpoint chính:

- `GET /`  
  Check sức khoẻ service. Trả về JSON:

  ```json
  { "ok": true, "service": "vip-scoring", "lang": "vi" }
  ```

- `POST /score`  
  Chấm điểm lead dựa trên CSV được index.

- `POST /chat`  
  Hỏi đáp / tóm tắt thông tin dựa trên dữ liệu CSV.

- `GET /logs`  
  Xem logs (nếu bạn bật lưu log bằng biến môi trường tương ứng).

### 2.2. Chạy trực tiếp bằng uvicorn (local, không Docker)

```bash
cd api
export CONFIG_PATH=./config/server_config.yaml
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Khi đó API cũng chạy ở `http://localhost:8080` tương tự Docker.


## 3. Script test hiệu suất LLM trên CSV

Script test nằm ở:

```text
api/app/eval/evaluate_llm.py
```

Mục tiêu:

- Dùng **cùng model LLM** mà API đang dùng (đọc từ `server_config.yaml`).
- Chạy model trên tập **test** được tách ra từ CSV `leads_erpnext_100k.csv`.
- Tính toán:
  - `accuracy` tổng (tỉ lệ đúng)
  - `per_label_accuracy` (độ chính xác theo từng nhãn)
- Lưu lại kết quả (gồm **tên model**, base_url, temperature, accuracy, ...) vào file CSV log:
  - `llm_eval_results.csv` (cùng thư mục với file data).


### 3.1. Cách script tìm file CSV dữ liệu

Field `training.data_path` trong `server_config.yaml` trỏ tới file gốc, ví dụ:

```yaml
training:
  data_path: "/app/data/training/leads_erpnext_100k.csv"
```

Logic của script:

1. Thử đúng path cấu hình (`/app/data/...`).
2. Nếu không có, thử map sang layout repo local:
   - `api/data/training/leads_erpnext_100k.csv`
3. Nếu vẫn không tìm thấy → báo lỗi `FileNotFoundError`.

Bạn chỉ cần đảm bảo:

- Khi chạy Docker:
  - Mount volume sao cho file thật nằm ở `/app/data/training/leads_erpnext_100k.csv`.
- Khi chạy local:
  - Copy file về `api/data/training/leads_erpnext_100k.csv`
  - Hoặc sửa `training.data_path` trỏ tới path thật.


### 3.2. Chia train / val / test

Script sẽ:

1. Đọc toàn bộ CSV gốc.
2. Giả định cột nhãn là:

   ```python
   TARGET_COL = "qualification_status"
   ```

   > Nếu bạn muốn dùng cột khác (vd: `status`), sửa trực tiếp hằng số này trong `evaluate_llm.py`.

3. Shuffle dữ liệu với `seed` (mặc định 42 cho bước chia data).
4. Chia theo tỉ lệ:
   - train: 70%
   - val:   15%
   - test:  15%
5. Lưu thành 3 file mới (cùng thư mục với CSV gốc):

   - `leads_erpnext_100k_train.csv`
   - `leads_erpnext_100k_val.csv`
   - `leads_erpnext_100k_test.csv`


### 3.3. Chạy test

Từ thư mục `api/`:

```bash
cd api
export CONFIG_PATH=./config/server_config.yaml   # nếu chưa set
python -m app.eval.evaluate_llm --max-rows 500 --temp 0.2 --seed 123
```

Tham số:

- `--max-rows`:
  - Mặc định `200`.
  - Nếu giá trị **< 0** (vd `-1`) → dùng toàn bộ test set.
- `--temp`:
  - Ghi đè temperature cho LLM (nếu không truyền → dùng `llm.temperature` trong config).
- `--seed`:
  - Seed cho việc shuffle / sample test subset (giúp run lặp lại được).

Ví dụ chạy full test set với temperature lấy từ config:

```bash
python -m app.eval.evaluate_llm --max-rows -1
```


### 3.4. Kết quả test trả về những gì?

Khi chạy xong, script sẽ:

1. **In JSON ra console**, ví dụ:

   ```json
   {
     "timestamp": "2025-11-24T10:15:30.123456",
     "model_name": "llama-3.3-70b-versatile",
     "base_url": "https://api.groq.com/openai/v1",
     "temperature": 0.2,
     "total": 200,
     "correct": 150,
     "accuracy": 0.75,
     "target_col": "qualification_status",
     "labels": ["Qualified", "Unqualified", "Junk"],
     "per_label_accuracy": {
       "Qualified": 0.8,
       "Unqualified": 0.7,
       "Junk": 0.6
     },
     "train_csv": "/app/data/training/leads_erpnext_100k_train.csv",
     "val_csv": "/app/data/training/leads_erpnext_100k_val.csv",
     "test_csv": "/app/data/training/leads_erpnext_100k_test.csv",
     "splits_csv": {
       "train": "/app/data/training/leads_erpnext_100k_train.csv",
       "val": "/app/data/training/leads_erpnext_100k_val.csv",
       "test": "/app/data/training/leads_erpnext_100k_test.csv"
     }
   }
   ```

2. **Append thêm 1 dòng vào file log CSV**:

   - File: `llm_eval_results.csv`
   - Vị trí: cùng thư mục với CSV gốc (ví dụ `/app/data/training/llm_eval_results.csv`).

   Các cột chính:

   | Cột               | Ý nghĩa                                                  |
   |-------------------|---------------------------------------------------------|
   | `timestamp`       | Thời điểm chạy test (UTC)                              |
   | `model_name`      | Tên model từ `llm.model` trong `server_config.yaml`    |
   | `base_url`        | Base URL LLM                                            |
   | `temperature`     | Temperature sử dụng cho lần test này                   |
   | `total`           | Số mẫu test                                             |
   | `correct`         | Số mẫu dự đoán đúng                                     |
   | `accuracy`        | Độ chính xác tổng (0–1)                                 |
   | `target_col`      | Tên cột nhãn (vd `qualification_status`)               |
   | `labels`          | Danh sách các nhãn (JSON string)                       |
   | `per_label_accuracy` | Độ chính xác theo từng nhãn (JSON string)          |
   | `train_csv`       | Đường dẫn file train                                    |
   | `val_csv`         | Đường dẫn file val                                      |
   | `test_csv`        | Đường dẫn file test                                     |

   Bạn có thể mở `llm_eval_results.csv` bằng Excel hoặc pandas để so sánh:

   - Model khác nhau (chỉ cần đổi `llm.model` trong config rồi chạy lại script).
   - Cùng model nhưng cấu hình temperature khác.
   - Các lần run khác nhau theo thời gian.


## 4. Phân biệt rõ “chạy chính” và “chạy test”

- **Chạy chính (API):**
  - Dùng `docker-compose up` hoặc `uvicorn app.main:app ...`
  - Phục vụ traffic thật / client thật.
  - Không tự động đánh giá accuracy.

- **Chạy test (đánh giá hiệu suất mô hình LLM):**
  - Dùng: `python -m app.eval.evaluate_llm ...`
  - Đọc đúng **model** từ `server_config.yaml`.
  - Chạy trên tập test sinh ra từ CSV.
  - Ghi log kết quả vào `llm_eval_results.csv` để tiện theo dõi về sau.

Bạn chỉ cần nhớ:

- **Đổi model** → sửa `llm.model` trong `api/config/server_config.yaml`.
- Sau đó:
  - **Run API** bình thường.
  - **Run script test** để đo hiệu suất và lưu lại.
