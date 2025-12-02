import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
from collections import defaultdict
import re

"""
Evaluate a language model on a CSV dataset.  This script has been updated to use
absolute imports for ``load_config`` and ``chat_completion`` so that it can be
executed as a standalone module without relying on a parent package.  The
original functionality and algorithm remain unchanged.
"""

from ..config import load_config
from ..llm import chat_completion  # Cần có hàm này từ LLM

TARGET_COL = "qualification_status"
RATE_LIMIT_SLEEP_SECONDS = 65
MAX_RATE_LIMIT_RETRIES = 20

USEFUL_COLS = [
    "source", "status", "is_vip", "no_of_employees", "city"
]

@dataclass
class EvalMetrics:
    timestamp: str
    model_name: str
    base_url: str
    temperature: float
    total: int
    correct: int
    accuracy: float
    target_col: str
    labels: List[str]
    per_label_accuracy: Dict[str, float]
    train_csv: str
    val_csv: str
    test_csv: str

def _resolve_data_path() -> str:
    """Trả về đường dẫn tới file CSV chứa dữ liệu."""
    return r"D:\HK9\DoAn\ddddd\vip_scoring_with_eval_and_docs\api\data\training\data.csv"

def _load_dataset(path: str, target_col: str) -> pd.DataFrame:
    """Đọc CSV và đảm bảo có cột nhãn."""
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"File {path} không có cột '{target_col}'")
    df[target_col] = df[target_col].astype(str).str.strip()
    df = df[df[target_col] != ""]
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    valid_labels = {"Junk", "Qualified", "Unqualified"}
    df = df[df[target_col].isin(valid_labels)].reset_index(drop=True)

    return df

def calculate_total_status_count(df: pd.DataFrame) -> dict:
    """Thống kê tổng số lượng các giá trị trong qualification_status."""
    status_count = df['qualification_status'].value_counts().to_dict()
    total_count = len(df)
    # Trả về số lượng các giá trị trong qualification_status
    return {
        "total_count": total_count,
        "qualification_status_count": status_count
    }

def calculate_statistics(df: pd.DataFrame, input_record: Dict[str, Any]) -> dict:
    """Tính toán tỷ lệ phần trăm của qualification_status cho mỗi giá trị trong các cột hữu ích và số lượng."""
    statistics = {}
    for column in USEFUL_COLS:
        if column not in df.columns:
            continue
        column_stats = defaultdict(lambda: {"Junk": 0, "Unqualified": 0, "Qualified": 0, "total": 0})

        # Lọc dữ liệu theo giá trị trong cột từ dòng đầu vào
        target_value = input_record[column]
        filtered_df = df[df[column] == target_value]

        for _, row in filtered_df.iterrows():
            label = row['qualification_status']
            column_stats[target_value][label] += 1
            column_stats[target_value]["total"] += 1  # Tổng số bản ghi cho giá trị này trong cột

        # Tính tỷ lệ phần trăm và số lượng cho mỗi giá trị trong cột
        for value, stats in column_stats.items():
            total = stats["total"]
            if total > 0:
                stats["Junk_percent"] = (stats["Junk"] / total) * 100
                stats["Unqualified_percent"] = (stats["Unqualified"] / total) * 100
                stats["Qualified_percent"] = (stats["Qualified"] / total) * 100
                stats["Junk_count"] = stats["Junk"]
                stats["Unqualified_count"] = stats["Unqualified"]
                stats["Qualified_count"] = stats["Qualified"]

        statistics[column] = column_stats

    return statistics

def build_input_record(row: pd.Series) -> Dict[str, Any]:
    """Tạo record đầu vào để gửi cho LLM, loại bỏ cột 'qualification_status' và 'final_score'."""
    # Loại bỏ cột 'qualification_status' và 'final_score'
    record = row.drop([TARGET_COL, "final_score"], errors="ignore").to_dict()
    return record

def build_model_prompt(statistics: Dict[str, Any], input_record: Dict[str, Any], total_status_count: dict) -> str:
    """Xây dựng nội dung prompt gửi tới LLM.

    Prompt bao gồm thống kê tóm tắt cho lead và hướng dẫn chọn nhãn. Ở cuối
    prompt, yêu cầu LLM chỉ trả về duy nhất một đối tượng JSON một dòng
    chứa ba trường: ``score`` (0–100), ``qualification_status`` (Junk,
    Unqualified, Qualified) và ``explain`` (lý do ngắn gọn). Không trả
    về thêm bất kỳ văn bản nào ngoài JSON này.
    """
    # Phần giới thiệu và thống kê tổng hợp
    prompt = f"\nDữ liệu của lead cần chấm:\n{json.dumps(input_record, ensure_ascii=False)}\n"
    prompt += (
        "Bạn là một hệ thống chấm điểm chất lượng lead chuyên nghiệp và thẳng thắn.\n\n"
        "Dưới đây là thông tin thống kê cho lead bạn hãy *bám sát vào thông tin và mô tả sau* để xác định nhãn chính xác:\n"
    )
    # Tổng số bản ghi trong qualification_status
    prompt += (
        f"Tổng số bản ghi trong qualification_status:\n"
        f"Junk: {total_status_count['qualification_status_count'].get('Junk', 0)}\n"
        f"Unqualified: {total_status_count['qualification_status_count'].get('Unqualified', 0)}\n"
        f"Qualified: {total_status_count['qualification_status_count'].get('Qualified', 0)}\n"
        f"Tổng cộng: {total_status_count['total_count']}\n\n"
    )
    # Thống kê từng cột hữu ích
    for column, column_stats in statistics.items():
        if column in USEFUL_COLS:
            for value, stats in column_stats.items():
                if value == "total":
                    continue
                prompt += (
                    f"- Tỷ lệ {value} trong cột {column}: "
                    f"Junk = {stats['Junk_percent']}% ({stats['Junk_count']} bản ghi), "
                    f"Unqualified = {stats['Unqualified_percent']}% ({stats['Unqualified_count']} bản ghi), "
                    f"Qualified = {stats['Qualified_percent']}% ({stats['Qualified_count']} bản ghi)\n"
                )
    # Hướng dẫn ra quyết định và yêu cầu đầu ra
    prompt += """
    Dựa trên thông tin thống kê trên, bạn hãy:
    1. Dựa vào tần suất và số lượng các nhãn so với nhãn tổng trong các lead tương tự để xác định nhãn cuối cùng cho lead này (Junk, Unqualified, Qualified).
    2. Nhãn cuối cùng là **Qualified** khi is_vip là 1 hoặc **% Qualified** lệch so với %Junk hoặc %Unqualified trên 8% và số lượng lệch từ 3 cột trở lên mới có thể được phân loại là **Qualified**.
    2. Nếu xác định **không phải là Qualified**, bạn cần **xem xét kỹ** số lượng các cột có tỷ lệ % nhãn Junk hay Unqualified cao hơn để đưa ra quyết định cuối cùng.
    3. Đưa ra điểm số cuối cùng cho lead này từ 0-100.
    
    Mức điểm:
    - **0-39**: Nhãn **"Junk"** – Lead này không đủ điều kiện, hoặc có khả năng chuyển đổi rất thấp. Có thể là lead không có giá trị hoặc thông tin không chính xác.
    - **40-79**: Nhãn **"Unqualified"** – Lead này chưa đủ điều kiện nhưng có khả năng cải thiện và có thể trở thành khách hàng tiềm năng trong tương lai.
    - **80-100**: Nhãn **"Qualified"** – Lead này đủ điều kiện và có khả năng cao để chuyển đổi thành khách hàng. Đây là các lead tiềm năng có thể chuyển đổi thành doanh thu.
    Đầu ra yêu cầu:
    - "score": số nguyên từ 0 đến 100.
    - "qualification_status": một trong các nhãn "Junk", "Unqualified", "Qualified".
    - "explain": một câu giải thích ngắn về lý do phân loại và chấm điểm.
    Chỉ trả về duy nhất một JSON một dòng chứa ba trường: 
        {\"score\": <số nguyên từ 0 đến 100>, \"qualification_status\": \"Junk|Unqualified|Qualified\", \"explain\": \"lý do ngắn gọn\"}. 
        Không trả về thêm bất kỳ văn bản nào ngoài JSON này.
    """
    # Dữ liệu của lead cần chấm
    return prompt


def _parse_llm_output(reply: str) -> Tuple[int, str, str]:
    default_score = 50
    default_label = "Unqualified"
    # Đầu tiên thử trích xuất JSON nếu có
    try:
        m = re.search(r"\{.*\}", reply, flags=re.S)
        if m:
            raw_json = m.group(0)
            data = json.loads(raw_json)
            score_raw = data.get("score", default_score)
            qualification_status = data.get("qualification_status", default_label)
            explain = data.get("explain", "")
            return int(score_raw), qualification_status, explain
    except Exception:
        # bỏ qua lỗi parse JSON, sẽ fallback bên dưới
        pass

    # Nếu không tìm thấy JSON, parse theo định dạng dòng "Đầu ra" của LLM.
    # Mặc định nếu không trích xuất được gì
    score = default_score
    qualification_status = default_label
    explain = ""
    try:
        # Tìm giá trị score: chấp nhận ký tự số sau dấu ':'
        score_match = re.search(r'["\']?score["\']?\s*[:：]\s*([0-9]{1,3})', reply)
        if score_match:
            score = int(score_match.group(1))
        # Tìm giá trị qualification_status: trích xuất tới hết từ hoặc tới dấu xuống dòng
        status_match = re.search(r'["\']?qualification_status["\']?\s*[:：]\s*["\']?([^\"\'\n]+)', reply)
        if status_match:
            qualification_status = status_match.group(1).strip().strip('"\' \,')
        # Tìm giá trị explain: trích xuất phần trong dấu nháy kép
        explain_match = re.search(r'["\']?explain["\']?\s*[:：]\s*["\'](.+?)["\']', reply, flags=re.S)
        if explain_match:
            explain = explain_match.group(1).strip()
    except Exception:
        pass
    # Nếu không tìm thấy explanation, ghi lại toàn bộ reply làm explain để debug
    if not explain:
        explain = f"LLM trả về khó parse: {reply}"
    return score, qualification_status, explain

def evaluate_llm_on_csv(
    max_rows: Optional[int] = 30,
    temperature: Optional[float] = None,
    seed: int = 42,
    output_path: Optional[str] = None,
    accuracy_log_path: str = "model_accuracy.txt",  # Đường dẫn tới file ghi accuracy
) -> Dict[str, Any]:
    """
    Chạy đánh giá LLM trên file CSV. Hàm này sẽ in ra kết quả dự đoán của từng mẫu,
    lưu toàn bộ kết quả vào file CSV (nếu ``output_path`` được cung cấp) và trả về
    các metrics như độ chính xác.

    :param max_rows: Số lượng bản ghi sẽ đánh giá.
    :param temperature: Tham số nhiệt độ cho LLM. Nếu None sẽ dùng 0.7.
    :param seed: Giá trị seed để chọn mẫu ngẫu nhiên.
    :param output_path: Đường dẫn tới file CSV để lưu kết quả. Nếu None, sẽ
                        lưu trong thư mục làm việc hiện tại với tên
                        ``llm_evaluation_results.csv``.
    :param accuracy_log_path: Đường dẫn tới file để lưu độ chính xác.
    :return: Dict chứa các thông số đánh giá.
    """
    df = _load_dataset(_resolve_data_path(), TARGET_COL)
    correct = 0
    test_rows: List[Dict[str, Any]] = []
    total_by_label: Dict[str, int] = {}
    correct_by_label: Dict[str, int] = {}
    temp = temperature if temperature is not None else 0.7

    df_eval = df.sample(n=max_rows, random_state=seed)
    total = len(df_eval)

    for idx, (_, row) in enumerate(df_eval.iterrows(), start=1):
        y_true = row[TARGET_COL]
        input_record = build_input_record(row)
        statistics = calculate_statistics(df, input_record)
        total_status_count = calculate_total_status_count(df)
        prompt = build_model_prompt(statistics, input_record, total_status_count)

        print(f"\n=============== Mẫu {idx} ===============")
        print("Prompt gửi tới LLM:")
        print(prompt)

        raw_reply: Optional[str] = None
        attempt = 0
        while attempt < MAX_RATE_LIMIT_RETRIES:
            try:
                raw_reply = chat_completion([{"role": "user", "content": prompt}], temperature=temp)
                print("Phản hồi gốc từ LLM:")
                print(raw_reply)
                break
            except Exception as e:
                msg = str(e)
                if "rate limit" in msg.lower() or "429" in msg:
                    attempt += 1
                    print(f"Bị rate limit (lần thử {attempt}), chờ {RATE_LIMIT_SLEEP_SECONDS}s...")
                    time.sleep(RATE_LIMIT_SLEEP_SECONDS)
                    continue
                raw_reply = None
                print(f"Lỗi khi gọi LLM: {e}")
                break

        if raw_reply is None:
            print("Không nhận được phản hồi từ LLM cho mẫu này. Bỏ qua.")
            continue

        score_val, y_pred, explain = _parse_llm_output(raw_reply)
        print("Kết quả dự đoán:")
        print(f"score: {score_val}, qualification_status: {y_pred}, explain: {explain}")

        out_record = {
            "name": row.get("name"),
            "qualification_status": y_pred,
            "score": score_val,
            "explain": explain,
        }
        test_rows.append(out_record)

        if isinstance(y_true, str):
            key_true = y_true.strip().lower()
            total_by_label[key_true] = total_by_label.get(key_true, 0) + 1
            if isinstance(y_pred, str):
                if y_pred.strip().lower() == key_true:
                    correct += 1
                    correct_by_label[key_true] = correct_by_label.get(key_true, 0) + 1

    accuracy = correct / total if total > 0 else 0.0
    per_label_accuracy: Dict[str, float] = {}
    for label_key, total_count in total_by_label.items():
        correct_count = correct_by_label.get(label_key, 0)
        per_label_accuracy[label_key.capitalize()] = correct_count / total_count if total_count > 0 else 0.0

    if output_path is None:
        output_path = os.path.join(os.getcwd(), "llm_evaluation_results.csv")

    try:
        df_output = pd.DataFrame(test_rows, columns=["name", "qualification_status", "score", "explain"])
        df_output.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\nĐã lưu kết quả đánh giá vào file: {output_path}")
    except Exception as exc:
        print(f"\nKhông thể ghi file kết quả: {exc}")

    # Ghi độ chính xác vào file model_accuracy.txt
    try:
        with open(accuracy_log_path, "a") as accuracy_file:
            accuracy_file.write(f"{datetime.utcnow().isoformat()} - Accuracy: {accuracy * 100:.2f}%\n")
        print(f"\nĐã lưu độ chính xác vào file: {accuracy_log_path}")
    except Exception as exc:
        print(f"\nKhông thể ghi độ chính xác vào file: {exc}")

    metrics = EvalMetrics(
        timestamp=datetime.utcnow().isoformat(),
        model_name="LLM Model",
        base_url="URL",
        temperature=temp,
        total=total,
        correct=correct,
        accuracy=accuracy,
        target_col=TARGET_COL,
        labels=list(df[TARGET_COL].unique()),
        per_label_accuracy=per_label_accuracy,
        train_csv=_resolve_data_path(),
        val_csv=_resolve_data_path(),
        test_csv=_resolve_data_path(),
    )

    return asdict(metrics)


def main():
    parser = argparse.ArgumentParser(description="Đánh giá LLM trên file CSV và lưu kết quả")
    parser.add_argument("--max-rows", type=int, default=30, help="Số lượng mẫu để đánh giá")
    parser.add_argument("--temp", type=float, default=None, help="Nhiệt độ khi gọi LLM")
    parser.add_argument("--seed", type=int, default=42, help="Seed để chọn mẫu ngẫu nhiên")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn tới file CSV để lưu kết quả. Mặc định là llm_evaluation_results.csv trong thư mục hiện tại.",
    )
    args = parser.parse_args()

    metrics = evaluate_llm_on_csv(
        max_rows=args.max_rows,
        temperature=args.temp,
        seed=args.seed,
        output_path=args.output,
    )

    # In ra metrics sau khi hoàn thành đánh giá
    print("\n============ Tổng kết đánh giá ============")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
