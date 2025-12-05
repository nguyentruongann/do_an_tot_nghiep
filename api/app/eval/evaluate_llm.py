import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from itertools import combinations
import re

import pandas as pd

"""
Evaluate a language model on a CSV dataset.

- Sử dụng data gốc (data.csv) để thống kê lịch sử (df_train).
- Sử dụng file test riêng (llm_eval_test_100.csv) để đo accuracy của LLM (df_eval).
- Lệnh chạy giữ nguyên:
  python -m app.eval.evaluate_llm --max-rows 5 --temp 0.2 --output <path_to_result.csv>
"""

from ..config import load_config
from ..llm import chat_completion  # Hàm gọi LLM

TARGET_COL = "qualification_status"
RATE_LIMIT_SLEEP_SECONDS = 65
MAX_RATE_LIMIT_RETRIES = 20

USEFUL_COLS = [
    "source",
    "status",
    "is_vip",
    "no_of_employees",
    "city",
]


# ============================================================
# Dataclass lưu metrics
# ============================================================

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


# ============================================================
# Đường dẫn dữ liệu
# ============================================================

def _resolve_data_path() -> str:
    """Đường dẫn tới CSV train / full data."""
    return r"D:\do_an_tot_nghiep\api\data\training\data.csv"


def _resolve_test_path() -> str:
    """Đường dẫn tới CSV test (100 dòng để evaluation)."""
    return r"D:\do_an_tot_nghiep\api\data\training\llm_eval_test_100.csv"


# ============================================================
# Load dataset & thống kê global
# ============================================================

def _load_dataset(path: str, target_col: str) -> pd.DataFrame:
    """Đọc CSV và đảm bảo có cột nhãn hợp lệ."""
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
    status_count = df["qualification_status"].value_counts().to_dict()
    total_count = len(df)
    return {
        "total_count": total_count,
        "qualification_status_count": status_count,
    }


# ============================================================
# Thống kê combos cho 1 lead (KHÔNG dùng single nữa)
# ============================================================

def calculate_statistics(df: pd.DataFrame, input_record: Dict[str, Any]) -> dict:
    """
    Tính thống kê SỐ LƯỢNG nhãn cho các tổ hợp 2..5 cột (combos),
    lọc đúng cấu hình của lead.

    Trả về:
    {
        "combos": [
            {
                "features": [...],
                "values": [...],
                "total": ...,
                "Junk": ...,
                "Unqualified": ...,
                "Qualified": ...
            },
            ...
        ]
    }
    """

    MAX_COMBO_SIZE = 5   # tổ hợp tối đa 5 cột
    MIN_TOTAL = 3        # tổ hợp có < 3 bản ghi thì bỏ qua

    cols = [
        c for c in USEFUL_COLS
        if c in df.columns and c in input_record and pd.notna(input_record[c])
    ]

    def count_labels(filtered: pd.DataFrame) -> Dict[str, int]:
        """Đếm số lượng từng nhãn trong qualification_status."""
        counts = {"Junk": 0, "Unqualified": 0, "Qualified": 0}
        vc = filtered["qualification_status"].value_counts()
        for label in counts.keys():
            counts[label] = int(vc.get(label, 0))
        return counts

    statistics: Dict[str, Any] = {
        "combos": [],
    }

    # Chỉ tính combos (2..5 cột)
    max_k = min(MAX_COMBO_SIZE, len(cols))
    for k in range(2, max_k + 1):
        for subset in combinations(cols, k):
            mask = pd.Series(True, index=df.index)
            for c in subset:
                mask &= (df[c] == input_record[c])

            filtered_df = df[mask]
            total = int(len(filtered_df))
            if total < MIN_TOTAL:
                continue

            label_count = count_labels(filtered_df)

            combo_entry = {
                "features": list(subset),
                "values": [input_record[c] for c in subset],
                "total": total,
                "Junk": label_count["Junk"],
                "Unqualified": label_count["Unqualified"],
                "Qualified": label_count["Qualified"],
            }
            statistics["combos"].append(combo_entry)

    return statistics


# ============================================================
# Build input_record & prompt
# ============================================================

def build_input_record(row: pd.Series) -> Dict[str, Any]:
    """Tạo record đầu vào để gửi cho LLM, loại bỏ cột nhãn & điểm."""
    record = row.drop([TARGET_COL, "final_score"], errors="ignore").to_dict()
    return record


def build_model_prompt(
    statistics: Dict[str, Any],
    input_record: Dict[str, Any],
    total_status_count: dict,
) -> str:
    """
    Xây dựng prompt gửi LLM chấm điểm lead.

    - Sử dụng thống kê SỐ LƯỢNG:
        + combos: tổ hợp nhiều cột (2–5 cột) theo các trường: source, status, is_vip, no_of_employees, city.
    - LLM chỉ được trả về 1 JSON: qualification_status, score, explain.
    """

    combo_stats = statistics.get("combos", [])

    qs_count = total_status_count.get("qualification_status_count", {})
    total_all = total_status_count.get("total_count", 0)

    prompt = ""

    # 1. Thông tin lead & thống kê global
    prompt += "Bạn là hệ thống chấm điểm lead B2B dựa trên dữ liệu lịch sử.\n"
    prompt += (
        "Mục tiêu: chọn một trong 3 nhãn cuối cùng cho lead: Junk, Unqualified hoặc Qualified "
        "và cho điểm từ 0 đến 100.\n\n"
    )

    prompt += "Thông tin lead hiện tại (các trường quan trọng như source, status, is_vip, no_of_employees, city):\n"
    prompt += json.dumps(input_record, ensure_ascii=False)
    prompt += "\n\n"

    prompt += "Thống kê toàn cục theo qualification_status trên toàn bộ dữ liệu:\n"
    prompt += f"- Junk: {qs_count.get('Junk', 0)} bản ghi\n"
    prompt += f"- Unqualified: {qs_count.get('Unqualified', 0)} bản ghi\n"
    prompt += f"- Qualified: {qs_count.get('Qualified', 0)} bản ghi\n"
    prompt += f"- Tổng cộng: {total_all} bản ghi\n\n"

    # 2. Thống kê combos
    prompt += (
        "Thống kê theo tổ hợp nhiều cột (combo_feature_stats), đã lọc đúng cấu hình của lead "
        "dựa trên các trường source, status, is_vip, no_of_employees, city.\n"
        "Mỗi phần tử có dạng:\n"
        '{ "features": [...], "values": [...], "total": ..., "Junk": ..., "Unqualified": ..., "Qualified": ... }\n'
        "Dưới đây là danh sách các combo tương ứng với lead này:\n"
    )
    prompt += json.dumps(combo_stats, ensure_ascii=False)
    prompt += "\n\n"

    # 3. Hướng dẫn chấm điểm – CHỈ DỰA TRÊN COMBO
    prompt += (
        "HƯỚNG DẪN CHẤM ĐIỂM (chỉ dựa trên combo_feature_stats, không dùng bất kỳ thông tin nào khác ngoài dữ liệu đã cho):\n"
        "1) Đối với từng combo trong combo_feature_stats:\n"
        "   - So sánh số lượng Junk, Unqualified, Qualified.\n"
        "   - Nhãn nào có số lượng lớn nhất thì combo đó được xem là nghiêng về nhãn đó.\n"
        "   - Nếu số lượng khá sát nhau (chênh lệch không đáng kể) thì coi combo đó là tín hiệu yếu.\n\n"
        "2) Tổng hợp tín hiệu từ tất cả combo:\n"
        "   - Đếm xem có bao nhiêu combo nghiêng rõ rệt về Junk, bao nhiêu combo nghiêng rõ rệt về Unqualified,\n"
        "     và bao nhiêu combo nghiêng rõ rệt về Qualified.\n"
        "   - Combo có chứa is_vip, no_of_employees, city được coi là quan trọng hơn combo chỉ có source/status.\n\n"
        "3) Quy tắc BẢO THỦ cho nhãn Qualified:\n"
        "   - Nếu is_vip = 1 và KHÔNG có nhiều combo mạnh nghiêng về Junk → bạn có thể xem Qualified là ứng viên chính.\n"
        "   - Nếu is_vip = 0 thì CHỈ gán nhãn Qualified khi:\n"
        "       + Có NHIỀU combo (ví dụ từ 3 combo trở lên) trong đó số lượng Qualified lớn hơn rõ rệt hai nhãn còn lại,\n"
        "         và các combo này có chứa các trường quan trọng (như is_vip, no_of_employees, city),\n"
        "       + Đồng thời không có nhóm combo nào nghiêng mạnh về Junk.\n\n"
        "4) Nếu không thỏa điều kiện để gán nhãn Qualified:\n"
        "   - Nếu phần lớn combo mạnh nghiêng về Junk → chọn Junk.\n"
        "   - Nếu không nghiêng rõ về Junk nhưng nhiều combo cho thấy Unqualified lớn hơn Qualified → chọn Unqualified.\n"
        "   - Nếu tín hiệu rất lẫn lộn, hãy ưu tiên Unqualified (trừ khi có lý do rất mạnh cho Junk).\n\n"
        "5) Chấm điểm (score 0–100) theo mức độ chắc chắn:\n"
        "   - Nếu nhãn cuối cùng là Qualified:\n"
        "       + Nếu rất nhiều combo ủng hộ rõ rệt → score cao (80–100).\n"
        "       + Nếu chỉ hơi nghiêng nhưng vẫn đủ điều kiện → score trung bình cao (60–80).\n"
        "   - Nếu nhãn cuối cùng là Unqualified → score khoảng 40–70 tùy mức độ trung tính.\n"
        "   - Nếu nhãn cuối cùng là Junk:\n"
        "       + Nếu nhiều combo ủng hộ rõ rệt → score thấp (0–30).\n"
        "       + Nếu chỉ hơi nghiêng → score khoảng 30–40.\n\n"
        "6) ĐẦU RA (BẮT BUỘC):\n"
        "   - Chỉ trả về DUY NHẤT MỘT ĐỐI TƯỢNG JSON, không thêm bất kỳ chữ nào trước hoặc sau JSON.\n"
        "   - JSON phải đúng định dạng sau:\n"
        '     {"qualification_status": "...", "score": ..., "explain": "..."}\n'
        "     trong đó:\n"
        '       - \"qualification_status\": một trong 3 giá trị \"Junk\", \"Unqualified\", \"Qualified\";\n'
        '       - \"score\": số nguyên từ 0 đến 100;\n'
        '       - \"explain\": vài câu tiếng Việt, giải thích ngắn gọn lý do chọn nhãn và điểm, '
        "nhấn mạnh các combo quan trọng mà bạn dựa vào.\n"
    )

    return prompt


# ============================================================
# Parse output LLM
# ============================================================

def _parse_llm_output(reply: str) -> Tuple[int, str, str]:
    """Cố gắng parse JSON; nếu fail thì dùng regex fallback."""
    default_score = 50
    default_label = "Unqualified"

    # 1) Thử parse JSON trực tiếp
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
        pass

    # 2) Fallback regex
    score = default_score
    qualification_status = default_label
    explain = ""

    try:
        score_match = re.search(r'["\']?score["\']?\s*[:：]\s*([0-9]{1,3})', reply)
        if score_match:
            score = int(score_match.group(1))

        status_match = re.search(r'["\']?qualification_status["\']?\s*[:：]\s*["\']?([^\"\'\n]+)', reply)
        if status_match:
            qualification_status = status_match.group(1).strip().strip('"\' ,')

        explain_match = re.search(r'["\']?explain["\']?\s*[:：]\s*["\'](.+?)["\']', reply, flags=re.S)
        if explain_match:
            explain = explain_match.group(1).strip()
    except Exception:
        pass

    if not explain:
        explain = f"LLM trả về khó parse: {reply}"

    return score, qualification_status, explain


# ============================================================
# Evaluate LLM
# ============================================================

def evaluate_llm_on_csv(
    max_rows: Optional[int] = 30,
    temperature: Optional[float] = None,
    seed: int = 42,
    output_path: Optional[str] = None,
    accuracy_log_path: str = "model_accuracy.txt",
) -> Dict[str, Any]:
    """
    Đánh giá LLM trên file CSV test.

    - df_train: dùng để tính thống kê (combos).
    - df_eval: dùng để đo accuracy (test set 100 dòng).
    """

    # Dữ liệu train / full data để thống kê
    df_train = _load_dataset(_resolve_data_path(), TARGET_COL)

    # Dữ liệu test để đánh giá accuracy
    df_eval_all = _load_dataset(_resolve_test_path(), TARGET_COL)

    if max_rows is not None and max_rows < len(df_eval_all):
        df_eval = df_eval_all.sample(n=max_rows, random_state=seed)
    else:
        df_eval = df_eval_all.copy()

    correct = 0
    test_rows: List[Dict[str, Any]] = []
    total_by_label: Dict[str, int] = {}
    correct_by_label: Dict[str, int] = {}

    temp = temperature if temperature is not None else 0.7
    total = len(df_eval)

    for idx, (_, row) in enumerate(df_eval.iterrows(), start=1):
        y_true = row[TARGET_COL]
        input_record = build_input_record(row)
        statistics = calculate_statistics(df_train, input_record)
        total_status_count = calculate_total_status_count(df_train)
        prompt = build_model_prompt(statistics, input_record, total_status_count)

        print(f"\n=============== Mẫu {idx} / {total} ===============")
        print("Prompt gửi tới LLM:")
        print(prompt)

        raw_reply: Optional[str] = None
        attempt = 0
        while attempt < MAX_RATE_LIMIT_RETRIES:
            try:
                raw_reply = chat_completion(
                    [{"role": "user", "content": prompt}],
                    temperature=temp,
                )
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
            "true_qualification_status": y_true,
            "pred_qualification_status": y_pred,
            "score": score_val,
            "explain": explain,
        }
        test_rows.append(out_record)

        # Tính accuracy
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
        per_label_accuracy[label_key.capitalize()] = (
            correct_count / total_count if total_count > 0 else 0.0
        )

    # Lưu file kết quả dự đoán
    if output_path is None:
        output_path = os.path.join(os.getcwd(), "llm_evaluation_results.csv")

    try:
        df_output = pd.DataFrame(
            test_rows,
            columns=[
                "name",
                "true_qualification_status",
                "pred_qualification_status",
                "score",
                "explain",
            ],
        )
        df_output.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\nĐã lưu kết quả đánh giá vào file: {output_path}")
    except Exception as exc:
        print(f"\nKhông thể ghi file kết quả: {exc}")

    # Ghi độ chính xác vào file model_accuracy.txt
    try:
        with open(accuracy_log_path, "a", encoding="utf-8") as accuracy_file:
            accuracy_file.write(
                f"{datetime.utcnow().isoformat()} - Accuracy: {accuracy * 100:.2f}% "
                f"(total={total}, correct={correct})\n"
            )
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
        labels=list(df_train[TARGET_COL].unique()),
        per_label_accuracy=per_label_accuracy,
        train_csv=_resolve_data_path(),
        val_csv=_resolve_data_path(),
        test_csv=_resolve_test_path(),
    )

    return asdict(metrics)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá LLM trên file CSV test và lưu kết quả"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=30,
        help="Số lượng mẫu để đánh giá (<= số dòng trong file test).",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=None,
        help="Nhiệt độ khi gọi LLM.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed để chọn mẫu ngẫu nhiên từ file test.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn tới file CSV để lưu kết quả. Mặc định: llm_evaluation_results.csv trong thư mục hiện tại.",
    )

    args = parser.parse_args()

    metrics = evaluate_llm_on_csv(
        max_rows=args.max_rows,
        temperature=args.temp,
        seed=args.seed,
        output_path=args.output,
    )

    print("\n============ Tổng kết đánh giá ============")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
