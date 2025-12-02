from __future__ import annotations

import json
import re
import os
from typing import Any, Dict, Optional, Tuple, List
from collections import defaultdict

import pandas as pd

from app.llm import chat_completion
from app.config import load_config

# ===================== CẤU HÌNH =====================
import numpy as np

TARGET_COL = "qualification_status"

# Chỉ 5 cột quan trọng để thống kê & dùng trong prompt
USEFUL_COLS = [
    "source",
    "status",
    "is_vip",
    "no_of_employees",
    "city",
]
USEFUL_COLS_SET = set(USEFUL_COLS)


# Dùng cùng bộ cột này cho heuristics
EVAL_USEFUL_COLS: List[str] = USEFUL_COLS


# ========= Chuẩn hoá / mapping nhãn =========

def _normalize_qualification_status(value: Any) -> Optional[str]:
    """Chuẩn hoá label về 1 trong 3 giá trị: Junk / Qualified / Unqualified."""
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None

    low = v.lower()

    mapping = {
        "qualified": "Qualified",
        "q": "Qualified",
        "unqualified": "Unqualified",
        "not qualified": "Unqualified",
        "not_qualified": "Unqualified",
        "khong phu hop": "Unqualified",
        "không phù hợp": "Unqualified",
        "junk": "Junk",
        "spam": "Junk",
        "trash": "Junk",
        "rac": "Junk",
        "rác": "Junk",
    }
    if low in mapping:
        return mapping[low]

    labels = ["Junk", "Qualified", "Unqualified"]

    for lab in labels:
        if low == lab.lower():
            return lab

    for lab in labels:
        if lab.lower() in low:
            return lab

    return None


def _label_from_score(score: int) -> str:
    """
    Quy tắc chặt giữa score và qualification_status:

    - 80–100 => Qualified
    - 40–79  => Unqualified
    - 0–39   => Junk
    """
    if score >= 80:
        return "Qualified"
    if score >= 40:
        return "Unqualified"
    return "Junk"


# ================================================================
# Data-driven scoring helpers (dựa trên evaluate_llm.py)
# ================================================================

def _load_dataset(path: str, target_col: str = TARGET_COL) -> pd.DataFrame:
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


def calculate_total_status_count(df: pd.DataFrame, target_col: str = TARGET_COL) -> dict:
    """
    Thống kê tổng số lượng các giá trị trong qualification_status (hoặc target_col).
    """
    status_count = df[target_col].value_counts().to_dict()
    total_count = len(df)
    return {
        "total_count": total_count,
        "qualification_status_count": status_count,
    }


# --------- HÀM CHUẨN HOÁ INPUT CHO THỐNG KÊ ---------

def _normalize_statistics_input(input_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chuẩn hoá input_record để dùng cho calculate_statistics:
    - Chỉ giữ 5 cột trong USEFUL_COLS.
    - Nếu giá trị là list/tuple/set/Series/Index -> lấy phần tử đầu tiên (nếu có).
    - Nếu rỗng -> dùng None.
    - Nếu là dict hoặc object lạ -> convert sang str.
    """
    cleaned: Dict[str, Any] = {}
    for col in USEFUL_COLS:
        val = input_record.get(col, None)

        # Giữ nguyên các kiểu đơn giản
        if isinstance(val, (str, int, float, bool)) or val is None:
            cleaned[col] = val
            continue

        # pandas Series / Index
        if isinstance(val, (pd.Series, pd.Index)):
            if len(val) > 0:
                cleaned[col] = val.iloc[0]
            else:
                cleaned[col] = None
            continue

        # list / tuple / set
        if isinstance(val, (list, tuple, set)):
            if len(val) > 0:
                cleaned[col] = list(val)[0]
            else:
                cleaned[col] = None
            continue

        # dict hoặc loại khác -> stringify
        cleaned[col] = str(val)

    return cleaned
def _normalize_stat_value(value: Any):
    """
    Chuẩn hoá giá trị 1 ô để dùng trong calculate_statistics:
    - Nếu là list/tuple/set: lấy phần tử đầu, rỗng thì trả về None.
    - Nếu là pandas Series/Index: lấy phần tử đầu, rỗng thì None.
    - Nếu là numpy array: rỗng thì None, ngược lại lấy phần tử đầu.
    - Nếu là dict: bỏ qua (trả về None).
    - Nếu là NaN hoặc chuỗi rỗng: trả về None.
    - Còn lại: giữ nguyên (scalar).
    """
    # list / tuple / set
    if isinstance(value, (list, tuple, set)):
        if len(value) == 0:
            return None
        # lấy phần tử đầu tiên
        return next(iter(value))

    # pandas Series / Index
    if isinstance(value, (pd.Series, pd.Index)):
        if len(value) == 0:
            return None
        try:
            return value.iloc[0]
        except Exception:
            return value[0]

    # numpy array
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        # phẳng rồi lấy phần tử đầu
        return value.reshape(-1)[0]

    # dict thì không có ý nghĩa so sánh bằng
    if isinstance(value, dict):
        return None

    # NaN
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    # Chuỗi rỗng / toàn khoảng trắng
    if isinstance(value, str):
        if value.strip() == "":
            return None

    return value


def calculate_statistics(df: pd.DataFrame, input_record: Dict[str, Any]) -> dict:
    """
    Tính toán tỷ lệ phần trăm của qualification_status cho mỗi GIÁ TRỊ
    trong các cột hữu ích (USEFUL_COLS) dựa trên 1 record đầu vào.

    ⚠️ Lưu ý: hàm này được thiết kế để chịu được input_record "bẩn"
    (giá trị là list/ndarray/Series rỗng, v.v.).
    """
    statistics: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # Cho phép input_record là dict HOẶC pandas.Series
    if isinstance(input_record, pd.Series):
        input_dict = input_record.to_dict()
    else:
        input_dict = dict(input_record)  # copy để tránh sửa original

    for column in USEFUL_COLS:
        # Bỏ qua nếu CSV không có cột này
        if column not in df.columns:
            continue

        raw_value = input_dict.get(column, None)
        target_value = _normalize_stat_value(raw_value)

        # Nếu sau khi chuẩn hoá vẫn không có giá trị → bỏ qua cột
        if target_value is None:
            continue

        # Lọc những dòng trong CSV có cùng giá trị ở cột này
        filtered_df = df[df[column] == target_value]

        if filtered_df.empty:
            # Không có mẫu nào trùng → cũng không cần thống kê
            continue

        # Khởi tạo cấu trúc lưu thống kê cho giá trị này
        col_stats = {
            str(target_value): {
                "Junk": 0,
                "Unqualified": 0,
                "Qualified": 0,
                "total": 0,
            }
        }

        # Đếm số lượng từng nhãn
        for label, count in filtered_df["qualification_status"].value_counts().items():
            if label not in {"Junk", "Unqualified", "Qualified"}:
                continue
            col_stats[str(target_value)][label] += int(count)
            col_stats[str(target_value)]["total"] += int(count)

        # Tính % cho từng nhãn
        stats_for_value = col_stats[str(target_value)]
        total = stats_for_value["total"]
        if total > 0:
            stats_for_value["Junk_percent"] = stats_for_value["Junk"] / total * 100
            stats_for_value["Unqualified_percent"] = (
                stats_for_value["Unqualified"] / total * 100
            )
            stats_for_value["Qualified_percent"] = (
                stats_for_value["Qualified"] / total * 100
            )
            # để thuận tay LLM, giữ lại *_count
            stats_for_value["Junk_count"] = stats_for_value["Junk"]
            stats_for_value["Unqualified_count"] = stats_for_value["Unqualified"]
            stats_for_value["Qualified_count"] = stats_for_value["Qualified"]

        statistics[column] = col_stats

    return statistics
def extract_info_from_text(text: str) -> Tuple[Dict[str, Any], str]:
    """
    Hàm nhận một mô tả tự nhiên và cố gắng trích xuất các trường
    source, status, is_vip, no_of_employees, city.

    Nếu đã cấu hình endpoint LLM, hàm sẽ gửi prompt yêu cầu trả về
    DUY NHẤT một JSON trên một dòng với các khóa này.

    - is_vip: số nguyên 0 hoặc 1 (không rõ thì 0)
    - các field còn lại là chuỗi, nếu không có thì là ""

    Trả về:
        (record_trich_xuat, raw_reply)
    """
    cfg = load_config()
    base_url = getattr(cfg.llm, "base_url", "") or ""
    temperature = getattr(cfg.llm, "temperature", 0.2)

    # Không có LLM endpoint thì chịu, trả về rỗng
    if not base_url.strip():
        return {}, ""

    prompt = (
        "Bạn là hệ thống trích xuất thông tin lead cho CRM. "
        "Đọc mô tả sau và trả về DUY NHẤT một đối tượng JSON trên một dòng với các khóa: "
        "source, status, is_vip, no_of_employees, city. "
        "Giá trị is_vip phải là số nguyên (0 hoặc 1), nếu không rõ thì đặt 0. "
        "Nếu không có giá trị cho các khóa khác thì để chuỗi rỗng. "
        "Không được thêm khóa khác. Chỉ trả về JSON, không lời giải thích.\n\n"
        f"Mô tả: {text}"
    )

    try:
        raw_reply = chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
        )
    except Exception:
        # gọi LLM lỗi thì trả dict rỗng, raw_reply rỗng
        return {}, ""

    # Cố gắng parse JSON từ reply
    try:
        m = re.search(r"\{.*\}", raw_reply, flags=re.S)
        raw_json = m.group(0) if m else raw_reply
        record = json.loads(raw_json)
    except Exception:
        # Không parse được thì trả raw_reply để debug
        return {}, raw_reply

    desired_keys = ["source", "status", "is_vip", "no_of_employees", "city"]
    cleaned: Dict[str, Any] = {}

    for key in desired_keys:
        val = record.get(key)

        if key == "is_vip":
            # ép về int, lỗi thì 0
            try:
                cleaned[key] = int(val)
            except Exception:
                cleaned[key] = 0
        else:
            # luôn là string, không có thì ""
            if isinstance(val, str):
                cleaned[key] = val
            elif val is None:
                cleaned[key] = ""
            else:
                cleaned[key] = str(val)

    return cleaned, raw_reply


def build_input_record_for_llm(raw_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loại bỏ các cột target khỏi record trước khi gửi cho LLM.
    """
    record = dict(raw_record)
    record.pop(TARGET_COL, None)
    record.pop("final_score", None)
    return record


def build_model_prompt(statistics: Dict[str, Any],
                       input_record: Dict[str, Any],
                       total_status_count: dict) -> str:
    """
    Xây dựng prompt gửi tới LLM (giữ nguyên nội dung như trong evaluate_llm.py).
    """
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
    # Thống kê từng cột hữu ích (5 cột USEFUL_COLS)
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
    # Hướng dẫn ra quyết định và yêu cầu đầu ra (giữ nguyên prompt)
    prompt += """
    Dựa trên thông tin thống kê trên, bạn hãy:
    1. Dựa vào tần suất và số lượng các nhãn so với nhãn tổng trong các lead tương tự để xác định nhãn cuối cùng cho lead này (Junk, Unqualified, Qualified).
    2. Nhãn cuối cùng là **Qualified** khi is_vip là 1 hoặc **% Qualified** lệch so với %Junk hoặc %Unqualified trên 8% và số lượng lệch chiếm số cột nhiều hơn mới có thể được phân loại là **Qualified**.
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
        {"score": <số nguyên từ 0 đến 100>, "qualification_status": "Junk|Unqualified|Qualified", "explain": "lý do ngắn gọn"}. 
        Không trả về thêm bất kỳ văn bản nào ngoài JSON này.
    """
    return prompt


# ================================================================
# Một số heuristic / fallback
# ================================================================

def _choose_label_by_statistics(stats: dict, input_record: Dict[str, Any]) -> Tuple[int, str, str]:
    """
    Dùng thống kê để chọn nhãn (khi không gọi LLM).
    """
    vip_val = input_record.get("is_vip")
    try:
        vip_int = int(vip_val)
    except Exception:
        vip_int = 0
    if vip_int == 1:
        return 90, "Qualified", "Lead được đánh dấu VIP nên được coi là Qualified"

    votes = {"Junk": 0, "Unqualified": 0, "Qualified": 0}
    for _, value_stats in stats.items():
        for _, stat in value_stats.items():
            total = stat.get("total", 0)
            if total == 0:
                continue
            perc = {
                "Junk": stat.get("Junk_percent", 0),
                "Unqualified": stat.get("Unqualified_percent", 0),
                "Qualified": stat.get("Qualified_percent", 0),
            }
            sorted_labels = sorted(perc.items(), key=lambda x: x[1], reverse=True)
            best_label, best_perc = sorted_labels[0]
            second_perc = sorted_labels[1][1]
            diff = best_perc - second_perc
            if best_label == "Qualified" and diff >= 8.0:
                votes["Qualified"] += 1
            elif best_label == "Unqualified":
                votes["Unqualified"] += 1
            else:
                votes["Junk"] += 1
            break

    if votes["Qualified"] >= 4:
        label = "Qualified"
    else:
        label = "Unqualified" if votes["Unqualified"] >= votes["Junk"] else "Junk"

    import random
    if label == "Qualified":
        score_val = random.randint(80, 100)
    elif label == "Unqualified":
        score_val = random.randint(40, 79)
    else:
        score_val = random.randint(0, 39)

    explain_parts = []
    if votes["Qualified"]:
        explain_parts.append(f"Có {votes['Qualified']} cột ủng hộ Qualified")
    if votes["Unqualified"]:
        explain_parts.append(f"Có {votes['Unqualified']} cột ủng hộ Unqualified")
    if votes["Junk"]:
        explain_parts.append(f"Có {votes['Junk']} cột ủng hộ Junk")
    explain = "; ".join(explain_parts) if explain_parts else "Không đủ dữ liệu để kết luận rõ"
    return score_val, label, explain


def heuristics_classification(input_record: Dict[str, Any]) -> Tuple[int, str, str]:
    """
    Fallback đơn giản khi không có dataset.
    """
    status = str(input_record.get("status", "")).strip().lower()
    is_vip = input_record.get("is_vip")
    try:
        is_vip_int = int(is_vip)
    except Exception:
        is_vip_int = 0
    no_emp = str(input_record.get("no_of_employees", "")).lower()

    if is_vip_int == 1:
        return 90, "Qualified", "is_vip = 1 nên lead được coi là Qualified"
    if status == "converted":
        return 85, "Qualified", "status = Converted nên lead được coi là Qualified"

    large_emp = ["1000", "5000", "10000", "100-999", "51-200", "201-500", "501-1000"]
    small_emp = ["1-10", "1-50", "1-100", "10", "11-50"]

    if any(k in no_emp for k in large_emp):
        return 80, "Qualified", "công ty lớn"
    if any(k in no_emp for k in small_emp):
        return 60, "Unqualified", "công ty nhỏ"

    return 20, "Junk", "thiếu thông tin rõ ràng hoặc không đủ tiềm năng"


def _score_with_dataset_or_fallback(raw_record: Dict[str, Any]) -> Tuple[int, str, str, str]:
    """
    Chấm điểm một record: ưu tiên dùng dataset + thống kê,
    nếu không có thì dùng heuristics.
    """
    cfg = load_config()
    data_path = getattr(cfg.training, "data_path", None)
    df: Optional[pd.DataFrame] = None
    if data_path:
        df = _load_dataset(data_path, TARGET_COL)

    if df is not None and not df.empty:
        input_record_llm = build_input_record_for_llm(raw_record)
        stats = calculate_statistics(df, input_record_llm)
        total_status_count = calculate_total_status_count(df, TARGET_COL)

        base_url = getattr(cfg.llm, "base_url", "") or ""
        temperature = getattr(cfg.llm, "temperature", 0.2)

        if base_url.strip():
            prompt = build_model_prompt(stats, input_record_llm, total_status_count)
            try:
                raw_reply = chat_completion(
                    [{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                score_val, label, explain = _parse_llm_output(raw_reply)
                return score_val, label, explain, raw_reply
            except Exception:
                score_val, label, explain = _choose_label_by_statistics(stats, input_record_llm)
                return score_val, label, explain, ""
        else:
            score_val, label, explain = _choose_label_by_statistics(stats, input_record_llm)
            return score_val, label, explain, ""

    # Không có dataset -> heuristics
    input_record = {k: raw_record.get(k) for k in EVAL_USEFUL_COLS}
    score_val, label, explain = heuristics_classification(input_record)
    return score_val, label, explain, ""


def score_rows_and_save(session_id: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Dùng cho endpoint /score: chấm nhiều dòng và lưu CSV.
    """
    results: List[Dict[str, Any]] = []
    for raw_record in rows:
        score_val, label, explain, _ = _score_with_dataset_or_fallback(raw_record)
        out_record = dict(raw_record)
        out_record.update({
            "score": score_val,
            "qualification_status": label,
            "explain": explain,
        })
        results.append(out_record)

    df = pd.DataFrame(results)
    cfg = load_config()

    sessions_dir = getattr(cfg.training, "sessions_dir", "/app/data/sessions")
    global_dir = getattr(cfg.training, "global_dir", "/app/data/global")
    os.makedirs(sessions_dir, exist_ok=True)
    os.makedirs(global_dir, exist_ok=True)

    session_csv = os.path.join(sessions_dir, f"{session_id}_scored.csv")
    if os.path.exists(session_csv):
        try:
            prev_df = pd.read_csv(session_csv)
            df = pd.concat([prev_df, df], ignore_index=True)
        except Exception:
            pass
    df.to_csv(session_csv, index=False)

    global_csv = os.path.join(global_dir, "global_scored.csv")
    if os.path.exists(global_csv):
        try:
            g_df = pd.read_csv(global_csv)
            g_df = pd.concat([g_df, df], ignore_index=True)
        except Exception:
            g_df = df.copy()
    else:
        g_df = df.copy()
    g_df.to_csv(global_csv, index=False)

    preview = df.head(5).to_dict(orient="records")
    return {
        "saved_session_csv": session_csv,
        "updated_global_csv": global_csv,
        "rows_scored": len(rows),
        "results_preview": preview,
    }


def _parse_llm_output(reply: str) -> Tuple[int, str, str]:
    """
    Parse output từ LLM giống evaluate_llm.py, nhưng đảm bảo score
    nằm đúng miền với nhãn.
    """
    default_score = 50
    default_label = _label_from_score(default_score)

    try:
        m = re.search(r"\{.*\}", reply, flags=re.S)
        raw_json = m.group(0) if m else reply
        data = json.loads(raw_json)
    except Exception:
        explain = f"LLM trả về khó parse: {reply}"
        return default_score, default_label, explain

    q_raw = data.get("qualification_status")
    q_model = _normalize_qualification_status(q_raw)

    score_raw = data.get("score", None)
    try:
        score_val: Optional[int] = int(score_raw)
    except Exception:
        score_val = None

    explain = str(data.get("explain", ""))

    if q_model in {"Junk", "Qualified", "Unqualified"}:
        if q_model == "Qualified":
            if score_val is None or score_val < 80:
                score_val = 85
            elif score_val > 100:
                score_val = 100
        elif q_model == "Unqualified":
            if score_val is None or not (40 <= score_val <= 79):
                score_val = 60
        else:  # Junk
            if score_val is None or score_val >= 40:
                score_val = 20

        score_val = int(max(0, min(100, score_val)))
        q_final = q_model
        return score_val, q_final, explain

    if score_val is None:
        score_val = default_score
    score_val = int(max(0, min(100, score_val)))
    q_final = _label_from_score(score_val)
    return score_val, q_final, explain


def score_lead(raw_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    API chấm điểm 1 lead (nếu bạn muốn dùng riêng lẻ).
    """
    score_val, label, explain, raw_reply = _score_with_dataset_or_fallback(raw_record)
    return {
        "score": int(score_val),
        "qualification_status": label,
        "explain": explain,
        "raw_reply": raw_reply,
    }
