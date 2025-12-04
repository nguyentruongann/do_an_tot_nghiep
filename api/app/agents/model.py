from __future__ import annotations

import json
import re
import os
from typing import Any, Dict, Optional, Tuple, List
from collections import defaultdict

import pandas as pd
from itertools import combinations
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

    Quan trọng:
    - new_df: chỉ chứa các dòng mới được chấm TRONG LẦN GỌI NÀY.
    - session_df: toàn bộ lịch sử của session (cũ + mới).
    - global_csv: chỉ append thêm new_df (tránh nhân bản dữ liệu cũ).
    """
    # 1. Chấm điểm các dòng mới
    results: List[Dict[str, Any]] = []
    for raw_record in rows:
        score_val, label, explain, _ = _score_with_dataset_or_fallback(raw_record)
        out_record = dict(raw_record)
        out_record.update(
            {
                "score": score_val,
                "qualification_status": label,
                "explain": explain,
            }
        )
        results.append(out_record)

    # DataFrame chỉ của LẦN GỌI NÀY
    new_df = pd.DataFrame(results)

    cfg = load_config()
    sessions_dir = getattr(cfg.training, "sessions_dir", "/app/data/sessions")
    global_dir = getattr(cfg.training, "global_dir", "/app/data/global")
    os.makedirs(sessions_dir, exist_ok=True)
    os.makedirs(global_dir, exist_ok=True)

    # 2. Cập nhật file session: prev_df + new_df
    session_csv = os.path.join(sessions_dir, f"{session_id}_scored.csv")
    if os.path.exists(session_csv):
        try:
            prev_df = pd.read_csv(session_csv)
            session_df = pd.concat([prev_df, new_df], ignore_index=True)
        except Exception:
            # lỗi đọc file cũ -> dùng new_df
            session_df = new_df.copy()
    else:
        session_df = new_df.copy()
    session_df.to_csv(session_csv, index=False)

    # 3. Cập nhật file global: CHỈ append new_df (tránh trùng)
    global_csv = os.path.join(global_dir, "global_scored.csv")
    if os.path.exists(global_csv):
        try:
            g_df = pd.read_csv(global_csv)
            g_df = pd.concat([g_df, new_df], ignore_index=True)
        except Exception:
            g_df = new_df.copy()
    else:
        g_df = new_df.copy()
    g_df.to_csv(global_csv, index=False)

    # 4. Preview: chỉ các dòng mới cho lần gọi hiện tại
    preview = new_df.head(5).to_dict(orient="records")

    return {
        "saved_session_csv": session_csv,
        "updated_global_csv": global_csv,
        "rows_scored": len(rows),
        "results_preview": preview,
        # Nếu muốn debug cả session thì có thể trả thêm:
        # "session_preview": session_df.tail(5).to_dict(orient="records"),
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
