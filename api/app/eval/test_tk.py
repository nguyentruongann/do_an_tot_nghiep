import pandas as pd
from collections import defaultdict
import json

# Đọc dữ liệu từ CSV
def _load_dataset(path: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"File {path} không có cột '{target_col}'")
    df[target_col] = df[target_col].astype(str).str.strip()
    df = df[df[target_col] != ""]
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    valid_labels = {"Junk", "Qualified", "Unqualified"}
    df = df[df[target_col].isin(valid_labels)].reset_index(drop=True)
    return df

# Thống kê tổng số các giá trị trong qualification_status của toàn bộ file
def calculate_total_status_count(df: pd.DataFrame) -> dict:
    status_count = df['qualification_status'].value_counts().to_dict()
    total_count = len(df)
    # Trả về số lượng các giá trị trong qualification_status
    return {
        "total_count": total_count,
        "qualification_status_count": status_count
    }

# Tính toán thống kê tỷ lệ phần trăm cho từng qualification_status cho mỗi giá trị trong cột
def calculate_statistics(df: pd.DataFrame, input_row: pd.Series) -> dict:
    statistics = {}
    USEFUL_COLS = [
        "source", "status", "is_vip", "no_of_employees",
        "job_title", "city"
    ]

    for column in USEFUL_COLS:
        if column not in df.columns or column not in input_row:
            continue  # Nếu cột không có trong dữ liệu hoặc không có trong dòng đầu vào thì bỏ qua

        # Thống kê cho các giá trị trong cột
        column_stats = defaultdict(lambda: {"Junk": 0, "Unqualified": 0, "Qualified": 0, "total": 0})

        # Lọc dữ liệu theo giá trị trong cột từ dòng đầu vào
        target_value = input_row[column]
        filtered_df = df[df[column] == target_value]

        for _, row in filtered_df.iterrows():
            label = row['qualification_status']
            column_stats[target_value][label] += 1
            column_stats[target_value]["total"] += 1  # Tổng số bản ghi cho giá trị này trong cột

        # Tính tỷ lệ phần trăm cho mỗi giá trị trong cột
        for value, stats in column_stats.items():
            total = stats["total"]
            if total > 0:
                stats["Junk_percent"] = (stats["Junk"] / total) * 100
                stats["Unqualified_percent"] = (stats["Unqualified"] / total) * 100
                stats["Qualified_percent"] = (stats["Qualified"] / total) * 100

        statistics[column] = column_stats

    return statistics

# Hàm để chuyển các giá trị kiểu int64 sang kiểu int trước khi in ra JSON
def convert_int64_to_int(obj):
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): convert_int64_to_int(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_int64_to_int(i) for i in obj]
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    return obj

# Kiểm tra thống kê cho tất cả các cột có trong USEFUL_COLS cho 1 dòng dữ liệu
def test_calculate_statistics_for_row(file_path: str, row_index: int):
    # Đọc dữ liệu từ file CSV
    df = _load_dataset(file_path, "qualification_status")
    
    # Lấy một dòng mẫu từ dữ liệu
    input_row = df.iloc[row_index]
    
    # Tính toán thống kê cho dòng mẫu
    statistics = calculate_statistics(df, input_row)
    
    # Tính thống kê tổng cho qualification_status
    total_status_count = calculate_total_status_count(df)
    
    # Chuyển đổi các giá trị kiểu int64 trước khi in ra JSON
    statistics = convert_int64_to_int(statistics)
    total_status_count = convert_int64_to_int(total_status_count)
    
    # In kết quả thống kê cho các cột có trong dòng dữ liệu
    print(f"Statistics for Row {row_index}:")
    print(json.dumps(statistics, ensure_ascii=False, indent=4))
    
    # In thống kê tổng số lượng trong qualification_status
    print(f"\nTotal qualification_status Count for the whole dataset:")
    print(json.dumps(total_status_count, ensure_ascii=False, indent=4))

# Đường dẫn tới file CSV của bạn
file_path = r'D:\HK9\DoAn\ccccccccc\vip_scoring_with_eval_and_docs\api\data\training\data.csv'

# Chọn dòng mẫu cần kiểm tra (ví dụ: dòng thứ ba)
row_index = 2  # Bạn có thể thay đổi chỉ số dòng ở đây

# Gọi hàm kiểm tra cho dòng mẫu
test_calculate_statistics_for_row(file_path, row_index)
