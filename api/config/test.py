import pandas as pd

# Đọc dữ liệu từ file CSV
file_path = r"D:\HK9\DoAn\ccccccccc\vip_scoring_with_eval_and_docs\api\data\training\data.csv"  # Đường dẫn đến file CSV của bạn

# Đọc file CSV
df = pd.read_csv(file_path)

# Kiểm tra các cột hiện có trong dữ liệu
print("Các cột hiện có trong dữ liệu:")
print(df.columns)

# Làm tròn giá trị trong cột 'final_score' thành số nguyên
df['final_score'] = df['final_score'].round().astype(int)

# Kiểm tra lại sau khi làm tròn
print("\nGiá trị sau khi làm tròn 'final_score':")
print(df[['final_score']].head())  # In ra 5 dòng đầu của cột 'final_score'

# Lưu lại file CSV sau khi đã làm tròn
df.to_csv(r'D:\HK9\DoAn\ccccccccc\vip_scoring_with_eval_and_docs\api\data\training\data.csv', index=False)  # Đường dẫn lưu file mới

print("\nFile đã được lưu thành công.")
