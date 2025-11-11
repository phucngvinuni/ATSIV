import pandas as pd

# --- Cấu hình ---
MASTER_CSV_PATH = "master_dataset.csv"
SPLIT_NAME = "test_manual"
TRAIT_COLUMNS = ['adipose_fin', 'pelvic_fin', 'barbel', 'multiple_dorsal_fin']

def analyze_manual_test_set_distribution():
    """
    Phân tích số lượng mẫu dương, âm và không xác định
    cho các đặc điểm trong tập test_manual.
    """
    print(f"--- Phân tích phân phối nhãn trong tập '{SPLIT_NAME}' ---")

    try:
        df = pd.read_csv(MASTER_CSV_PATH, low_memory=False)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{MASTER_CSV_PATH}'.")
        print("Vui lòng chạy 'create_master_csv.py' trước.")
        return

    # Lọc ra tập test_manual
    manual_df = df[df['split'] == SPLIT_NAME]

    if manual_df.empty:
        print(f"Không tìm thấy mẫu nào cho split '{SPLIT_NAME}'. Vui lòng kiểm tra lại file master_dataset.csv.")
        return

    print(f"Tổng số mẫu trong tập '{SPLIT_NAME}': {len(manual_df)}\n")

    print("Phân phối nhãn cho từng đặc điểm:")
    print("-" * 40)
    for trait in TRAIT_COLUMNS:
        if trait not in manual_df.columns:
            print(f"Cột '{trait}' không tồn tại.")
            continue
            
        # .value_counts() sẽ tự động bỏ qua các giá trị NaN
        counts = manual_df[trait].value_counts(dropna=False)
        
        present_count = counts.get(1.0, 0)
        absent_count = counts.get(0.0, 0)
        unknown_count = counts.get(-1.0, 0)
        nan_count = counts.get(pd.NA, 0) + counts.get(None, 0)

        print(f"Đặc điểm: {trait}")
        print(f"  - Có (Present = 1.0):   {present_count} mẫu")
        print(f"  - Không có (Absent = 0.0): {absent_count} mẫu")
        print(f"  - Không rõ (Unknown = -1.0): {unknown_count} mẫu")
        print(f"  - Thiếu dữ liệu (NaN): {nan_count} mẫu")
        print("-" * 20)

if __name__ == "__main__":
    analyze_manual_test_set_distribution()