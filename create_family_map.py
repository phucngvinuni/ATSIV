# create_family_map.py

import pandas as pd
import json
import os

# --- Cấu hình ---
MASTER_CSV_PATH = "master_dataset.csv"
OUTPUT_MAP_PATH = "family_to_id.json"

def main():
    """
    Đọc file master dataset, tìm tất cả các họ (family) duy nhất,
    và tạo ra một file JSON để ánh xạ từ tên họ sang một ID số.
    """
    print(f"Bắt đầu xử lý file: {MASTER_CSV_PATH}")
    
    # Kiểm tra xem file master có tồn tại không
    if not os.path.exists(MASTER_CSV_PATH):
        print(f"Lỗi: Không tìm thấy file '{MASTER_CSV_PATH}'.")
        print("Vui lòng chạy script 'create_master_csv.py' trước để tạo file này.")
        return

    # Đọc file CSV
    df = pd.read_csv(MASTER_CSV_PATH, low_memory=False)

    # Kiểm tra xem cột 'family' có tồn tại không
    if 'family' not in df.columns:
        print("Lỗi: Cột 'family' không được tìm thấy trong file master_dataset.csv.")
        return

    # Lấy danh sách các họ duy nhất và loại bỏ các giá trị NaN (không xác định)
    # Sắp xếp theo thứ tự alphabet để đảm bảo tính nhất quán mỗi lần chạy
    all_families = sorted(df['family'].dropna().unique())
    
    # Tạo từ điển mapping
    family_to_id = {family: i for i, family in enumerate(all_families)}
    
    num_families = len(family_to_id)

    # Ghi file JSON
    try:
        with open(OUTPUT_MAP_PATH, 'w') as f:
            json.dump(family_to_id, f, indent=4, ensure_ascii=False)
        
        print("\nHoàn thành!")
        print(f"Đã tìm thấy tổng cộng {num_families} họ (families).")
        print(f"Đã lưu file mapping tại: {OUTPUT_MAP_PATH}")
        
        # In ra 5 họ đầu tiên làm ví dụ
        print("\nVí dụ 5 họ đầu tiên trong mapping:")
        for i, (family, id_val) in enumerate(family_to_id.items()):
            if i >= 5:
                break
            print(f"  - '{family}': {id_val}")
            
    except Exception as e:
        print(f"\nĐã xảy ra lỗi khi ghi file: {e}")

if __name__ == "__main__":
    main()