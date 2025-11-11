import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import Counter

# --- Cấu hình ---
MASTER_CSV_PATH = "master_dataset.csv"
OUTPUT_DIR = "dataset_analysis"
TOP_N_SPECIES = 30  # Số lượng loài phổ biến nhất để vẽ biểu đồ
TRAIT_COLUMNS = ['adipose_fin', 'pelvic_fin', 'barbel', 'multiple_dorsal_fin']

# --- Hàm helper ---
def save_plot(fig, filename):
    """Lưu biểu đồ vào thư mục output."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=150)
    print(f"Đã lưu biểu đồ tại: {filepath}")
    plt.close(fig)

def main():
    print(f"Bắt đầu phân tích file: {MASTER_CSV_PATH}")
    
    # --- 1. Load dữ liệu ---
    try:
        df = pd.read_csv(MASTER_CSV_PATH, low_memory=False)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{MASTER_CSV_PATH}'. Hãy chạy script create_master_csv.py trước.")
        return

    # --- 2. Phân tích Tổng quan ---
    print("\n--- 2. Phân tích Tổng quan ---")
    print(f"Tổng số mẫu (hàng) trong dataset: {len(df)}")
    print(f"Tổng số cột: {len(df.columns)}")
    
    split_counts = df['split'].value_counts()
    print("\nPhân bổ dữ liệu theo các tập (split):")
    print(split_counts)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=split_counts.index, y=split_counts.values, ax=ax)
    ax.set_title("Số lượng mẫu trong mỗi tập dữ liệu (Split)")
    ax.set_ylabel("Số lượng mẫu")
    ax.set_xlabel("Tên tập (Split)")
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    save_plot(fig, "01_split_distribution.png")

    # --- 3. Phân tích Tác vụ Phân loại Loài (Classification) ---
    print("\n--- 3. Phân tích Tác vụ Phân loại Loài ---")
    species_df = df.dropna(subset=['standardized_species'])
    num_unique_species = species_df['standardized_species'].nunique()
    print(f"Tổng số loài duy nhất có nhãn: {num_unique_species}")

    # Phân phối số lượng ảnh mỗi loài (Long-tail distribution)
    species_counts = species_df['standardized_species'].value_counts()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    species_counts.plot(logy=True, ax=ax, grid=True)
    ax.set_title("Phân phối số lượng ảnh mỗi loài (Thang Log)")
    ax.set_xlabel("Các loài (đã sắp xếp)")
    ax.set_ylabel("Số lượng ảnh (log scale)")
    save_plot(fig, "02_species_long_tail_distribution.png")

    # Biểu đồ cho TOP_N_SPECIES loài phổ biến nhất
    top_species = species_counts.head(TOP_N_SPECIES)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(y=top_species.index, x=top_species.values, orient='h', ax=ax)
    ax.set_title(f"Top {TOP_N_SPECIES} loài có nhiều ảnh nhất")
    ax.set_xlabel("Số lượng ảnh")
    ax.set_ylabel("Tên loài")
    save_plot(fig, "03_top_n_species.png")

    # --- 4. Phân tích Tác vụ Nhận dạng Đặc điểm (Identification) ---
    print("\n--- 4. Phân tích Tác vụ Nhận dạng Đặc điểm ---")
    trait_df = df.dropna(subset=TRAIT_COLUMNS, how='all')
    print(f"Tổng số mẫu có ít nhất một nhãn đặc điểm: {len(trait_df)}")
    
    trait_presence = {}
    for col in TRAIT_COLUMNS:
        counts = trait_df[col].value_counts(dropna=True)
        trait_presence[col] = {
            'Present (1)': counts.get(1.0, 0),
            'Absent (0)': counts.get(0.0, 0),
            'Unknown (-1)': counts.get(-1.0, 0)
        }
    
    trait_presence_df = pd.DataFrame(trait_presence).T
    print("\nPhân bổ sự hiện diện của các đặc điểm:")
    print(trait_presence_df)

    fig = trait_presence_df.plot(kind='bar', stacked=True, figsize=(12, 7),
                                 title="Phân bổ nhãn Hiện diện/Vắng mặt/Không rõ cho mỗi đặc điểm").get_figure()
    plt.ylabel("Số lượng mẫu")
    plt.xticks(rotation=45, ha="right")
    save_plot(fig, "04_trait_distribution.png")

    # --- 5. Phân tích Tác vụ Phân đoạn (Segmentation) ---
    print("\n--- 5. Phân tích Tác vụ Phân đoạn ---")
    seg_df = df.dropna(subset=['mask_path'])
    print(f"Tổng số mẫu có mặt nạ phân đoạn: {len(seg_df)}")
    
    # Phân bổ theo split
    seg_split_counts = seg_df['split'].value_counts()
    print("\nPhân bổ mẫu có mặt nạ theo split:")
    print(seg_split_counts)

    # Phân tích các đặc điểm có trong mặt nạ (đòi hỏi đọc file ảnh, có thể chậm)
    # Tạm thời bỏ qua bước này để script chạy nhanh.
    # Nếu muốn phân tích, bạn cần code để đọc từng file mask và đếm các giá trị pixel.
    # Ví dụ:
    # all_traits_in_masks = []
    # for mask_path in tqdm(seg_df['mask_path'], desc="Analyzing masks"):
    #     mask = np.array(Image.open(mask_path))
    #     unique_pixels = np.unique(mask)
    #     all_traits_in_masks.extend(unique_pixels)
    # trait_counts_in_masks = Counter(all_traits_in_masks)
    # print("\nSố lần xuất hiện của mỗi ID đặc điểm trong tất cả các mặt nạ:")
    # print(trait_counts_in_masks)

    # --- 6. Phân tích sự Chồng chéo giữa các Tác vụ ---
    print("\n--- 6. Phân tích sự Chồng chéo giữa các Tác vụ ---")
    df['has_species_label'] = df['standardized_species'].notna()
    df['has_trait_labels'] = df[TRAIT_COLUMNS].notna().any(axis=1)
    df['has_mask'] = df['mask_path'].notna()

    overlap_counts = df.groupby(['has_species_label', 'has_trait_labels', 'has_mask']).size().reset_index(name='count')
    print("\nSố lượng mẫu dựa trên sự kết hợp các tác vụ:")
    print(overlap_counts.to_string())
    # ... (giữ nguyên tất cả các phần code từ 1 đến 6) ...

    # --- 7. Phân tích sự Đầy đủ Nhãn theo Loài ---
    print("\n--- 7. Phân tích sự Đầy đủ Nhãn theo Loài ---")
    
    # Đảm bảo các cột boolean đã được tạo
    if 'has_trait_labels' not in df.columns:
        df['has_trait_labels'] = df[TRAIT_COLUMNS].notna().any(axis=1)
    if 'has_mask' not in df.columns:
        df['has_mask'] = df['mask_path'].notna()

    # Nhóm theo loài và kiểm tra sự tồn tại của các loại nhãn
    # .any() sẽ trả về True nếu có ít nhất một hàng trong nhóm là True
    species_label_completeness = df.groupby('standardized_species').agg(
        total_images=('filename', 'count'),
        has_any_trait_label=('has_trait_labels', 'any'),
        has_any_mask_label=('has_mask', 'any')
    ).reset_index()

    # --- Phân loại các loài ---
    # Loài có đủ 3 loại nhãn (species label luôn có, chỉ cần check trait và mask)
    full_label_species = species_label_completeness[
        (species_label_completeness['has_any_trait_label'] == True) & 
        (species_label_completeness['has_any_mask_label'] == True)
    ]

    # Loài thiếu nhãn trait
    missing_trait_species = species_label_completeness[
        species_label_completeness['has_any_trait_label'] == False
    ]

    # Loài thiếu nhãn mask
    missing_mask_species = species_label_completeness[
        species_label_completeness['has_any_mask_label'] == False
    ]
    
    # Loài thiếu cả hai
    # === SỬA LỖI Ở ĐÂY ===
    missing_both_species = species_label_completeness[
        (species_label_completeness['has_any_trait_label'] == False) & 
        (species_label_completeness['has_any_mask_label'] == False)
    ]
    # ======================
    
    print("\nThống kê sự đầy đủ nhãn trên toàn bộ các loài:")
    print(f"  - Tổng số loài: {len(species_label_completeness)}")
    print(f"  - Số loài có ít nhất 1 ảnh với nhãn TRAIT và 1 ảnh với nhãn MASK: {len(full_label_species)}")
    print(f"  - Số loài KHÔNG có bất kỳ ảnh nào có nhãn TRAIT: {len(missing_trait_species)}")
    print(f"  - Số loài KHÔNG có bất kỳ ảnh nào có nhãn MASK: {len(missing_mask_species)}")
    print(f"  - Số loài thiếu cả hai loại nhãn (TRAIT và MASK): {len(missing_both_species)}")
    
    # --- Lưu danh sách các loài thiếu nhãn để tiện cho việc gán nhãn sau này ---
    output_filename_missing_traits = os.path.join(OUTPUT_DIR, "species_missing_trait_labels.csv")
    missing_trait_species.to_csv(output_filename_missing_traits, index=False)
    print(f"\nĐã lưu danh sách các loài thiếu nhãn TRAIT tại: {output_filename_missing_traits}")
    
    output_filename_missing_masks = os.path.join(OUTPUT_DIR, "species_missing_mask_labels.csv")
    missing_mask_species.to_csv(output_filename_missing_masks, index=False)
    print(f"Đã lưu danh sách các loài thiếu nhãn MASK tại: {output_filename_missing_masks}")
    
    # --- Tìm các loài "Ứng cử viên" tốt nhất để gán nhãn bổ sung ---
    # Ưu tiên các loài có nhiều ảnh nhưng lại thiếu nhãn
    
    print("\n--- Các loài 'Ứng cử viên' tiềm năng để gán nhãn bổ sung ---")
    
    # Ứng cử viên cho gán nhãn TRAIT: loài có nhiều ảnh, có MASK nhưng thiếu TRAIT
    candidates_for_trait_labeling = species_label_completeness[
        (species_label_completeness['has_any_trait_label'] == False) & 
        (species_label_completeness['has_any_mask_label'] == True)
    ].sort_values(by='total_images', ascending=False)
    
    print(f"\nTop 10 loài ứng cử viên để gán nhãn TRAIT (có mask sẵn):")
    if not candidates_for_trait_labeling.empty:
        print(candidates_for_trait_labeling.head(10).to_string())
    else:
        print("Không có loài nào phù hợp.")

    # Ứng cử viên cho gán nhãn MASK: loài có nhiều ảnh, có TRAIT nhưng thiếu MASK
    candidates_for_mask_labeling = species_label_completeness[
        (species_label_completeness['has_any_trait_label'] == True) & 
        (species_label_completeness['has_any_mask_label'] == False)
    ].sort_values(by='total_images', ascending=False)
    
    print(f"\nTop 10 loài ứng cử viên để gán nhãn MASK (có trait sẵn):")
    if not candidates_for_mask_labeling.empty:
        print(candidates_for_mask_labeling.head(10).to_string())
    else:
        print("Không có loài nào phù hợp.")

# --- File đầy đủ ---
if __name__ == "__main__":
    main()