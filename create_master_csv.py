import pandas as pd
import os
import json

# --- Cấu hình ---
DATA_DIR = "./" 
IMAGE_DIR = "Images"
SEG_MASK_DIR = "segmentation_masks/images" 
SEG_ID_MAP_FILE = "segmentation_masks/seg_id_trait_map.json"

def main():
    print("Bắt đầu quá trình tạo master dataset...")
    # --- Đọc các file CSV gốc ---
    try:
        df_cls_train = pd.read_csv(os.path.join(DATA_DIR, 'classification_train.csv'), low_memory=False)
        df_cls_val = pd.read_csv(os.path.join(DATA_DIR, 'classification_val.csv'), low_memory=False)
        df_cls_test = pd.read_csv(os.path.join(DATA_DIR, 'classification_test.csv'), low_memory=False)

        df_id_train = pd.read_csv(os.path.join(DATA_DIR, 'identification_train.csv'), low_memory=False)
        df_id_val = pd.read_csv(os.path.join(DATA_DIR, 'identification_val.csv'), low_memory=False)
        df_id_test_in = pd.read_csv(os.path.join(DATA_DIR, 'identification_test_insp.csv'), low_memory=False)
        df_id_test_out = pd.read_csv(os.path.join(DATA_DIR, 'identification_test_lvsp.csv'), low_memory=False)
        df_id_manual = pd.read_csv(os.path.join(DATA_DIR, 'identification_test_manual_annot.csv'), low_memory=False)

        df_seg_train = pd.read_csv(os.path.join(DATA_DIR, 'segmentation_train.csv'), low_memory=False)
        df_seg_val = pd.read_csv(os.path.join(DATA_DIR, 'segmentation_val.csv'), low_memory=False)
        df_seg_test = pd.read_csv(os.path.join(DATA_DIR, 'segmentation_test.csv'), low_memory=False)
        
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file CSV. {e}")
        print("Hãy đảm bảo bạn đang ở đúng thư mục gốc của repo fish-vista và đã tải dataset đầy đủ.")
        return

    # --- Hợp nhất tất cả các DataFrame ---
    print("Đang hợp nhất các file CSV...")
    all_dfs = [
        df_cls_train, df_cls_val, df_cls_test,
        df_id_train, df_id_val, df_id_test_in, df_id_test_out, df_id_manual,
        df_seg_train, df_seg_val, df_seg_test,
    ]
    master_df = pd.concat(all_dfs, ignore_index=True)

    # Gộp thông tin từ các hàng trùng lặp dựa trên 'filename'
    # 'first' sẽ ưu tiên lấy giá trị không-null đầu tiên mà nó gặp
    aggregation_funcs = {col: 'first' for col in master_df.columns if col != 'filename'}
    master_df = master_df.groupby('filename', as_index=False).agg(aggregation_funcs)

    # --- Tạo các cột đường dẫn ---
    print("Đang tạo các cột đường dẫn ảnh và mặt nạ...")
    
    # Sử dụng cột 'file_name' đã có sẵn cho đường dẫn ảnh
    master_df['image_path'] = master_df['file_name']

    def get_seg_path(filename):
        if pd.isna(filename):
            return None
        base_name = os.path.splitext(str(filename))[0]
        path = os.path.join(SEG_MASK_DIR, f"{base_name}.png")
        if os.path.exists(path):
            return path
        return None

    master_df['mask_path'] = master_df['filename'].apply(get_seg_path)

    # --- Xác định lại split cho từng ảnh ---
    print("Đang xác định lại các tập train/val/test...")
    filename_to_split = {}

    def update_split(df, split_name):
        for filename in df['filename']:
            if pd.notna(filename):
                filename_to_split[filename] = split_name

    update_split(df_cls_train, 'train'); update_split(df_id_train, 'train'); update_split(df_seg_train, 'train')
    update_split(df_cls_val, 'val'); update_split(df_id_val, 'val'); update_split(df_seg_val, 'val')
    update_split(df_cls_test, 'test'); update_split(df_id_test_in, 'test'); update_split(df_seg_test, 'test')
    update_split(df_id_test_out, 'test_ood') # Out-of-distribution
    update_split(df_id_manual, 'test_manual') # Test set có chú thích tay

    master_df['split'] = master_df['filename'].map(filename_to_split)

    # --- Lưu file và In kết quả ---
    output_filename = 'master_dataset.csv'
    master_df.to_csv(output_filename, index=False)
    
    print("\n-------------------------------------------")
    print(f"Hoàn thành! Đã tạo file master dataset tại: {output_filename}")
    print("-------------------------------------------")

    print(f"\nTổng số ảnh duy nhất: {len(master_df)}")
    
    print("\nSố lượng ảnh trong mỗi split:")
    print(master_df['split'].value_counts())
    
    num_masks_found = master_df['mask_path'].notna().sum()
    print(f"\nSố lượng mặt nạ segmentation được tìm thấy: {num_masks_found}")
    if num_masks_found == 0:
        print("CẢNH BÁO: Không tìm thấy file mask nào. Vui lòng kiểm tra lại đường dẫn SEG_MASK_DIR.")

    # In ra mapping ID-Trait từ file JSON
    try:
        with open(SEG_ID_MAP_FILE, 'r') as f:
            seg_id_map = json.load(f)
        print("\nMapping ID -> Trait cho tác vụ Segmentation:")
        print(seg_id_map)
    except FileNotFoundError:
        print(f"\nCảnh báo: Không tìm thấy file '{SEG_ID_MAP_FILE}'.")


if __name__ == "__main__":
    main()