import pandas as pd
import numpy as np
import torch
import json
from tqdm import tqdm

MASTER_CSV_PATH = "master_dataset.csv"
SPECIES_MAP_PATH = "species_to_id.json"
OUTPUT_PATH = "species_trait_constraints.pt"
TRAIT_COLUMNS = ['adipose_fin', 'pelvic_fin', 'barbel', 'multiple_dorsal_fin']
MIN_SAMPLES_FOR_CONSTRAINT = 5 # Cần ít nhất 5 mẫu để đưa ra kết luận chắc chắn

def main():
    print("Bắt đầu tạo Ma trận Ràng buộc Loài-Đặc điểm...")
    
    df = pd.read_csv(MASTER_CSV_PATH, low_memory=False)
    with open(SPECIES_MAP_PATH, 'r') as f:
        species_to_id = json.load(f)
        
    num_species = len(species_to_id)
    num_traits = len(TRAIT_COLUMNS)
    
    # Khởi tạo ma trận với -1 (không có ràng buộc)
    constraint_matrix = torch.full((num_species, num_traits), -1.0, dtype=torch.float)
    
    for species_name, species_id in tqdm(species_to_id.items(), desc="Processing species"):
        # Lọc ra các hàng của loài hiện tại có nhãn trait
        species_df = df[(df['standardized_species'] == species_name) & (df[TRAIT_COLUMNS].notna().any(axis=1))]
        
        if len(species_df) < MIN_SAMPLES_FOR_CONSTRAINT:
            continue
            
        for i, trait in enumerate(TRAIT_COLUMNS):
            # Lấy các nhãn hợp lệ cho đặc điểm này
            trait_labels = species_df[trait].dropna()
            
            if len(trait_labels) < MIN_SAMPLES_FOR_CONSTRAINT:
                continue

            unique_labels = trait_labels.unique()
            
            if len(unique_labels) == 1:
                label = unique_labels[0]
                if label == 1.0:
                    constraint_matrix[species_id, i] = 1.0
                elif label == 0.0:
                    constraint_matrix[species_id, i] = 0.0
    
    torch.save(constraint_matrix, OUTPUT_PATH)
    print(f"\nHoàn thành! Đã lưu ma trận ràng buộc tại: {OUTPUT_PATH}")
    
    # In ra một vài thống kê
    num_must_have = (constraint_matrix == 1.0).sum().item()
    num_must_not_have = (constraint_matrix == 0.0).sum().item()
    total_constraints = num_must_have + num_must_not_have
    print(f"Tổng số ràng buộc được tìm thấy: {total_constraints}")
    print(f"  - Bắt buộc có: {num_must_have}")
    print(f"  - Bắt buộc không có: {num_must_not_have}")

if __name__ == "__main__":
    main()