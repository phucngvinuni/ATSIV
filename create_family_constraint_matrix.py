import pandas as pd
import numpy as np
import torch
import json
from tqdm import tqdm

MASTER_CSV_PATH = "master_dataset.csv"
FAMILY_MAP_PATH = "family_to_id.json" # Đầu vào
OUTPUT_PATH = "family_trait_constraints.pt" # Đầu ra
TRAIT_COLUMNS = ['adipose_fin', 'pelvic_fin', 'barbel', 'multiple_dorsal_fin']
MIN_SPECIES_FOR_CONSTRAINT = 3 # Cần ít nhất 3 loài trong một họ để đưa ra kết luận
MIN_SAMPLES_PER_SPECIES = 5 # Mỗi loài phải có ít nhất 5 mẫu

def main():
    print("Bắt đầu tạo Ma trận Ràng buộc Họ-Đặc điểm...")
    
    df = pd.read_csv(MASTER_CSV_PATH, low_memory=False)
    with open(FAMILY_MAP_PATH, 'r') as f:
        family_to_id = json.load(f)
        
    num_families = len(family_to_id)
    num_traits = len(TRAIT_COLUMNS)
    
    constraint_matrix = torch.full((num_families, num_traits), -1.0, dtype=torch.float)
    
    for family_name, family_id in tqdm(family_to_id.items(), desc="Processing families"):
        family_df = df[df['family'] == family_name]
        species_in_family = family_df['standardized_species'].dropna().unique()

        if len(species_in_family) < MIN_SPECIES_FOR_CONSTRAINT:
            continue

        for i, trait in enumerate(TRAIT_COLUMNS):
            trait_values_for_family = []
            is_conclusive = True
            
            for species in species_in_family:
                species_df = family_df[family_df['standardized_species'] == species]
                trait_labels = species_df[trait].dropna()

                if len(trait_labels) < MIN_SAMPLES_PER_SPECIES:
                    continue # Bỏ qua loài không đủ mẫu

                unique_labels = trait_labels.unique()
                if len(unique_labels) == 1:
                    trait_values_for_family.append(unique_labels[0])
                else:
                    # Nếu một loài trong họ có cả 0 và 1, thì họ đó không có ràng buộc
                    is_conclusive = False
                    break
            
            if not is_conclusive or not trait_values_for_family:
                continue

            # Kiểm tra xem tất cả các loài trong họ có cùng 1 giá trị trait không
            unique_family_trait_values = np.unique(trait_values_for_family)
            if len(unique_family_trait_values) == 1:
                label = unique_family_trait_values[0]
                if label == 1.0:
                    constraint_matrix[family_id, i] = 1.0
                elif label == 0.0:
                    constraint_matrix[family_id, i] = 0.0
    
    torch.save(constraint_matrix, OUTPUT_PATH)
    print(f"\nHoàn thành! Đã lưu ma trận ràng buộc cho họ tại: {OUTPUT_PATH}")
    
    num_must_have = (constraint_matrix == 1.0).sum().item()
    num_must_not_have = (constraint_matrix == 0.0).sum().item()
    total_constraints = num_must_have + num_must_not_have
    print(f"Tổng số ràng buộc cấp độ họ được tìm thấy: {total_constraints}")
    print(f"  - Bắt buộc có: {num_must_have}")
    print(f"  - Bắt buộc không có: {num_must_not_have}")

if __name__ == "__main__":
    main()