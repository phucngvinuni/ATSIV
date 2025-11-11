import pandas as pd
import numpy as np
import torch
import json

MASTER_CSV_PATH = "master_dataset.csv"
SPECIES_MAP_PATH = "species_to_id.json" # Sẽ được tạo nếu chưa có
WEIGHTS_PATH = "class_weights.pt"
BETA = 0.999 # Hyperparameter, 0.999 là giá trị phổ biến

def main():
    print("Đang tính toán trọng số Class-Balanced...")
    df = pd.read_csv(MASTER_CSV_PATH, low_memory=False)
    
    # Lọc tập train và các loài hợp lệ
    train_df = df[df['split'] == 'train'].dropna(subset=['standardized_species'])
    
    # Tạo/Load species_to_id map
    all_species = sorted(df['standardized_species'].dropna().unique())
    species_to_id = {species: i for i, species in enumerate(all_species)}
    num_classes = len(species_to_id)
    
    with open(SPECIES_MAP_PATH, 'w') as f:
        json.dump(species_to_id, f, indent=4)
    print(f"Đã lưu species_to_id map tại {SPECIES_MAP_PATH}")

    # Đếm số lượng mẫu mỗi lớp
    counts = train_df['standardized_species'].value_counts()
    
    # Sắp xếp lại counts theo đúng thứ tự ID
    class_counts = np.zeros(num_classes)
    for species, count in counts.items():
        class_id = species_to_id[species]
        class_counts[class_id] = count
        
    # Tính toán trọng số Class-Balanced
    # effective_num = 1.0 - np.power(BETA, class_counts)
    # weights = (1.0 - BETA) / np.array(effective_num)
    
    # Một cách tính đơn giản hơn và cũng hiệu quả: Inverse Number of Samples
    weights = 1.0 / np.array(class_counts)
    # Chuẩn hóa để tổng trọng số bằng số lớp
    weights = weights / np.sum(weights) * num_classes

    # Chuyển thành tensor và lưu lại
    weights_tensor = torch.FloatTensor(weights)
    torch.save(weights_tensor, WEIGHTS_PATH)
    print(f"Đã lưu trọng số class tại {WEIGHTS_PATH}")

if __name__ == "__main__":
    main()