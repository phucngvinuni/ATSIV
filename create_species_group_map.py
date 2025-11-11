import pandas as pd
import json

MASTER_CSV_PATH = "master_dataset.csv"
OUTPUT_JSON_PATH = "species_to_group.json"

def main():
    print("Đang tạo file mapping từ loài sang nhóm (Major/Minor/Ultra-Rare)...")
    
    # Load master dataset
    df = pd.read_csv(MASTER_CSV_PATH, low_memory=False)
    
    # Lọc ra chỉ tập train
    train_df = df[df['split'] == 'train'].dropna(subset=['standardized_species'])
    
    # Đếm số lượng mẫu cho mỗi loài trong tập train
    species_counts = train_df['standardized_species'].value_counts()
    
    # Tạo mapping từ tên loài sang ID (để nhất quán với dataset class)
    all_species = sorted(df['standardized_species'].dropna().unique())
    species_to_id = {species: i for i, species in enumerate(all_species)}
    
    # Tạo mapping từ ID của loài sang nhóm
    species_id_to_group = {}
    for species_name, count in species_counts.items():
        species_id = species_to_id.get(species_name)
        if species_id is not None:
            if count >= 500:
                group = "Majority"
            elif 100 <= count < 500:
                group = "Neutral"
            elif 10 <= count < 100:
                group = "Minority"
            else: # count < 10
                group = "Ultra-Rare"
            species_id_to_group[str(species_id)] = group # Chuyển ID sang string để làm key JSON
            
    # Lưu file JSON
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(species_id_to_group, f, indent=4)
        
    print(f"Hoàn thành! Đã lưu file mapping tại: {OUTPUT_JSON_PATH}")
    
    # In ra một vài thống kê
    group_counts = pd.Series(species_id_to_group.values()).value_counts()
    print("\nSố lượng loài trong mỗi nhóm:")
    print(group_counts)

if __name__ == "__main__":
    main()