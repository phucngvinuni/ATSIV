import torch
import pandas as pd
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Cấu hình ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MASTER_CSV_PATH = "master_dataset.csv"
SPLIT_NAME = "test_manual"
MT_MODEL_PATH = "./checkpoints_improved/best_model_improved.pth"
OUTPUT_DIR = "error_analysis_manual_test"
NUM_SAMPLES_TO_VISUALIZE = 10
IMG_SIZE = 224

# Import model của bạn (cần file multitask_model.py)
try:
    from multitask_model import MultiTaskSwinTransformer
except ImportError:
    print("Lỗi: Không tìm thấy file multitask_model.py. Vui lòng đặt script này ở thư mục gốc.")
    exit()

def visualize_predictions(model, df_samples, species_to_id):
    """
    Trực quan hóa dự đoán của model trên các mẫu được chọn.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    id_to_species = {i: s for s, i in species_to_id.items()}
    trait_names = ['adipose_fin', 'pelvic_fin', 'barbel', 'multiple_dorsal_fin']

    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"\nBắt đầu trực quan hóa {len(df_samples)} mẫu vào thư mục '{OUTPUT_DIR}'...")

    for i, row in df_samples.iterrows():
        image_path = row['image_path']
        filename = os.path.basename(image_path)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Bỏ qua: không tìm thấy ảnh {image_path}")
            continue

        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)

        # --- Lấy kết quả dự đoán ---
        pred_species_id = outputs['species'].argmax(dim=1).item()
        pred_species_name = id_to_species.get(pred_species_id, "Unknown")
        
        pred_traits = torch.sigmoid(outputs['traits']).squeeze().cpu().numpy()
        
        # --- Lấy nhãn thực tế ---
        true_species_name = row['standardized_species']
        true_traits = row[trait_names].values.astype(float)
        
        # --- Tạo hình ảnh trực quan ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image.resize((IMG_SIZE * 2, IMG_SIZE * 2))) # Hiển thị ảnh lớn hơn cho rõ
        ax.axis('off')

        info_text = f"File: {filename}\n\n--- Species Classification ---\n"
        info_text += f"Ground Truth: {true_species_name}\n"
        info_text += f"Prediction:   {pred_species_name}\n"
        is_species_correct = (true_species_name == pred_species_name)
        species_color = 'green' if is_species_correct else 'red'

        info_text += f"\n--- Trait Identification ---\n"
        info_text += f"{'Trait':<22} {'GT':<5} {'Pred':<7}\n"
        info_text += "-" * 40 + "\n"
        
        all_traits_correct = True
        trait_colors = []
        for j, name in enumerate(trait_names):
            gt = true_traits[j]
            pred_val = pred_traits[j]
            pred_label = 1.0 if pred_val > 0.5 else 0.0
            
            # Bỏ qua so sánh nếu GT là NaN hoặc -1
            is_trait_correct = True
            if not np.isnan(gt) and gt != -1.0:
                is_trait_correct = (gt == pred_label)
            
            if not is_trait_correct:
                all_traits_correct = False
                
            trait_colors.append('green' if is_trait_correct else 'red')
            info_text += f"{name:<22} {gt:<5.1f} {pred_val:<7.3f}\n"

        # Đặt box text với màu nền dựa trên kết quả
        overall_correct = is_species_correct and all_traits_correct
        box_color = 'lightgreen' if overall_correct else 'salmon'
        
        ax.text(1.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.4))
        
        plt.title(f"Prediction Analysis (Correct: {overall_correct})", color='green' if overall_correct else 'red')
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, f"analysis_{i}_{'CORRECT' if overall_correct else 'WRONG'}_{filename}")
        plt.savefig(save_path, dpi=150)
        plt.close()

    print(f"Hoàn thành! Hình ảnh phân tích lỗi đã được lưu vào '{OUTPUT_DIR}'.")

def main():
    # Load data
    df = pd.read_csv(MASTER_CSV_PATH, low_memory=False)
    manual_df = df[df['split'] == SPLIT_NAME].dropna(subset=['image_path', 'standardized_species'])
    
    # Lấy toàn bộ danh sách loài để tạo mapping
    all_species = sorted(df['standardized_species'].dropna().unique())
    species_to_id = {s: i for i, s in enumerate(all_species)}
    num_species = len(species_to_id)

    # Load model
    print("Đang tải model...")
    model = MultiTaskSwinTransformer(num_species=num_species, num_traits=4, num_seg_classes=10)
    try:
        model.load_state_dict(torch.load(MT_MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Lỗi: không tìm thấy file checkpoint '{MT_MODEL_PATH}'")
        return
        
    model.to(DEVICE).eval()
    
    # Lấy một vài mẫu ngẫu nhiên để trực quan hóa
    samples_to_check = manual_df.sample(n=min(NUM_SAMPLES_TO_VISUALIZE, len(manual_df)), random_state=42)
    
    visualize_predictions(model, samples_to_check, species_to_id)


if __name__ == "__main__":
    main()