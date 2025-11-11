import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import json
import timm
from sklearn.metrics import accuracy_score
from collections import defaultdict

# Import các hàm/class cần thiết
try:
    from multitask_dataset import FishVistaMultiTaskDataset
except ImportError:
    print("Lỗi: Không tìm thấy file multitask_dataset.py.")
    exit()

# --- Cấu hình ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MASTER_CSV = "master_dataset.csv"
SPECIES_GROUP_MAP_JSON = "species_to_group.json"
IMG_SIZE = 224
BATCH_SIZE = 32 # Tăng batch size để đánh giá nhanh hơn
NUM_WORKERS = 4

# --- THÔNG TIN MODEL CẦN ĐÁNH GIÁ ---
MODEL_NAME = 'maxvit_base_tf_224.in1k'
CHECKPOINT_DIR = "./checkpoints_cls_only_noweightsbase"
CHECKPOINT_FILENAME = f"{MODEL_NAME.replace('.', '_')}_best.pth"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)

RESULTS_FILE = f"evaluation_results_{MODEL_NAME.replace('.', '_')}.json"

def evaluate_classification(model, loader, split_name, species_id_to_group):
    """Hàm đánh giá chỉ cho tác vụ classification."""
    model.eval()
    
    all_preds, all_labels = [], []

    print(f"\n--- Evaluating on '{split_name}' split ---")
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
            # Dùng collate_fn đơn giản hơn vì không cần các cờ 'has_...'
            images, labels = batch
            
            if images.shape[0] == 0: continue

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            with torch.amp.autocast(device_type=DEVICE.split(':')[0]):
                outputs = model(images)
            
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    results = {}

    if not all_labels:
        print(f"[{split_name}] No valid samples found to evaluate.")
        return results

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    
    # Tính accuracy tổng thể
    species_acc_total = accuracy_score(labels, preds)
    results['species_accuracy_total'] = species_acc_total
    print(f"[{split_name}] Species Accuracy (Total): {species_acc_total:.4f}")

    # Tính accuracy theo nhóm
    group_correct = defaultdict(int)
    group_total = defaultdict(int)
    
    for i in range(len(labels)):
        label_id = labels[i]
        pred_id = preds[i]
        group = species_id_to_group.get(str(label_id), "Unknown")
        group_total[group] += 1
        if label_id == pred_id:
            group_correct[group] += 1

    group_accuracies = {}
    for group, total_count in sorted(group_total.items()):
        if total_count > 0:
            acc = group_correct[group] / total_count
            group_accuracies[group] = acc
            print(f"[{split_name}] Species Accuracy ({group}): {acc:.4f} ({group_correct[group]}/{total_count})")
    
    results['species_accuracy_per_group'] = group_accuracies
    
    return results

def main():
    print("Loading data and group mappings...")
    master_df = pd.read_csv(MASTER_CSV, low_memory=False)
    
    try:
        with open(SPECIES_GROUP_MAP_JSON, 'r') as f:
            species_id_to_group = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{SPECIES_GROUP_MAP_JSON}'.")
        print("Vui lòng chạy script 'create_species_group_map.py' trước.")
        return

    num_species = master_df['standardized_species'].nunique()
    print(f"Found {num_species} species.")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Tạo collate_fn để chỉ lấy ảnh và nhãn loài
    def collate_fn_eval(batch):
        valid_items = [item for item in batch if item['has_species_label']]
        if not valid_items:
            return torch.empty(0, 3, IMG_SIZE, IMG_SIZE), torch.empty(0, dtype=torch.long)
        images = torch.stack([item['image'] for item in valid_items])
        labels = torch.stack([item['species_label'] for item in valid_items])
        return images, labels

    full_dataset = FishVistaMultiTaskDataset(csv_file=MASTER_CSV, transform=transform)

    test_splits = {
        'test': master_df[master_df['split'] == 'test'].index.tolist(),
        'test_ood': master_df[master_df['split'] == 'test_ood'].index.tolist(),
        'test_manual': master_df[master_df['split'] == 'test_manual'].index.tolist(),
    }

    test_loaders = {}
    for name, indices in test_splits.items():
        if indices:
            subset = Subset(full_dataset, indices)
            test_loaders[name] = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_eval)

    # --- 2. Load Model ---
    print(f"Loading model from checkpoint: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at '{CHECKPOINT_PATH}'")
        return
        
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_species).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print("Model loaded successfully.")

    # --- 3. Run Evaluation ---
    final_results = {}
    for split_name, loader in test_loaders.items():
        results = evaluate_classification(model, loader, split_name, species_id_to_group)
        final_results[split_name] = results
        
    # --- 4. Save Results ---
    with open(RESULTS_FILE, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\nEvaluation complete. All results saved to '{RESULTS_FILE}'.")
    print(json.dumps(final_results, indent=4))

if __name__ == "__main__":
    main()