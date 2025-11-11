import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import json
from sklearn.metrics import average_precision_score, accuracy_score
from torchmetrics import JaccardIndex
from collections import defaultdict

from multitask_dataset import FishVistaMultiTaskDataset
from multitask_model import MultiTaskSwinTransformer

# --- Cấu hình ---
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
MASTER_CSV = "master_dataset.csv"
SPECIES_GROUP_MAP_JSON = "species_to_group.json" # File mapping mới
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 4
CHECKPOINT_PATH = "checkpoints_improved/best_model_improved.pth"
RESULTS_FILE = "best_evaluation_results_improved.json"

NUM_TRAITS = 4
NUM_SEG_CLASSES = 10

def evaluate_on_split(model, loader, num_species, split_name, species_id_to_group):
    """Hàm đánh giá đã được cập nhật để tính accuracy theo nhóm."""
    model.eval()
    
    all_species_preds, all_species_labels = [], []
    all_trait_preds, all_trait_labels = {}, {}
    jaccard = JaccardIndex(task="multiclass", num_classes=NUM_SEG_CLASSES, ignore_index=0).to(DEVICE)

    print(f"\n--- Evaluating on '{split_name}' split ---")
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
            images = batch['image'].to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE.split(':')[0]):
                outputs = model(images)

            if batch['has_species_label'].any():
                species_labels = batch['species_label']
                species_mask = (species_labels != -1)
                if species_mask.any():
                    all_species_preds.append(outputs['species'][species_mask].argmax(dim=1).cpu())
                    all_species_labels.append(species_labels[species_mask].cpu())

            # (Phần xử lý trait và segmentation giữ nguyên)
            if batch['has_trait_labels'].any():
                trait_labels = batch['trait_labels']
                trait_mask = (trait_labels != -1.0)
                if trait_mask.any():
                    trait_preds_sig = torch.sigmoid(outputs['traits'])
                    for i in range(NUM_TRAITS):
                        mask_i = trait_mask[:, i]
                        if mask_i.any():
                            if i not in all_trait_preds:
                                all_trait_preds[i] = []
                                all_trait_labels[i] = []
                            all_trait_preds[i].append(trait_preds_sig[mask_i, i].cpu().numpy())
                            all_trait_labels[i].append(trait_labels[mask_i, i].cpu().numpy())

            if batch['has_mask'].any():
                seg_masks = batch['segmentation_mask'].to(DEVICE)
                seg_mask_flag = batch['has_mask'].to(DEVICE)
                if seg_mask_flag.any():
                    seg_preds = outputs['segmentation'][seg_mask_flag].argmax(dim=1)
                    jaccard.update(seg_preds, seg_masks[seg_mask_flag])

    results = {}

    # --- TÍNH TOÁN ACCURACY THEO NHÓM ---
    if all_species_labels:
        preds = torch.cat(all_species_preds).numpy()
        labels = torch.cat(all_species_labels).numpy()
        
        # Tính accuracy tổng thể
        species_acc_total = accuracy_score(labels, preds)
        results['species_accuracy_total'] = species_acc_total
        print(f"[{split_name}] Species Accuracy (Total): {species_acc_total:.4f}")

        # Chuẩn bị để tính accuracy theo nhóm
        group_correct = defaultdict(int)
        group_total = defaultdict(int)
        
        for i in range(len(labels)):
            label_id = labels[i]
            pred_id = preds[i]
            
            group = species_id_to_group.get(str(label_id), "Unknown") # Lấy nhóm của loài
            
            group_total[group] += 1
            if label_id == pred_id:
                group_correct[group] += 1

        # Tính và lưu kết quả
        group_accuracies = {}
        for group, total_count in group_total.items():
            if total_count > 0:
                acc = group_correct[group] / total_count
                group_accuracies[group] = acc
                print(f"[{split_name}] Species Accuracy ({group}): {acc:.4f} ({group_correct[group]}/{total_count})")
        
        results['species_accuracy_per_group'] = group_accuracies

    # (Phần kết quả trait và segmentation giữ nguyên)
    if all_trait_labels:
        aps = []
        trait_names = ['adipose_fin', 'pelvic_fin', 'barbel', 'multiple_dorsal_fin']
        for i in range(NUM_TRAITS):
            if i in all_trait_labels:
                try:
                    ap = average_precision_score(np.concatenate(all_trait_labels[i]), np.concatenate(all_trait_preds[i]))
                    aps.append(ap)
                    results[f'trait_ap_{trait_names[i]}'] = ap
                except ValueError:
                    aps.append(0.0)
        
        if aps:
            mean_ap = np.mean(aps)
            results['trait_mAP'] = mean_ap
            print(f"[{split_name}] Trait mAP: {mean_ap:.4f}")

    mIoU = jaccard.compute().item()
    results['segmentation_mIoU'] = mIoU
    print(f"[{split_name}] Segmentation mIoU (excluding background): {mIoU:.4f}")
    
    return results

def main():
    print("Loading data and group mappings...")
    master_df = pd.read_csv(MASTER_CSV, low_memory=False)
    
    # Load file mapping nhóm loài
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
            test_loaders[name] = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Loading model from checkpoint: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at '{CHECKPOINT_PATH}'")
        return
        
    model = MultiTaskSwinTransformer(
        num_species=num_species,
        num_traits=NUM_TRAITS,
        num_seg_classes=NUM_SEG_CLASSES
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print("Model loaded successfully.")

    final_results = {}
    for split_name, loader in test_loaders.items():
        results = evaluate_on_split(model, loader, num_species, split_name, species_id_to_group)
        final_results[split_name] = results
        
    with open(RESULTS_FILE, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\nEvaluation complete. All results saved to '{RESULTS_FILE}'.")
    print(json.dumps(final_results, indent=4))

if __name__ == "__main__":
    main()