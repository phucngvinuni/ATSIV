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
from sklearn.metrics import average_precision_score, accuracy_score, f1_score
from torchmetrics import JaccardIndex
from collections import defaultdict

# --- Import các module cần thiết từ dự án của bạn ---
try:
    from multitask_dataset import FishVistaMultiTaskDataset
    from multitask_model import MultiTaskSwinTransformer
except ImportError as e:
    print(f"Lỗi import: {e}")
    print("Hãy đảm bảo script này được chạy từ thư mục gốc của dự án và các file cần thiết tồn tại.")
    exit()

# ==============================================================================
# 1. CẤU HÌNH CHUNG
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MASTER_CSV = "master_dataset.csv"
SPECIES_GROUP_MAP_JSON = "species_to_group.json"
IMG_SIZE = 224
BATCH_SIZE = 32 # Tăng batch size để đánh giá nhanh hơn
NUM_WORKERS = 4
OUTPUT_FILE = "all_models_evaluation_summary.json"

NUM_TRAITS = 4
NUM_SEG_CLASSES = 10 # 9 classes + 1 background

# ==============================================================================
# 2. ĐỊNH NGHĨA CÁC MÔ HÌNH CẦN ĐÁNH GIÁ
# CẬP NHẬT ĐƯỜNG DẪN CHECKPOINT CỦA BẠN TẠI ĐÂY!
# ==============================================================================
MODELS_TO_EVALUATE = {
    "MT-Swin-Improved": {
        "type": "multitask",
        "checkpoint_path": "./checkpoints_improved/best_model_improved.pth",
        "model_class": MultiTaskSwinTransformer,
        "init_params": {"num_species": -1, "num_traits": NUM_TRAITS, "num_seg_classes": NUM_SEG_CLASSES} # num_species sẽ được cập nhật sau
    },
    "Baseline-MaxViT-Base": {
        "type": "classification_only",
        "checkpoint_path": "./checkpoints_cls_only_noweightsbase/maxvit_base_tf_224_in1k_best.pth",
        "model_name": "maxvit_base_tf_224.in1k"
    },
    # Thêm các model khác nếu bạn muốn, ví dụ:
    "Baseline-MaxViT-Tiny": {
        "type": "classification_only",
        "checkpoint_path": "./path/to/your/maxvit_tiny_best.pth",
        "model_name": "maxvit_tiny_tf_224.in1k"
    },
    # "MT-Swin-Original": {
    #     "type": "multitask",
    #     "checkpoint_path": "./checkpoints/best_model.pth",
    #     "model_class": MultiTaskSwinTransformer,
    #     "init_params": {"num_species": -1, "num_traits": NUM_TRAITS, "num_seg_classes": NUM_SEG_CLASSES}
    # }
}

# ==============================================================================
# 3. HÀM ĐÁNH GIÁ (TỔNG HỢP VÀ LINH HOẠT)
# ==============================================================================
def universal_evaluate(model, model_type, loader, split_name, species_id_to_group, num_species):
    """
    Hàm đánh giá chung, có khả năng xử lý cả model đa tác vụ và chỉ phân loại.
    """
    model.eval()
    
    # Khởi tạo các biến lưu kết quả
    all_species_preds, all_species_labels = [], []
    all_trait_preds, all_trait_labels = {}, {}
    jaccard = JaccardIndex(task="multiclass", num_classes=NUM_SEG_CLASSES, ignore_index=0).to(DEVICE)
    results = {}

    print(f"--- Evaluating '{split_name}' split ---")
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
            images = batch['image'].to(DEVICE)
            
            with torch.amp.autocast(device_type=DEVICE.split(':')[0]):
                if model_type == 'multitask':
                    outputs = model(images)
                else: # classification_only
                    # Giả lập output của model đa tác vụ để tái sử dụng logic
                    species_logits = model(images)
                    outputs = {
                        'species': species_logits,
                        'traits': torch.zeros(images.size(0), NUM_TRAITS, device=DEVICE),
                        'segmentation': torch.zeros(images.size(0), NUM_SEG_CLASSES, *images.shape[2:], device=DEVICE)
                    }

            # --- Thu thập dữ liệu cho metrics ---
            if batch['has_species_label'].any():
                species_labels = batch['species_label']
                mask = (species_labels != -1)
                if mask.any():
                    all_species_preds.append(outputs['species'][mask].argmax(dim=1).cpu())
                    all_species_labels.append(species_labels[mask].cpu())

            if model_type == 'multitask' and batch['has_trait_labels'].any():
                trait_labels = batch['trait_labels']
                mask = (trait_labels != -1.0)
                if mask.any():
                    preds_sig = torch.sigmoid(outputs['traits'])
                    for i in range(NUM_TRAITS):
                        mask_i = mask[:, i]
                        if mask_i.any():
                            if i not in all_trait_preds:
                                all_trait_preds[i], all_trait_labels[i] = [], []
                            all_trait_preds[i].append(preds_sig[mask_i, i].cpu().numpy())
                            all_trait_labels[i].append(trait_labels[mask_i, i].cpu().numpy())
            
            if model_type == 'multitask' and batch['has_mask'].any():
                seg_masks = batch['segmentation_mask'].to(DEVICE)
                mask = batch['has_mask']
                if mask.any():
                    seg_preds = outputs['segmentation'][mask].argmax(dim=1)
                    jaccard.update(seg_preds, seg_masks[mask])

    # --- Tính toán metrics ---
    # 1. Species Classification
    if all_species_labels:
        preds = torch.cat(all_species_preds).numpy()
        labels = torch.cat(all_species_labels).numpy()
        
        results['species_accuracy_total'] = accuracy_score(labels, preds)
        results['species_f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
        
        group_correct, group_total = defaultdict(int), defaultdict(int)
        for i in range(len(labels)):
            group = species_id_to_group.get(str(labels[i]), "Unknown")
            group_total[group] += 1
            if labels[i] == preds[i]:
                group_correct[group] += 1
        
        group_accuracies = {group: (group_correct[group] / total if total > 0 else 0) for group, total in sorted(group_total.items())}
        results['species_accuracy_per_group'] = group_accuracies

    # 2. Trait Identification
    if model_type == 'multitask' and all_trait_labels:
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
                    results[f'trait_ap_{trait_names[i]}'] = 0.0
        if aps:
            results['trait_mAP'] = np.mean(aps)

    # 3. Trait Segmentation
    if model_type == 'multitask':
        mIoU = jaccard.compute().item()
        results['segmentation_mIoU'] = mIoU if mIoU > 0 else 0.0 # Trả về 0 nếu không có mẫu seg

    return results

# ==============================================================================
# 4. HÀM MAIN ĐIỀU KHIỂN
# ==============================================================================
def main():
    print("=" * 80)
    print("Bắt đầu quy trình đánh giá toàn bộ mô hình")
    print("=" * 80)
    print(f"Sử dụng thiết bị: {DEVICE}")

    # --- 1. Load Dữ liệu và Metadata ---
    print("\n[1/4] Đang tải dữ liệu và metadata...")
    master_df = pd.read_csv(MASTER_CSV, low_memory=False)
    
    try:
        with open(SPECIES_GROUP_MAP_JSON, 'r') as f:
            species_id_to_group = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy '{SPECIES_GROUP_MAP_JSON}'. Vui lòng chạy 'create_species_group_map.py'.")
        return

    num_species = master_df['standardized_species'].nunique()
    print(f"Đã tìm thấy {num_species} loài.")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = FishVistaMultiTaskDataset(csv_file=MASTER_CSV, transform=transform)

    # Xác định các split cần đánh giá
    test_splits_indices = {
        'test': master_df[master_df['split'] == 'test'].index.tolist(),
        'test_ood': master_df[master_df['split'] == 'test_ood'].index.tolist(),
        'test_manual': master_df[master_df['split'] == 'test_manual'].index.tolist(),
    }

    test_loaders = {}
    for name, indices in test_splits_indices.items():
        if indices:
            subset = Subset(full_dataset, indices)
            test_loaders[name] = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print("Đã tạo xong DataLoaders.")

    # --- 2. Lặp qua và Đánh giá từng Model ---
    all_results = {}
    total_models = len(MODELS_TO_EVALUATE)
    
    for i, (model_name, config) in enumerate(MODELS_TO_EVALUATE.items()):
        print("\n" + "-" * 80)
        print(f"[2/4] Đang xử lý Model {i+1}/{total_models}: {model_name}")
        print("-" * 80)

        # --- 2a. Load Model ---
        print(f"Đang tải model từ checkpoint: {config['checkpoint_path']}")
        if not os.path.exists(config['checkpoint_path']):
            print(f"!!! CẢNH BÁO: Không tìm thấy checkpoint cho '{model_name}'. Bỏ qua model này. !!!")
            all_results[model_name] = {"error": "Checkpoint not found"}
            continue
        
        try:
            if config['type'] == 'multitask':
                config['init_params']['num_species'] = num_species
                model = config['model_class'](**config['init_params'])
            else: # classification_only
                model = timm.create_model(config['model_name'], pretrained=False, num_classes=num_species)
            
            model.load_state_dict(torch.load(config['checkpoint_path'], map_location=DEVICE))
            model.to(DEVICE)
            print("Model đã được tải thành công.")
        except Exception as e:
            print(f"!!! LỖI: Không thể tải model '{model_name}'. Lỗi: {e}. Bỏ qua model này. !!!")
            all_results[model_name] = {"error": f"Failed to load model: {e}"}
            continue

        # --- 2b. Chạy Đánh giá trên các Split ---
        model_results = {}
        for split_name, loader in test_loaders.items():
            print(f"\nĐang đánh giá '{model_name}' trên split '{split_name}'...")
            
            # Hàm universal_evaluate sẽ tự xử lý logic cho từng loại model
            split_results = universal_evaluate(
                model=model,
                model_type=config['type'],
                loader=loader,
                split_name=split_name,
                species_id_to_group=species_id_to_group,
                num_species=num_species
            )
            model_results[split_name] = split_results
        
        all_results[model_name] = model_results
        
        # Giải phóng bộ nhớ GPU
        del model
        torch.cuda.empty_cache()

    # --- 3. Lưu Kết quả ---
    print("\n" + "=" * 80)
    print(f"[3/4] Đang lưu tất cả kết quả vào file: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=4)
    print("Lưu thành công.")

    # --- 4. In Kết quả tóm tắt ---
    print("\n[4/4] Bảng tóm tắt kết quả:")
    print("-" * 80)
    print(f"{'Model':<25} | {'Split':<12} | {'Acc Total':<10} | {'F1 Macro':<10} | {'mAP':<10} | {'mIoU':<10}")
    print("-" * 80)
    for model_name, model_results in all_results.items():
        if "error" in model_results:
            print(f"{model_name:<25} | {'ERROR':<59}")
            continue
        for split_name, split_results in model_results.items():
            acc = f"{split_results.get('species_accuracy_total', 0):.2%}"
            f1 = f"{split_results.get('species_f1_macro', 0):.4f}"
            map_val = f"{split_results.get('trait_mAP', 0):.4f}"
            miou = f"{split_results.get('segmentation_mIoU', 0):.4f}"
            print(f"{model_name:<25} | {split_name:<12} | {acc:<10} | {f1:<10} | {map_val:<10} | {miou:<10}")
    print("-" * 80)

if __name__ == "__main__":
    main()