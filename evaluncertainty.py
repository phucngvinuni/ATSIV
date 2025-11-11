# evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import json
import argparse
from sklearn.metrics import average_precision_score, accuracy_score, f1_score
from torchmetrics import JaccardIndex
import torch.nn.functional as F
from collections import defaultdict
import timm

# ============================================================
# 1. SAO CHÉP CÁC ĐỊNH NGHĨA CẦN THIẾT TỪ SCRIPT TRAINING
# ============================================================

# --- Định nghĩa Model ---
class FPNHead(nn.Module):
    # (Sao chép y hệt class FPNHead từ script training)
    def __init__(self, in_channels_list, out_channels):
        super(FPNHead, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.output_convs.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
    def forward(self, features):
        laterals = [conv(features[i]) for i, conv in enumerate(self.lateral_convs)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(laterals[i], size=laterals[i-1].shape[2:], mode='bilinear', align_corners=False)
        outputs = [self.output_convs[i](laterals[i]) for i in range(len(laterals))]
        h, w = outputs[0].shape[2:]
        for i in range(1, len(outputs)):
            outputs[i] = F.interpolate(outputs[i], size=(h, w), mode='bilinear', align_corners=False)
        return torch.cat(outputs, dim=1)

class MultiTaskSwinTransformer(nn.Module):
    # (Sao chép y hệt class MultiTaskSwinTransformer từ script training)
    def __init__(self, num_species, num_traits, num_seg_classes, model_name='swin_base_patch4_window7_224.ms_in22k'):
        super(MultiTaskSwinTransformer, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        backbone_out_channels = self.backbone.feature_info.channels()
        feature_dim = backbone_out_channels[-1]
        self.species_head = nn.Linear(feature_dim, num_species)
        self.trait_head = nn.Linear(feature_dim, num_traits)
        fpn_channels = 256
        self.seg_fpn_head = FPNHead(in_channels_list=backbone_out_channels, out_channels=fpn_channels)
        self.seg_classifier = nn.Conv2d(len(backbone_out_channels) * fpn_channels, num_seg_classes, 1)
        self.log_var_species = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_var_trait = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_var_seg = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_var_consistency = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        features = self.backbone(x)
        features_2d = [feat.permute(0, 3, 1, 2).contiguous() for feat in features]
        last_feature_map = features_2d[-1]
        pooled_features = torch.flatten(F.adaptive_avg_pool2d(last_feature_map, (1, 1)), 1)
        species_logits = self.species_head(pooled_features)
        trait_logits = self.trait_head(pooled_features)
        fpn_features = self.seg_fpn_head(features_2d)
        fpn_features_upsampled = F.interpolate(fpn_features, scale_factor=4, mode='bilinear', align_corners=False)
        seg_output = self.seg_classifier(fpn_features_upsampled)
        seg_logits = F.interpolate(seg_output, size=x.shape[2:], mode='bilinear', align_corners=False)
        return {
            'species': species_logits,
            'traits': trait_logits,
            'segmentation': seg_logits,
            'log_vars': {
                'species': self.log_var_species,
                'trait': self.log_var_trait,
                'seg': self.log_var_seg,
                'consistency': self.log_var_consistency
            }
        }

# --- Các class helper ---
from multitask_dataset import FishVistaMultiTaskDataset # Cần file này

# ============================================================
# 2. HÀM ĐÁNH GIÁ (SAO CHÉP VÀ SỬA ĐỔI NHẸ)
# ============================================================
def evaluate(model, loader, split_name, species_id_to_group, args):
    model.eval()
    all_species_preds, all_species_labels = [], []
    all_trait_preds, all_trait_labels = {}, {}
    jaccard = JaccardIndex(task="multiclass", num_classes=args.num_seg_classes, ignore_index=0).to(args.device)
    results = {}

    print(f"--- Evaluating on '{split_name}' split ---")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Evaluating {split_name}")):
            images = batch['image'].to(args.device)
            
            # Chạy forward pass không cần autocast trong evaluation để đảm bảo chính xác
            outputs = model(images)

            # --- Thu thập metrics ---
            if batch['has_species_label'].any():
                species_labels = batch['species_label']; mask = (species_labels != -1)
                if mask.any(): all_species_preds.append(outputs['species'][mask].argmax(dim=1).cpu()); all_species_labels.append(species_labels[mask].cpu())
            
            if batch['has_trait_labels'].any():
                trait_labels = batch['trait_labels']; mask = (trait_labels != -1.0)
                if mask.any():
                    preds_sig = torch.sigmoid(outputs['traits'])
                    for i in range(args.num_traits):
                        mask_i = mask[:, i]
                        if mask_i.any():
                            if i not in all_trait_preds: all_trait_preds[i], all_trait_labels[i] = [], []
                            all_trait_preds[i].append(preds_sig[mask_i, i].cpu().numpy()); all_trait_labels[i].append(trait_labels[mask_i, i].cpu().numpy())
            
            if batch['has_mask'].any():
                seg_masks = batch['segmentation_mask'].to(args.device); mask = batch['has_mask']
                if mask.any(): seg_preds = outputs['segmentation'][mask].argmax(dim=1); jaccard.update(seg_preds, seg_masks[mask])
    
    # --- Tính toán metrics cuối cùng ---
    if all_species_labels:
        preds, labels = torch.cat(all_species_preds).numpy(), torch.cat(all_species_labels).numpy()
        results['species_accuracy_total'] = accuracy_score(labels, preds); results['species_f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
        group_correct, group_total = defaultdict(int), defaultdict(int)
        for i in range(len(labels)):
            group = species_id_to_group.get(str(labels[i]), "Unknown"); group_total[group] += 1
            if labels[i] == preds[i]: group_correct[group] += 1
        results['species_accuracy_per_group'] = {g: (group_correct[g] / t if t > 0 else 0) for g, t in sorted(group_total.items())}
    
    if all_trait_labels:
        aps = []; trait_names = ['adipose_fin', 'pelvic_fin', 'barbel', 'multiple_dorsal_fin']
        for i in range(args.num_traits):
            if i in all_trait_labels:
                try:
                    ap = average_precision_score(np.concatenate(all_trait_labels[i]), np.concatenate(all_trait_preds[i])); aps.append(ap)
                    results[f'trait_ap_{trait_names[i]}'] = ap
                except ValueError: results[f'trait_ap_{trait_names[i]}'] = 0.0
        if aps: results['trait_mAP'] = np.mean(aps)
    
    results['segmentation_mIoU'] = jaccard.compute().item()
    
    print(f"Results for {split_name} - Species Acc: {results.get('species_accuracy_total', 0):.4f}, Trait mAP: {results.get('trait_mAP', 0):.4f}, Seg mIoU: {results.get('segmentation_mIoU', 0):.4f}")
    return results

# ============================================================
# 3. HÀM MAIN
# ============================================================
def main(args):
    # --- Thiết lập ---
    if not args.checkpoint_path or not os.path.exists(args.checkpoint_path):
        print(f"Lỗi: File checkpoint không tồn tại tại '{args.checkpoint_path}'. Vui lòng cung cấp đường dẫn hợp lệ qua --checkpoint-path.")
        return
        
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Data & Metadata ---
    print("\n[PHASE 1/3] Loading Data & Metadata...");
    master_df = pd.read_csv(args.master_csv, low_memory=False)
    try:
        with open('species_to_id.json', 'r') as f: species_to_id = json.load(f)
        with open('species_to_group.json', 'r') as f: species_id_to_group = json.load(f)
    except FileNotFoundError as e:
        print(f"Lỗi: {e}. Vui lòng đảm bảo các file 'species_to_id.json' và 'species_to_group.json' tồn tại."); return
        
    num_species = len(species_to_id)
    print(f"Found {num_species} species.")
    
    # --- Khởi tạo Model ---
    print("\n[PHASE 2/3] Initializing and Loading Model...")
    model = MultiTaskSwinTransformer(num_species=num_species, num_traits=args.num_traits, num_seg_classes=args.num_seg_classes).to(args.device)
    print(f"Loading model weights from: {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device, weights_only=True))
    
    # --- Chạy Đánh giá ---
    print("\n[PHASE 3/3] Running Evaluation on Test Splits...")
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    full_dataset_val = FishVistaMultiTaskDataset(csv_file=args.master_csv, transform=val_transform, species_to_id=species_to_id)
    
    test_splits_indices = { 
        'test': master_df[master_df['split'] == 'test'].index.tolist(), 
        'test_ood': master_df[master_df['split'] == 'test_ood'].index.tolist(), 
        'test_manual': master_df[master_df['split'] == 'test_manual'].index.tolist() 
    }
    test_loaders = {name: DataLoader(Subset(full_dataset_val, indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) 
                    for name, indices in test_splits_indices.items() if indices}

    final_results = {}
    for split_name, loader in test_loaders.items():
        split_results = evaluate(model, loader, split_name, species_id_to_group, args)
        final_results[split_name] = split_results
    
    # --- Lưu và In kết quả ---
    with open(args.output_file, 'w') as f: 
        json.dump(final_results, f, indent=4)
    print(f"\nEvaluation complete. Results saved to '{args.output_file}'.")
    print("\n--- Final Results Summary ---")
    print(json.dumps(final_results, indent=4))

# --- `if __name__ == "__main__":` ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fish-Vista: Evaluate a Multi-Task Model')
    
    # Paths
    parser.add_argument('--checkpoint-path', type=str, required=True, help='Path to the model checkpoint (.pth file) to evaluate.')
    parser.add_argument('--output-file', type=str, default='evaluation_results.json', help='Name of the output JSON file for evaluation results.')
    parser.add_argument('--master-csv', type=str, default='master_dataset.csv')
    
    # Evaluation Config
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation.')
    
    # Model and Data Config
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--num-traits', type=int, default=4)
    parser.add_argument('--num-seg-classes', type=int, default=10)

    # System
    parser.add_argument('--num-workers', type=int, default=4)

    args = parser.parse_args()
    main(args)