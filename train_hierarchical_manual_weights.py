# train_hierarchical_manual_weights.py

import torch
import torch.nn as nn
import torch.optim as optim
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

# --- Import các module của dự án ---
try:
    from hierarchical_dataset import FishVistaHierarchicalDataset
except ImportError:
    print("Lỗi: Không thể import 'FishVistaHierarchicalDataset'. Hãy đảm bảo file 'hierarchical_dataset.py' tồn tại và đúng vị trí.")
    exit()

# ============================================================
# 1. ĐỊNH NGHĨA MODEL & LOSS
# ============================================================

class FPNHead(nn.Module):
    """Feature Pyramid Network Head để kết hợp các feature map ở các mức khác nhau."""
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

class HierarchicalMultiTaskModel(nn.Module):
    def __init__(self, num_species, num_families, num_traits, num_seg_classes, model_name='swin_base_patch4_window7_224.ms_in22k'):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        backbone_channels = self.backbone.feature_info.channels()
        feature_dim = backbone_channels[-1]

        # Heads cho các tác vụ
        self.species_head = nn.Linear(feature_dim, num_species)
        self.family_head = nn.Linear(feature_dim, num_families)
        self.trait_head = nn.Linear(feature_dim, num_traits)

        # Head cho segmentation
        fpn_channels = 256
        self.seg_fpn_head = FPNHead(in_channels_list=backbone_channels, out_channels=fpn_channels)
        self.seg_classifier = nn.Conv2d(len(backbone_channels) * fpn_channels, num_seg_classes, 1)

        # BỎ các tham số log_var cho Uncertainty Weighting

    def forward(self, x):
        features = self.backbone(x)
        features_2d = [feat.permute(0, 3, 1, 2).contiguous() for feat in features]
        last_feature_map = features_2d[-1]
        pooled_features = torch.flatten(F.adaptive_avg_pool2d(last_feature_map, (1, 1)), 1)
        
        # Lấy logits từ các head
        species_logits = self.species_head(pooled_features)
        family_logits = self.family_head(pooled_features)
        trait_logits = self.trait_head(pooled_features)
        
        fpn_features = self.seg_fpn_head(features_2d)
        fpn_features_upsampled = F.interpolate(fpn_features, scale_factor=4, mode='bilinear', align_corners=False)
        seg_output = self.seg_classifier(fpn_features_upsampled)
        seg_logits = F.interpolate(seg_output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # BỎ trả về 'log_vars'
        return {
            'species': species_logits,
            'family': family_logits,
            'traits': trait_logits,
            'segmentation': seg_logits,
        }

class SafeCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input, target):
        return super().forward(input, target) if (target != self.ignore_index).any() else torch.tensor(0.0, device=input.device, requires_grad=True)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        if logits.shape[0] == 0: return torch.tensor(0.0, device=logits.device)
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        cardinality = torch.sum(probs, dim=(2,3)) + torch.sum(targets_one_hot, dim=(2,3))
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score.mean()

class TaxonTraitConsistencyLoss(nn.Module):
    def __init__(self, constraint_matrix):
        super().__init__()
        self.register_buffer('constraint_matrix', constraint_matrix)
    def forward(self, taxon_logits, trait_logits):
        with torch.no_grad():
            pred_taxon_ids = torch.argmax(taxon_logits, dim=1)
        relevant_constraints = self.constraint_matrix[pred_taxon_ids]
        valid_mask = (relevant_constraints != -1.0)
        
        if not valid_mask.any(): return torch.tensor(0.0, device=taxon_logits.device)
        
        return F.binary_cross_entropy_with_logits(trait_logits[valid_mask], relevant_constraints[valid_mask])

# ============================================================
# 2. HÀM ĐÁNH GIÁ
# ============================================================
def evaluate(model, loader, split_name, criteria, args):
    model.eval()
    all_species_preds, all_species_labels = [], []
    all_family_preds, all_family_labels = [], []
    all_trait_preds, all_trait_labels = defaultdict(list), defaultdict(list)
    jaccard = JaccardIndex(task="multiclass", num_classes=args.num_seg_classes, ignore_index=0).to(args.device)
    results = {}
    total_val_loss, num_valid_batches = 0.0, 0
    zero_loss = torch.tensor(0.0, device=args.device)

    print(f"--- Evaluating on '{split_name}' split ---")
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
            outputs = model(batch['image'].to(args.device))
            
            # --- Tính các loss thô ---
            loss_species = criteria['species'](outputs['species'], batch['species_label'].to(args.device))
            loss_family = criteria['family'](outputs['family'], batch['family_label'].to(args.device))
            
            loss_trait = zero_loss
            if batch['has_trait_labels'].any():
                trait_labels, mask = batch['trait_labels'].to(args.device), (batch['trait_labels'] != -1.0).to(args.device)
                if mask.any(): loss_trait = (criteria['trait'](outputs['traits'], trait_labels) * mask).sum() / mask.sum().clamp(min=1e-6)

            loss_seg = zero_loss
            if batch['has_mask'].any():
                seg_masks = batch['segmentation_mask'].to(args.device)
                loss_seg = criteria['seg_ce'](outputs['segmentation'], seg_masks) + criteria['seg_dice'](outputs['segmentation'], seg_masks)
            
            loss_s_consistency = criteria['s_consistency'](outputs['species'], outputs['traits'])
            loss_f_consistency = criteria['f_consistency'](outputs['family'], outputs['traits'])

            # --- TÍNH LOSS TỔNG HỢP VỚI TRỌNG SỐ CỐ ĐỊNH ---
            total_loss = ((args.w_species * loss_species) +
                          (args.w_family * loss_family) +
                          (args.w_trait * loss_trait) +
                          (args.w_seg * loss_seg) +
                          (args.w_s_consistency * loss_s_consistency) +
                          (args.w_f_consistency * loss_f_consistency))

            if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                total_val_loss += total_loss.item()
                num_valid_batches += 1

            # --- Thu thập metrics ---
            s_mask = batch['species_label'] != -1
            if s_mask.any():
                all_species_preds.append(outputs['species'][s_mask].argmax(dim=1).cpu())
                all_species_labels.append(batch['species_label'][s_mask].cpu())
            
            f_mask = batch['family_label'] != -1
            if f_mask.any():
                all_family_preds.append(outputs['family'][f_mask].argmax(dim=1).cpu())
                all_family_labels.append(batch['family_label'][f_mask].cpu())

            if batch['has_trait_labels'].any():
                t_mask = (batch['trait_labels'] != -1.0)
                if t_mask.any():
                    preds_sig = torch.sigmoid(outputs['traits']).cpu()
                    for i in range(args.num_traits):
                        mask_i = t_mask[:, i]
                        if mask_i.any():
                            all_trait_preds[i].append(preds_sig[mask_i, i].numpy())
                            all_trait_labels[i].append(batch['trait_labels'][mask_i, i].cpu().numpy())
            
            if batch['has_mask'].any():
                seg_mask = batch['has_mask']
                if seg_mask.any():
                    jaccard.update(outputs['segmentation'][seg_mask].argmax(dim=1), batch['segmentation_mask'].to(args.device)[seg_mask])
    
    # --- Tính toán metrics cuối cùng ---
    if all_species_labels:
        preds, labels = torch.cat(all_species_preds).numpy(), torch.cat(all_species_labels).numpy()
        results['species_accuracy'] = accuracy_score(labels, preds)
        results['species_f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
    
    if all_family_labels:
        preds, labels = torch.cat(all_family_preds).numpy(), torch.cat(all_family_labels).numpy()
        results['family_accuracy'] = accuracy_score(labels, preds)
        results['family_f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
    
    if all_trait_labels:
        aps = []
        for i in range(args.num_traits):
            if all_trait_labels[i]:
                try: aps.append(average_precision_score(np.concatenate(all_trait_labels[i]), np.concatenate(all_trait_preds[i])))
                except ValueError: pass
        if aps: results['trait_mAP'] = np.mean(aps)
        
    results['segmentation_mIoU'] = jaccard.compute().item()
    avg_val_loss = total_val_loss / num_valid_batches if num_valid_batches > 0 else float('inf')
    
    print(f"Eval Results ({split_name}) - Loss: {avg_val_loss:.4f}, "
          f"Species Acc: {results.get('species_accuracy', 0):.4f}, "
          f"Family Acc: {results.get('family_accuracy', 0):.4f}, "
          f"Trait mAP: {results.get('trait_mAP', 0):.4f}, "
          f"Seg mIoU: {results.get('segmentation_mIoU', 0):.4f}")
    return avg_val_loss, results

# ============================================================
# 3. HÀM MAIN
# ============================================================
def main(args):
    args.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    run_checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
    if not args.evaluate_only and not os.path.exists(run_checkpoint_dir): os.makedirs(run_checkpoint_dir)

    print("\n[PHASE 1/3] Loading Data & Metadata..."); master_df = pd.read_csv(args.master_csv, low_memory=False)
    try:
        with open('species_to_id.json', 'r') as f: species_to_id = json.load(f)
        with open('family_to_id.json', 'r') as f: family_to_id = json.load(f)
        num_species, num_families = len(species_to_id), len(family_to_id)
        print(f"Found {num_species} species and {num_families} families.")
        s_constraint = torch.load(args.s_constraint_path, map_location=args.device)
        f_constraint = torch.load(args.f_constraint_path, map_location=args.device)
        print("Đã tải thành công cả 2 ma trận ràng buộc.")
    except FileNotFoundError as e: print(f"Lỗi: {e}. Hãy chạy các script chuẩn bị dữ liệu trước."); return
    
    train_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.TrivialAugmentWide(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), transforms.RandomErasing(p=0.25)])
    val_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dataset_args = {'csv_file': args.master_csv, 'species_to_id': species_to_id, 'family_to_id': family_to_id}
    full_dataset_train = FishVistaHierarchicalDataset(transform=train_transform, **dataset_args)
    full_dataset_val = FishVistaHierarchicalDataset(transform=val_transform, **dataset_args)
    
    train_indices = master_df[master_df['split'] == 'train'].index.tolist()
    val_indices = master_df[master_df['split'] == 'val'].index.tolist()
    train_loader = DataLoader(Subset(full_dataset_train, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(Subset(full_dataset_val, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("Initializing model...")
    model = HierarchicalMultiTaskModel(num_species, num_families, args.num_traits, args.num_seg_classes).to(args.device)
    
    criteria = {
        'species': nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1),
        'family': nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1),
        'trait': nn.BCEWithLogitsLoss(reduction='none'),
        'seg_ce': SafeCrossEntropyLoss(ignore_index=0),
        'seg_dice': DiceLoss(),
        's_consistency': TaxonTraitConsistencyLoss(s_constraint),
        'f_consistency': TaxonTraitConsistencyLoss(f_constraint)
    }
    
    if not args.evaluate_only:
        print("\n[PHASE 2/3] Starting Training...")
        optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)
        
        best_val_loss = float('inf'); epochs_no_improve = 0
        best_model_path = os.path.join(run_checkpoint_dir, "best_model.pth")
        
        for epoch in range(args.epochs):
            model.train(); total_train_loss = 0.0; num_train_batches = 0; zero_loss = torch.tensor(0.0, device=args.device)
            if epoch < args.warmup_epochs:
                lr_scale = (epoch + 1) / args.warmup_epochs
                for pg in optimizer.param_groups: pg['lr'] = args.base_lr * lr_scale
            print(f"\nEpoch {epoch+1}/{args.epochs} - LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch['image'].to(args.device))
                
                loss_species = criteria['species'](outputs['species'], batch['species_label'].to(args.device))
                loss_family = criteria['family'](outputs['family'], batch['family_label'].to(args.device))
                
                loss_trait = zero_loss
                if batch['has_trait_labels'].any():
                    trait_labels, mask = batch['trait_labels'].to(args.device), (batch['trait_labels'] != -1.0).to(args.device)
                    if mask.any(): loss_trait = (criteria['trait'](outputs['traits'], trait_labels) * mask).sum() / mask.sum().clamp(min=1e-6)

                loss_seg = zero_loss
                if batch['has_mask'].any():
                    seg_masks = batch['segmentation_mask'].to(args.device)
                    loss_seg = criteria['seg_ce'](outputs['segmentation'], seg_masks) + criteria['seg_dice'](outputs['segmentation'], seg_masks)

                loss_s_consistency = criteria['s_consistency'](outputs['species'], outputs['traits'])
                loss_f_consistency = criteria['f_consistency'](outputs['family'], outputs['traits'])

                # Áp dụng trọng số cố định
                loss = ((args.w_species * loss_species) +
                        (args.w_family * loss_family) +
                        (args.w_trait * loss_trait) +
                        (args.w_seg * loss_seg) +
                        (args.w_s_consistency * loss_s_consistency) +
                        (args.w_f_consistency * loss_f_consistency))

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    num_train_batches += 1
                    
                    if batch_idx > 0 and batch_idx % 500 == 0:
                        print(f"\n  Batch {batch_idx}: Total Loss: {loss.item():.4f}")
                        print(f"    Weighted Losses -> S:{(args.w_species*loss_species).item():.2f}, F:{(args.w_family*loss_family).item():.2f}, T:{(args.w_trait*loss_trait).item():.2f}, G:{(args.w_seg*loss_seg).item():.2f}, SC:{(args.w_s_consistency*loss_s_consistency).item():.2f}, FC:{(args.w_f_consistency*loss_f_consistency).item():.2f}")
            
            if epoch >= args.warmup_epochs - 1: scheduler.step()
            avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
            print(f"Epoch {epoch+1} - Average Train Loss: {avg_train_loss:.4f}")
            
            avg_val_loss, _ = evaluate(model, val_loader, "Validation", criteria, args)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss; epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path); print(f"✅ New best model saved (Val Loss: {best_val_loss:.4f})")
            else:
                epochs_no_improve += 1; print(f"⚠️ No improvement for {epochs_no_improve} epoch(s). Patience: {epochs_no_improve}/{args.patience}")
            if epochs_no_improve >= args.patience: print(f"⏹️ Early stopping triggered after {epoch+1} epochs."); break
        
        print("\nTraining complete. Loading best model for final evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=args.device))
    else:
        print(f"\n[PHASE 2/3] Loading model from: {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device))

    print("\n[PHASE 3/3] Running Final Evaluation on Test Splits...")
    test_splits = ['test', 'test_ood', 'test_manual']
    final_results = {}
    for split_name in test_splits:
        indices = master_df[master_df['split'] == split_name].index.tolist()
        if not indices: continue
        loader = DataLoader(Subset(full_dataset_val, indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        _, split_results = evaluate(model, loader, split_name, criteria, args)
        final_results[split_name] = split_results
    
    output_filename = args.output_file if args.evaluate_only else os.path.join(run_checkpoint_dir, "final_evaluation_results.json")
    with open(output_filename, 'w') as f: json.dump(final_results, f, indent=4)
    print(f"\nEvaluation complete. Results saved to '{output_filename}'."); print(json.dumps(final_results, indent=4))

# --- `if __name__ == "__main__":` ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fish-Vista: Hierarchical Training with Manual Loss Weights')
    parser.add_argument('--run-name', type=str, default='hierarchical_manual_weights_exp1')
    parser.add_argument('--master-csv', type=str, default='master_dataset.csv')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_hierarchical')
    parser.add_argument('--s-constraint-path', type=str, default='species_trait_constraints.pt')
    parser.add_argument('--f-constraint-path', type=str, default='family_trait_constraints.pt')
    
    # THÊM CÁC THAM SỐ TRỌNG SỐ CỐ ĐỊNH
    parser.add_argument('--w-species', type=float, default=1.0)
    parser.add_argument('--w-family', type=float, default=1.0)
    parser.add_argument('--w-trait', type=float, default=1.0)
    parser.add_argument('--w-seg', type=float, default=1.0)
    parser.add_argument('--w-s-consistency', type=float, default=0.5, help='Weight for species-trait consistency.')
    parser.add_argument('--w-f-consistency', type=float, default=0.5, help='Weight for family-trait consistency.')

    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--base-lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--num-traits', type=int, default=4)
    parser.add_argument('--num-seg-classes', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--checkpoint-path', type=str, default='')
    parser.add_argument('--output-file', type=str, default='evaluation_results.json')

    args = parser.parse_args()
    main(args)