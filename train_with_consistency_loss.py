# train_with_consistency_loss.py (v10 - Final Stable Version: No Autocast, No Logit Clipping)

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

# --- Import các module của dự án ---
try:
    from multitask_dataset import FishVistaMultiTaskDataset
    from multitask_model import MultiTaskSwinTransformer
except ImportError as e:
    print(f"Lỗi import: {e}. Hãy đảm bảo script được chạy từ thư mục gốc.")
    exit()

# ============================================================
# 1. ĐỊNH NGHĨA CÁC LOSS FUNCTIONS VÀ LỚP MỚI
# ============================================================

class SafeCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    An toàn cho segmentation, trả về 0 nếu tất cả các pixel đều bị bỏ qua.
    """
    def forward(self, input, target):
        if (target != self.ignore_index).any():
            return super().forward(input, target)
        else:
            return torch.tensor(0.0, device=input.device, requires_grad=True)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        if logits.shape[0] == 0: return torch.tensor(0.0, device=logits.device)
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        cardinality = torch.sum(probs, dim=(2,3)) + torch.sum(targets_one_hot, dim=(2,3))
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score.mean()

class SpeciesTraitConsistencyLoss(nn.Module):
    def __init__(self, constraint_matrix):
        super().__init__()
        self.register_buffer('constraint_matrix', constraint_matrix)
    def forward(self, species_logits, trait_logits):
        pred_species_ids = torch.argmax(species_logits, dim=1)
        relevant_constraints = self.constraint_matrix[pred_species_ids]
        valid_mask = (relevant_constraints != -1.0)
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=species_logits.device)
        
        consistency_loss = F.binary_cross_entropy_with_logits(
            trait_logits[valid_mask],
            relevant_constraints[valid_mask]
        )
        return consistency_loss

# ============================================================
# 2. HÀM ĐÁNH GIÁ
# ============================================================
def evaluate(model, loader, split_name, species_id_to_group, num_species, criteria, args):
    model.eval()
    all_species_preds, all_species_labels = [], []
    all_trait_preds, all_trait_labels = {}, {}
    jaccard = JaccardIndex(task="multiclass", num_classes=args.num_seg_classes, ignore_index=0).to(args.device)
    results = {}
    total_val_loss, num_valid_batches = 0.0, 0
    zero_loss = torch.tensor(0.0, device=args.device)

    print(f"--- Evaluating on '{split_name}' split ---")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Evaluating {split_name}")):
            images = batch['image'].to(args.device)
            
            # LOẠI BỎ AUTOCAST
            outputs = model(images)

            # --- Tính loss ---
            loss_species = criteria['species'](outputs['species'], batch['species_label'].to(args.device)) if batch['has_species_label'].any() else zero_loss
            loss_trait = zero_loss
            if batch['has_trait_labels'].any():
                trait_labels, trait_mask = batch['trait_labels'].to(args.device), (batch['trait_labels'] != -1.0).to(args.device)
                if trait_mask.any():
                    loss_trait = (criteria['trait'](outputs['traits'], trait_labels) * trait_mask).sum() / trait_mask.sum().clamp(min=1e-6)
            loss_seg = zero_loss
            if batch['has_mask'].any():
                seg_masks = batch['segmentation_mask'].to(args.device)
                loss_seg = criteria['seg_ce'](outputs['segmentation'], seg_masks) + criteria['seg_dice'](outputs['segmentation'], seg_masks)
            loss_consistency = criteria['consistency'](outputs['species'], outputs['traits'])
            
            total_loss = (args.w_species * loss_species) + (args.w_trait * loss_trait) + \
                         (args.w_seg * loss_seg) + (args.w_consistency * loss_consistency)

            if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                total_val_loss += total_loss.item()
                num_valid_batches += 1
            else:
                print(f"\n[WARNING] Bỏ qua batch {batch_idx} trong {split_name} do loss là NaN/inf.")
                print(f"  - loss_species: {loss_species.item():.4f}, loss_trait: {loss_trait.item():.4f}, loss_seg: {loss_seg.item():.4f}, loss_consistency: {loss_consistency.item():.4f}")

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
                except ValueError: pass
        if aps: results['trait_mAP'] = np.mean(aps)
    results['segmentation_mIoU'] = jaccard.compute().item()
    avg_val_loss = total_val_loss / num_valid_batches if num_valid_batches > 0 else float('inf')
    
    print(f"Validation Results - Loss: {avg_val_loss:.4f}, Species Acc: {results.get('species_accuracy_total', 0):.4f}, Trait mAP: {results.get('trait_mAP', 0):.4f}, Seg mIoU: {results.get('segmentation_mIoU', 0):.4f}")
    return avg_val_loss, results

# ============================================================
# 3. HÀM MAIN
# ============================================================
def main(args):
    # --- Thiết lập ---
    if args.evaluate_only and not args.checkpoint_path: print("Lỗi: Chế độ --evaluate-only yêu cầu --checkpoint-path."); return
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
    if not args.evaluate_only and not os.path.exists(run_checkpoint_dir): os.makedirs(run_checkpoint_dir)

    # --- Load Data, Metadata, Constraint Matrix ---
    print("\n[PHASE 1/3] Loading Data & Metadata..."); master_df = pd.read_csv(args.master_csv, low_memory=False)
    try:
        with open('species_to_id.json', 'r') as f: species_to_id = json.load(f)
        with open('species_to_group.json', 'r') as f: species_id_to_group = json.load(f)
    except FileNotFoundError as e: print(f"Lỗi: {e}."); return
    num_species = len(species_to_id); print(f"Found {num_species} species.")
    print("Loading constraint matrix...")
    try: constraint_matrix = torch.load(args.constraint_matrix_path, map_location=args.device, weights_only=True)
    except FileNotFoundError: print(f"Lỗi: Không tìm thấy file '{args.constraint_matrix_path}'."); return
    
    val_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    full_dataset_val = FishVistaMultiTaskDataset(csv_file=args.master_csv, transform=val_transform, species_to_id=species_to_id)
    print("Initializing model...")
    model = MultiTaskSwinTransformer(num_species=num_species, num_traits=args.num_traits, num_seg_classes=args.num_seg_classes).to(args.device)
    
    # SỬ DỤNG nn.CrossEntropyLoss tiêu chuẩn cho species
    criteria = { 
        'species': nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1), 
        'trait': nn.BCEWithLogitsLoss(reduction='none'), 
        'seg_ce': SafeCrossEntropyLoss(ignore_index=0), 
        'seg_dice': DiceLoss(), 
        'consistency': SpeciesTraitConsistencyLoss(constraint_matrix) 
    }

    # --- Chế độ Chỉ Đánh giá ---
    if args.evaluate_only:
        print("\n[PHASE 2/3] Running in EVALUATION-ONLY mode...")
        print(f"Loading model from: {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device, weights_only=True))
    # --- Chế độ Huấn luyện ---
    else:
        print("\n[PHASE 2/3] Starting Training...")
        train_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.TrivialAugmentWide(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),])
        full_dataset_train = FishVistaMultiTaskDataset(csv_file=args.master_csv, transform=train_transform, species_to_id=species_to_id)
        train_indices = master_df[master_df['split'] == 'train'].index.tolist()
        val_indices = master_df[master_df['split'] == 'val'].index.tolist()
        train_loader = DataLoader(Subset(full_dataset_train, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(Subset(full_dataset_val, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)
        
        # LOẠI BỎ GradScaler
        
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
                
                # LOẠI BỎ `with autocast`
                outputs = model(batch['image'].to(args.device))
                loss_species = criteria['species'](outputs['species'], batch['species_label'].to(args.device)) if batch['has_species_label'].any() else zero_loss
                loss_trait = zero_loss
                if batch['has_trait_labels'].any():
                    trait_labels, trait_mask = batch['trait_labels'].to(args.device), (batch['trait_labels'] != -1.0).to(args.device)
                    if trait_mask.any():
                        loss_trait = (criteria['trait'](outputs['traits'], trait_labels) * trait_mask).sum() / trait_mask.sum().clamp(min=1e-6)
                loss_seg = zero_loss
                if batch['has_mask'].any():
                    seg_masks = batch['segmentation_mask'].to(args.device)
                    loss_seg = criteria['seg_ce'](outputs['segmentation'], seg_masks) + criteria['seg_dice'](outputs['segmentation'], seg_masks)
                loss_consistency = criteria['consistency'](outputs['species'], outputs['traits'])
                loss = (args.w_species * loss_species) + (args.w_trait * loss_trait) + (args.w_seg * loss_seg) + (args.w_consistency * loss_consistency)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    # THAY THẾ CÁC LỆNH CỦA SCALER
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Vẫn giữ Grad Clipping
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    num_train_batches += 1
                    
                    if batch_idx > 0 and batch_idx % 500 == 0:
                        print(f"\n  Batch {batch_idx}: Total Loss: {loss.item():.4f} "
                              f"[S: {(args.w_species*loss_species).item():.2f}, T: {(args.w_trait*loss_trait).item():.2f}, "
                              f"G: {(args.w_seg*loss_seg).item():.2f}, C: {(args.w_consistency*loss_consistency).item():.2f}]")
                else:
                    print(f"\n[WARNING] Bỏ qua backward cho batch {batch_idx} do loss là NaN/inf.")
            
            if epoch >= args.warmup_epochs - 1: scheduler.step()
            avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
            print(f"Epoch {epoch+1} - Average Train Loss: {avg_train_loss:.4f}")
            
            avg_val_loss, _ = evaluate(model, val_loader, "Validation", species_id_to_group, num_species, criteria, args)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss; epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ New best model saved (Val Loss: {best_val_loss:.4f})")
            else:
                epochs_no_improve += 1; print(f"⚠️ No improvement for {epochs_no_improve} epoch(s). Patience: {epochs_no_improve}/{args.patience}")
            if epochs_no_improve >= args.patience: print(f"⏹️ Early stopping triggered after {epoch+1} epochs."); break
        
        print("\nTraining complete. Loading best model for final evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=args.device, weights_only=True))

    # --- Đánh giá cuối cùng ---
    print("\n[PHASE 3/3] Running Final Evaluation on Test Splits...")
    test_splits_indices = { 'test': master_df[master_df['split'] == 'test'].index.tolist(), 'test_ood': master_df[master_df['split'] == 'test_ood'].index.tolist(), 'test_manual': master_df[master_df['split'] == 'test_manual'].index.tolist() }
    test_loaders = {name: DataLoader(Subset(full_dataset_val, indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) for name, indices in test_splits_indices.items() if indices}
    final_results = {}
    for split_name, loader in test_loaders.items():
        _, split_results = evaluate(model, loader, split_name, species_id_to_group, num_species, criteria, args)
        final_results[split_name] = split_results
    
    output_filename = args.output_file if args.evaluate_only else os.path.join(run_checkpoint_dir, "final_evaluation_results.json")
    with open(output_filename, 'w') as f: json.dump(final_results, f, indent=4)
    print(f"\nEvaluation complete. Results saved to '{output_filename}'."); print(json.dumps(final_results, indent=4))

# --- `if __name__ == "__main__":` ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fish-Vista Multi-Task Training with Consistency Loss')
    # Paths and Names
    parser.add_argument('--run-name', type=str, default='consistency_exp_1', help='Name for this experiment run.')
    parser.add_argument('--master-csv', type=str, default='master_dataset.csv')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_consistency')
    parser.add_argument('--constraint-matrix-path', type=str, default='species_trait_constraints.pt')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16, help='Reduce if OOM')
    parser.add_argument('--base-lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    
    # Loss Weights
    parser.add_argument('--w-species', type=float, default=1.0)
    parser.add_argument('--w-trait', type=float, default=1.0)
    parser.add_argument('--w-seg', type=float, default=1.0)
    parser.add_argument('--w-consistency', type=float, default=0.5, help='Weight for the new consistency loss.')

    # Model and Data Config
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--num-traits', type=int, default=4)
    parser.add_argument('--num-seg-classes', type=int, default=10)

    # System
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Modes
    parser.add_argument('--evaluate-only', action='store_true', help='Run in evaluation-only mode.')
    parser.add_argument('--checkpoint-path', type=str, default='', help='Path to model checkpoint for evaluation.')
    parser.add_argument('--output-file', type=str, default='evaluation_results.json', help='Output file for evaluation results.')

    args = parser.parse_args()
    main(args)