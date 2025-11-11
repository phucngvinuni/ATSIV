import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import json
from sklearn.metrics import average_precision_score, accuracy_score
from torchmetrics import JaccardIndex
import torch.nn.functional as F

# Import từ các file khác của bạn
from multitask_dataset import get_datasets
from multitask_model import MultiTaskSwinTransformer

# --- Cấu hình ---
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
MASTER_CSV = "master_dataset.csv"
CLASS_WEIGHTS_PATH = "class_weights.pt"
IMG_SIZE = 224
BATCH_SIZE = 16 

# === CẬP NHẬT CẤU HÌNH LEARNING RATE ===
INITIAL_LR = 1e-4       # LR mục tiêu sau khi warm-up
WARMUP_EPOCHS = 2       # Số epoch để warm-up
EPOCHS = 20             # Tăng số epoch lên để có thời gian cho LR giảm
# =======================================

NUM_WORKERS = 4
CHECKPOINT_DIR = "./checkpoints_focal_scheduler" # Thư mục checkpoint mới

NUM_TRAITS = 4 
NUM_SEG_CLASSES = 10 

W_SPECIES = 1.0
W_TRAIT = 1.0
W_SEG = 2.0

# --- CÁC CLASS LOSS (FocalLoss, DiceLoss) GIỮ NGUYÊN ---
class FocalLoss(nn.Module):
    # ... (code giữ nguyên)
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.gamma, self.alpha, self.reduction, self.ignore_index = gamma, alpha, reduction, ignore_index
    def forward(self, inputs, targets):
        valid_mask = targets != self.ignore_index
        if not valid_mask.any(): return torch.tensor(0.0, device=inputs.device)
        inputs, targets = inputs[valid_mask], targets[valid_mask]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt)**self.gamma) * ce_loss
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device: self.alpha = self.alpha.to(focal_loss.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

class DiceLoss(nn.Module):
    # ... (code giữ nguyên)
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        if logits.shape[0] == 0: return torch.tensor(0.0, device=logits.device)
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = nn.functional.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        cardinality = torch.sum(probs + targets_one_hot, dim=(2, 3))
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score.mean()

# --- HÀM EVALUATE (Giữ nguyên) ---
def evaluate(model, loader, criteria, epoch):
    # ... (code giữ nguyên)
    model.eval()
    total_loss = 0.0
    all_species_preds, all_species_labels = [], []
    all_trait_preds, all_trait_labels = {}, {}
    jaccard = JaccardIndex(task="multiclass", num_classes=NUM_SEG_CLASSES, ignore_index=0).to(DEVICE)
    zero_loss = torch.tensor(0.0, device=DEVICE)
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Validating Epoch {epoch+1}"):
            images = batch['image'].to(DEVICE)
            species_labels, trait_labels, seg_masks = batch['species_label'].to(DEVICE), batch['trait_labels'].to(DEVICE), batch['segmentation_mask'].to(DEVICE)
            species_mask_flag, trait_mask_flag, seg_mask_flag = batch['has_species_label'], batch['has_trait_labels'], batch['has_mask']
            with torch.amp.autocast(device_type=DEVICE.split(':')[0]):
                outputs = model(images)
                loss_species = criteria['species'](outputs['species'], species_labels) if species_mask_flag.any() else zero_loss
                if trait_mask_flag.any():
                    trait_preds, valid_trait_labels, valid_trait_mask = outputs['traits'][trait_mask_flag], trait_labels[trait_mask_flag], (trait_labels[trait_mask_flag] != -1.0)
                    loss_trait = (criteria['trait'](trait_preds, valid_trait_labels) * valid_trait_mask).sum() / valid_trait_mask.sum().clamp(min=1e-6)
                else: loss_trait = zero_loss
                if seg_mask_flag.any():
                    loss_seg_ce, loss_seg_dice = criteria['seg_ce'](outputs['segmentation'][seg_mask_flag], seg_masks[seg_mask_flag]), criteria['seg_dice'](outputs['segmentation'][seg_mask_flag], seg_masks[seg_mask_flag])
                    loss_seg = loss_seg_ce + loss_seg_dice
                else: loss_seg = zero_loss
                loss = (W_SPECIES * loss_species) + (W_TRAIT * loss_trait) + (W_SEG * loss_seg)
                if not torch.isnan(loss): total_loss += loss.item()
                if species_mask_flag.any():
                    valid_labels, valid_preds = species_labels[species_mask_flag], outputs['species'][species_mask_flag]
                    all_species_preds.append(valid_preds.argmax(dim=1).cpu()); all_species_labels.append(valid_labels.cpu())
                if trait_mask_flag.any():
                    trait_preds_batch_sig, valid_trait_labels, valid_trait_mask = torch.sigmoid(outputs['traits'][trait_mask_flag]), trait_labels[trait_mask_flag], (trait_labels[trait_mask_flag] != -1.0)
                    for i in range(trait_preds_batch_sig.shape[1]):
                        if mask_i := valid_trait_mask[:, i].any():
                           if i not in all_trait_preds: all_trait_preds[i], all_trait_labels[i] = [], []
                           all_trait_preds[i].append(trait_preds_batch_sig[valid_trait_mask[:, i], i].cpu().numpy()); all_trait_labels[i].append(valid_trait_labels[valid_trait_mask[:, i], i].cpu().numpy())
                if seg_mask_flag.any(): jaccard.update(outputs['segmentation'][seg_mask_flag].argmax(dim=1), seg_masks[seg_mask_flag])
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    species_acc = accuracy_score(torch.cat(all_species_labels).numpy(), torch.cat(all_species_preds).numpy()) if all_species_labels else 0.0
    trait_map = np.mean([average_precision_score(np.concatenate(all_trait_labels[i]), np.concatenate(all_trait_preds[i])) for i in range(NUM_TRAITS) if i in all_trait_labels]) if all_trait_labels else 0.0
    mIoU = jaccard.compute().item()
    print(f"Validation Results - Loss: {avg_loss:.4f}, Species Acc: {species_acc:.4f}, Trait mAP: {trait_map:.4f}, Seg mIoU: {mIoU:.4f}")
    return avg_loss

def main():
    print("Loading data...")
    # ... (code load data giữ nguyên)
    data_transforms = {
        'train': transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        'val': transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    }
    train_subset, val_subset, _, species_to_id = get_datasets(MASTER_CSV)
    train_subset.dataset.transform, val_subset.dataset.transform = data_transforms['train'], data_transforms['val']
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    num_species = len(species_to_id); print(f"Found {num_species} species.")
    
    print("Initializing model...")
    model = MultiTaskSwinTransformer(num_species, NUM_TRAITS, NUM_SEG_CLASSES).to(DEVICE)
    
    try:
        class_weights = torch.load(CLASS_WEIGHTS_PATH)
        print(f"Đã load thành công trọng số class từ '{CLASS_WEIGHTS_PATH}'.")
    except FileNotFoundError:
        class_weights = None; print(f"Cảnh báo: Không tìm thấy '{CLASS_WEIGHTS_PATH}'.")

    criteria = {'species': FocalLoss(gamma=2.0, alpha=class_weights, ignore_index=-1), 'trait': nn.BCEWithLogitsLoss(reduction='none'), 'seg_ce': nn.CrossEntropyLoss(), 'seg_dice': DiceLoss()}
    
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR)
    scaler = torch.amp.GradScaler()
    
    # --- THÊM LR SCHEDULER ---
    # Scheduler chính là CosineAnnealingLR
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    # Scheduler phụ cho warm-up
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    # Kết hợp cả hai
    lr_scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[WARMUP_EPOCHS])
    # --- KẾT THÚC THÊM SCHEDULER ---

    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        zero_loss = torch.tensor(0.0, device=DEVICE)
        
        # In ra LR hiện tại ở đầu mỗi epoch
        print(f"Epoch {epoch+1}/{EPOCHS}, Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}"):
            # (Code bên trong vòng lặp training giữ nguyên, chỉ thêm Gradient Clipping)
            images, species_labels, trait_labels, seg_masks = batch['image'].to(DEVICE), batch['species_label'].to(DEVICE), batch['trait_labels'].to(DEVICE), batch['segmentation_mask'].to(DEVICE)
            species_mask_flag, trait_mask_flag, seg_mask_flag = batch['has_species_label'].to(DEVICE), batch['has_trait_labels'].to(DEVICE), batch['has_mask'].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.split(':')[0]):
                outputs = model(images)
                loss_species = criteria['species'](outputs['species'], species_labels) if species_mask_flag.any() else zero_loss
                if trait_mask_flag.any():
                    trait_preds, valid_trait_labels, valid_trait_mask = outputs['traits'][trait_mask_flag], trait_labels[trait_mask_flag], (trait_labels[trait_mask_flag] != -1.0)
                    loss_trait = (criteria['trait'](trait_preds, valid_trait_labels) * valid_trait_mask).sum() / valid_trait_mask.sum().clamp(min=1e-6)
                else: loss_trait = zero_loss
                if seg_mask_flag.any():
                    loss_seg_ce, loss_seg_dice = criteria['seg_ce'](outputs['segmentation'][seg_mask_flag], seg_masks[seg_mask_flag]), criteria['seg_dice'](outputs['segmentation'][seg_mask_flag], seg_masks[seg_mask_flag])
                    loss_seg = loss_seg_ce + loss_seg_dice
                else: loss_seg = zero_loss
                loss = (W_SPECIES * loss_species) + (W_TRAIT * loss_trait) + (W_SEG * loss_seg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if not torch.isnan(loss): total_train_loss += loss.item()
        
        # --- CẬP NHẬT LR SCHEDULER ---
        lr_scheduler.step()
        # --- KẾT THÚC CẬP NHẬT ---
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Train Loss: {avg_train_loss:.4f}")

        avg_val_loss = evaluate(model, val_loader, criteria, epoch)

        if avg_val_loss < best_val_loss and not np.isnan(avg_val_loss):
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model_focal_scheduler.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved to {checkpoint_path} with validation loss: {best_val_loss:.4f}")
    
    print("Training complete.")
    torch.save(model.state_dict(), "multitask_model_final_focal_scheduler.pth")
    print("Final model saved to multitask_model_final_focal_scheduler.pth")

if __name__ == "__main__":
    main()