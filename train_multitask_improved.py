import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, accuracy_score
from torchmetrics import JaccardIndex
import torch.nn.functional as F

from multitask_dataset import get_datasets
from multitask_model import MultiTaskSwinTransformer

# ============================================================
# CẤU HÌNH
# ============================================================
DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"
MASTER_CSV = "master_dataset.csv"
IMG_SIZE = 224
BATCH_SIZE = 16

EPOCHS = 50
BASE_LEARNING_RATE = 1e-4
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 1e-2
PATIENCE = 10

NUM_WORKERS = 4
CHECKPOINT_DIR = "./checkpoints_improved"
NUM_TRAITS = 4
NUM_SEG_CLASSES = 10

W_SPECIES = 1.0
W_TRAIT = 1.0
W_SEG = 1.0

# ============================================================
# LOSS FUNCTIONS
# ============================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        if logits.shape[0] == 0: return torch.tensor(0.0, device=logits.device)
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        cardinality = torch.sum(probs + targets_one_hot, dim=(2, 3))
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score.mean()

# ============================================================
# HÀM ĐÁNH GIÁ ĐÃ SỬA LỖI
# ============================================================
def evaluate(model, loader, criteria, epoch):
    model.eval()
    total_loss = 0.0
    all_species_preds, all_species_labels = [], []
    all_trait_preds, all_trait_labels = {}, {} # Khởi tạo là dictionary
    jaccard = JaccardIndex(task="multiclass", num_classes=NUM_SEG_CLASSES, ignore_index=0).to(DEVICE)
    zero_loss = torch.tensor(0.0, device=DEVICE)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Validating Epoch {epoch+1}"):
            images = batch['image'].to(DEVICE)
            species_labels = batch['species_label'].to(DEVICE)
            trait_labels = batch['trait_labels'].to(DEVICE)
            seg_masks = batch['segmentation_mask'].to(DEVICE)
            species_mask_flag = batch['has_species_label']
            trait_mask_flag = batch['has_trait_labels']
            seg_mask_flag = batch['has_mask']

            with torch.amp.autocast(device_type=DEVICE.split(':')[0]):
                outputs = model(images)
                # (Tính loss giữ nguyên)
                loss_species = criteria['species'](outputs['species'][species_mask_flag], species_labels[species_mask_flag]) if species_mask_flag.any() else zero_loss
                if trait_mask_flag.any():
                    trait_preds, valid_labels, valid_mask = outputs['traits'][trait_mask_flag], trait_labels[trait_mask_flag], (trait_labels[trait_mask_flag] != -1.0)
                    loss_tensor = criteria['trait'](trait_preds, valid_labels)
                    loss_trait = (loss_tensor * valid_mask).sum() / valid_mask.sum().clamp(min=1e-6)
                else: loss_trait = zero_loss
                if seg_mask_flag.any():
                    loss_ce = criteria['seg_ce'](outputs['segmentation'][seg_mask_flag], seg_masks[seg_mask_flag])
                    loss_dice = criteria['seg_dice'](outputs['segmentation'][seg_mask_flag], seg_masks[seg_mask_flag])
                    loss_seg = loss_ce + loss_dice
                else: loss_seg = zero_loss
                loss = (W_SPECIES * loss_species) + (W_TRAIT * loss_trait) + (W_SEG * loss_seg)
                total_loss += loss.item()

            # (Thu thập dữ liệu cho metrics)
            if species_mask_flag.any():
                all_species_preds.append(outputs['species'][species_mask_flag].argmax(dim=1).cpu())
                all_species_labels.append(species_labels[species_mask_flag].cpu())

            if trait_mask_flag.any():
                preds_sig, valid_labels, valid_mask = torch.sigmoid(outputs['traits'][trait_mask_flag]), trait_labels[trait_mask_flag], (trait_labels[trait_mask_flag] != -1.0)
                for i in range(preds_sig.shape[1]):
                    mask_i = valid_mask[:, i]
                    if mask_i.any():
                        if i not in all_trait_preds:
                            all_trait_preds[i] = []
                            all_trait_labels[i] = []
                        all_trait_preds[i].append(preds_sig[mask_i, i].cpu().numpy())
                        all_trait_labels[i].append(valid_labels[mask_i, i].cpu().numpy())
            
            if seg_mask_flag.any():
                jaccard.update(outputs['segmentation'][seg_mask_flag].argmax(dim=1), seg_masks[seg_mask_flag])

    # (Tính toán metrics)
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    species_acc = accuracy_score(torch.cat(all_species_labels).numpy(), torch.cat(all_species_preds).numpy()) if all_species_labels else 0.0
    trait_map = 0.0
    if all_trait_labels:
        aps = []
        for i in range(NUM_TRAITS):
            if i in all_trait_labels:
                try: aps.append(average_precision_score(np.concatenate(all_trait_labels[i]), np.concatenate(all_trait_preds[i])))
                except ValueError: aps.append(0.0)
        if aps: trait_map = np.mean(aps)
    mIoU = jaccard.compute().item()
    print(f"Validation - Loss: {avg_loss:.4f}, Species Acc: {species_acc:.4f}, Trait mAP: {trait_map:.4f}, Seg mIoU: {mIoU:.4f}")
    return avg_loss

# ============================================================
# TRAINING (Hàm main giữ nguyên)
# ============================================================
def main():
    print("Loading data...")
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }

    train_subset, val_subset, _, species_to_id = get_datasets(MASTER_CSV)
    train_subset.dataset.transform = data_transforms['train']
    val_subset.dataset.transform = data_transforms['val']

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    num_species = len(species_to_id)
    print(f"Found {num_species} species.")

    print("Initializing model...")
    model = MultiTaskSwinTransformer(num_species=num_species, num_traits=NUM_TRAITS, num_seg_classes=NUM_SEG_CLASSES).to(DEVICE)

    criteria = {
        'species': nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1),
        'trait': nn.BCEWithLogitsLoss(reduction='none'),
        'seg_ce': nn.CrossEntropyLoss(),
        'seg_dice': DiceLoss()
    }

    optimizer = optim.AdamW(model.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler()

    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("Starting training with improved settings...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        zero_loss = torch.tensor(0.0, device=DEVICE)

        if epoch < WARMUP_EPOCHS:
            lr_scale = (epoch + 1) / WARMUP_EPOCHS
            for param_group in optimizer.param_groups: param_group['lr'] = BASE_LEARNING_RATE * lr_scale
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} - Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images, species_labels, trait_labels, seg_masks = batch['image'].to(DEVICE), batch['species_label'].to(DEVICE), batch['trait_labels'].to(DEVICE), batch['segmentation_mask'].to(DEVICE)
            species_mask_flag, trait_mask_flag, seg_mask_flag = batch['has_species_label'].to(DEVICE), batch['has_trait_labels'].to(DEVICE), batch['has_mask'].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.split(':')[0]):
                outputs = model(images)
                loss_species = criteria['species'](outputs['species'][species_mask_flag], species_labels[species_mask_flag]) if species_mask_flag.any() else zero_loss
                if trait_mask_flag.any():
                    trait_preds, valid_labels, valid_mask = outputs['traits'][trait_mask_flag], trait_labels[trait_mask_flag], (trait_labels[trait_mask_flag] != -1.0)
                    loss_tensor = criteria['trait'](trait_preds, valid_labels)
                    loss_trait = (loss_tensor * valid_mask).sum() / valid_mask.sum().clamp(min=1e-6)
                else: loss_trait = zero_loss
                if seg_mask_flag.any():
                    loss_ce = criteria['seg_ce'](outputs['segmentation'][seg_mask_flag], seg_masks[seg_mask_flag])
                    loss_dice = criteria['seg_dice'](outputs['segmentation'][seg_mask_flag], seg_masks[seg_mask_flag])
                    loss_seg = loss_ce + loss_dice
                else: loss_seg = zero_loss
                loss = (W_SPECIES * loss_species) + (W_TRAIT * loss_trait) + (W_SEG * loss_seg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Train Loss: {avg_train_loss:.4f}")

        avg_val_loss = evaluate(model, val_loader, criteria, epoch)
        
        if epoch >= WARMUP_EPOCHS - 1:
            scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model_improved.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ New best model saved ({best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s). Patience: {epochs_no_improve}/{PATIENCE}")

        if epochs_no_improve >= PATIENCE:
            print(f"⏹️ Early stopping triggered after {epoch+1} epochs (best val loss: {best_val_loss:.4f})")
            break

    print("\nTraining complete.")
    torch.save(model.state_dict(), "multitask_model_final_improved.pth")
    print("💾 Final model saved to multitask_model_final_improved.pth")

if __name__ == "__main__":
    main()