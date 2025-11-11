import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score

from multitask_dataset import get_datasets
from multitask_model import MultiTaskSwinTransformer

# ============================================================
# CẤU HÌNH
# ============================================================
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
MASTER_CSV = "master_dataset.csv"
IMG_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 50
NUM_WORKERS = 4
CHECKPOINT_DIR = "./checkpoints"

NUM_TRAITS = 4
NUM_SEG_CLASSES = 10  # 9 traits + 1 background

# Trọng số loss
W_SPECIES = 1.0
W_TRAIT = 1.0
W_SEG = 1.0

# Early Stopping config
PATIENCE = 7  # ⬅️ nếu val_loss không giảm sau 7 epoch thì dừng


# ============================================================
# LOSS FUNCTION
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        import torch.nn.functional as F
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device)

        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        if logits.shape[0] == 0:
            return torch.tensor(0.0, device=logits.device)
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = nn.functional.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()

        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        cardinality = torch.sum(probs + targets_one_hot, dim=(2, 3))

        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score.mean()


# ============================================================
# HÀM ĐÁNH GIÁ
# ============================================================
def evaluate(model, loader, criteria, epoch):
    model.eval()
    total_loss = 0.0
    all_species_preds, all_species_labels = [], []
    all_trait_preds, all_trait_labels = [], []
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

                loss_species = criteria['species'](outputs['species'][species_mask_flag], species_labels[species_mask_flag]) if species_mask_flag.any() else zero_loss

                if trait_mask_flag.any():
                    trait_preds_batch = outputs['traits'][trait_mask_flag]
                    valid_trait_labels = trait_labels[trait_mask_flag]
                    valid_trait_mask = (valid_trait_labels != -1.0)
                    loss_trait_tensor = criteria['trait'](trait_preds_batch, valid_trait_labels)
                    loss_trait = (loss_trait_tensor * valid_trait_mask).sum() / valid_trait_mask.sum().clamp(min=1e-6)
                else:
                    loss_trait = zero_loss

                if seg_mask_flag.any():
                    loss_seg_ce = criteria['seg_ce'](outputs['segmentation'][seg_mask_flag], seg_masks[seg_mask_flag])
                    loss_seg_dice = criteria['seg_dice'](outputs['segmentation'][seg_mask_flag], seg_masks[seg_mask_flag])
                    loss_seg = loss_seg_ce + loss_seg_dice
                else:
                    loss_seg = zero_loss

                loss = (W_SPECIES * loss_species) + (W_TRAIT * loss_trait) + (W_SEG * loss_seg)
                total_loss += loss.item()

                if species_mask_flag.any():
                    all_species_preds.append(outputs['species'][species_mask_flag].argmax(dim=1).cpu())
                    all_species_labels.append(species_labels[species_mask_flag].cpu())

                if trait_mask_flag.any():
                    trait_preds_batch_sig = torch.sigmoid(outputs['traits'][trait_mask_flag])
                    valid_trait_labels = trait_labels[trait_mask_flag]
                    valid_trait_mask = (valid_trait_labels != -1.0)
                    for i in range(trait_preds_batch_sig.shape[1]):
                        mask_i = valid_trait_mask[:, i]
                        if mask_i.any():
                            all_trait_preds.append(trait_preds_batch_sig[mask_i, i].cpu().numpy())
                            all_trait_labels.append(valid_trait_labels[mask_i, i].cpu().numpy())

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    species_acc = accuracy_score(torch.cat(all_species_labels).numpy(), torch.cat(all_species_preds).numpy()) if all_species_labels else 0.0
    trait_map = 0.0
    if all_trait_labels:
        try:
            trait_map = average_precision_score(np.concatenate(all_trait_labels), np.concatenate(all_trait_preds))
        except ValueError:
            trait_map = 0.0

    print(f"Validation - Loss: {avg_loss:.4f}, Species Acc: {species_acc:.4f}, Trait mAP: {trait_map:.4f}")
    return avg_loss


# ============================================================
# TRAINING
# ============================================================
def main():
    print("Loading data...")
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # --- THÊM CÁC AUGMENTATION MẠNH HƠN ---
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Dịch ảnh ngẫu nhiên
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)), # Xóa một vùng ngẫu nhiên
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
    model = MultiTaskSwinTransformer(
        num_species=num_species,
        num_traits=NUM_TRAITS,
        num_seg_classes=NUM_SEG_CLASSES
    ).to(DEVICE)

    criteria = {
        'species': nn.CrossEntropyLoss(ignore_index=-1),
        'trait': nn.BCEWithLogitsLoss(reduction='none'),
        'seg_ce': nn.CrossEntropyLoss(),
        'seg_dice': DiceLoss()
    }

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler()

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        zero_loss = torch.tensor(0.0, device=DEVICE)

        print(f"\nEpoch {epoch+1}/{EPOCHS} - Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}"):
            images = batch['image'].to(DEVICE)
            species_labels = batch['species_label'].to(DEVICE)
            trait_labels = batch['trait_labels'].to(DEVICE)
            seg_masks = batch['segmentation_mask'].to(DEVICE)

            species_mask_flag = batch['has_species_label'].to(DEVICE)
            trait_mask_flag = batch['has_trait_labels'].to(DEVICE)
            seg_mask_flag = batch['has_mask'].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=DEVICE.split(':')[0]):
                outputs = model(images)

                loss_species = criteria['species'](outputs['species'][species_mask_flag], species_labels[species_mask_flag]) if species_mask_flag.any() else zero_loss

                if trait_mask_flag.any():
                    trait_preds = outputs['traits'][trait_mask_flag]
                    valid_trait_labels = trait_labels[trait_mask_flag]
                    valid_trait_mask = (valid_trait_labels != -1.0)
                    loss_trait_tensor = criteria['trait'](trait_preds, valid_trait_labels)
                    loss_trait = (loss_trait_tensor * valid_trait_mask).sum() / valid_trait_mask.sum().clamp(min=1e-6)
                else:
                    loss_trait = zero_loss

                if seg_mask_flag.any():
                    loss_seg_ce = criteria['seg_ce'](outputs['segmentation'][seg_mask_flag], seg_masks[seg_mask_flag])
                    loss_seg_dice = criteria['seg_dice'](outputs['segmentation'][seg_mask_flag], seg_masks[seg_mask_flag])
                    loss_seg = loss_seg_ce + loss_seg_dice
                else:
                    loss_seg = zero_loss

                loss = (W_SPECIES * loss_species) + (W_TRAIT * loss_trait) + (W_SEG * loss_seg)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Train Loss: {avg_train_loss:.4f}")

        avg_val_loss = evaluate(model, val_loader, criteria, epoch)
        scheduler.step()

        # === Early Stopping logic ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ New best model saved ({best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= PATIENCE:
            print(f"⏹️ Early stopping triggered after {epoch+1} epochs (best val loss: {best_val_loss:.4f})")
            break

    print("\nTraining complete.")
    torch.save(model.state_dict(), "multitask_model_final.pth")
    print("💾 Final model saved to multitask_model_final.pth")


if __name__ == "__main__":
    main()
