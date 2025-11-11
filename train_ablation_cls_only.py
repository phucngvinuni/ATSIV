import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import accuracy_score

# Import từ các file của bạn
from multitask_dataset import get_datasets
from multitask_model import MultiTaskSwinTransformer

# --- Cấu hình ---
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
MASTER_CSV = "master_dataset.csv"
IMG_SIZE = 224
BATCH_SIZE = 16

# --- Cấu hình LR và Epochs (GIỮ NGUYÊN ĐỂ SO SÁNH CÔNG BẰNG) ---
INITIAL_LR = 1e-4
WARMUP_EPOCHS = 5
EPOCHS = 10
# ==========================================================

NUM_WORKERS = 8
# --- CẬP NHẬT Ở ĐÂY: Thư mục checkpoint mới cho Ablation Study ---
CHECKPOINT_DIR = "./checkpoints_cls_only" 

# Số lượng class
NUM_TRAITS = 4 
NUM_SEG_CLASSES = 10

# --- CẬP NHẬT Ở ĐÂY: "Tắt" các tác vụ phụ bằng cách đặt trọng số loss bằng 0 ---
W_SPECIES = 1.0
W_TRAIT = 0.0   # Tắt loss identification
W_SEG = 0.0     # Tắt loss segmentation
# =======================================================================

# --- HÀM EVALUATE (Đã được rút gọn) ---
def evaluate(model, loader, criteria, epoch):
    model.eval()
    total_loss = 0.0
    all_species_preds, all_species_labels = [], []
    zero_loss = torch.tensor(0.0, device=DEVICE)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Validating Epoch {epoch+1}"):
            images = batch['image'].to(DEVICE)
            species_labels = batch['species_label'].to(DEVICE)
            species_mask_flag = batch['has_species_label']

            with torch.amp.autocast(device_type=DEVICE.split(':')[0]):
                outputs = model(images)
                
                # Chỉ cần tính loss_species cho việc đánh giá
                loss_species = criteria['species'](outputs['species'][species_mask_flag], species_labels[species_mask_flag]) if species_mask_flag.any() else zero_loss
                loss = loss_species # Loss tổng thể bây giờ chỉ là loss species
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
                
                if species_mask_flag.any():
                    all_species_preds.append(outputs['species'][species_mask_flag].argmax(dim=1).cpu())
                    all_species_labels.append(species_labels[species_mask_flag].cpu())

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    species_acc = accuracy_score(torch.cat(all_species_labels).numpy(), torch.cat(all_species_preds).numpy()) if all_species_labels else 0.0
    
    print(f"Validation Results - Loss: {avg_loss:.4f}, Species Acc: {species_acc:.4f}")
    return avg_loss

def main():
    print("--- Ablation Study: Training Classification Only ---")
    print("Loading data...")
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
    
    # Dùng Cross Entropy Loss cho classification, các loss khác không quan trọng vì trọng số là 0
    criteria = {
        'species': nn.CrossEntropyLoss(ignore_index=-1),
        'trait': nn.BCEWithLogitsLoss(reduction='none'),
        'seg_ce': nn.CrossEntropyLoss(),
        'seg_dice': nn.Module() # Dùng module rỗng để tránh tính toán không cần thiết
    }

    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR)
    scaler = torch.amp.GradScaler()
    
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    lr_scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[WARMUP_EPOCHS])

    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        zero_loss = torch.tensor(0.0, device=DEVICE)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}"):
            images, species_labels = batch['image'].to(DEVICE), batch['species_label'].to(DEVICE)
            species_mask_flag = batch['has_species_label'].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.split(':')[0]):
                outputs = model(images)
                
                # Chỉ tính loss_species
                loss = criteria['species'](outputs['species'][species_mask_flag], species_labels[species_mask_flag]) if species_mask_flag.any() else zero_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if not torch.isnan(loss): total_train_loss += loss.item()
        
        lr_scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Train Loss: {avg_train_loss:.4f}")

        avg_val_loss = evaluate(model, val_loader, criteria, epoch)

        if avg_val_loss < best_val_loss and not np.isnan(avg_val_loss):
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model_cls_only.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved to {checkpoint_path} with validation loss: {best_val_loss:.4f}")
    
    print("Training complete.")
    torch.save(model.state_dict(), "final_model_cls_only.pth")
    print("Final model saved to final_model_cls_only.pth")

if __name__ == "__main__":
    main()