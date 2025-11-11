import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import os
import timm
from sklearn.metrics import accuracy_score

# Import các hàm loss và dataset từ các file khác của bạn
try:
    from multitask_dataset import get_datasets
except ImportError:
    print("Lỗi: Không tìm thấy file multitask_dataset.py.")
    print("Vui lòng đảm bảo file này nằm cùng thư mục.")
    exit()

# --- Cấu hình cho Thử nghiệm 1 ---
DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"
MASTER_CSV = "master_dataset.csv"
IMG_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 1e-4 # Khôi phục LR gốc vì không dùng weight
EPOCHS = 50
NUM_WORKERS = 4
CHECKPOINT_DIR = "./checkpoints_cls_only_noweightsbase" # Thư mục checkpoint mới cho thử nghiệm

MODEL_NAME = 'maxvit_base_tf_224.in1k' 

def main():
    print(f"--- THỬ NGHIỆM 1: Huấn luyện Classification-Only KHÔNG CÓ Class Weights ---")
    print(f"Model: {MODEL_NAME}, LR: {LEARNING_RATE}")
    
    # --- 1. Dữ liệu ---
    print("Loading data...")
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }

    train_subset, val_subset, _, species_to_id = get_datasets(MASTER_CSV)
    
    train_subset.dataset.transform = data_transforms['train']
    val_full_dataset = train_subset.dataset.__class__(csv_file=MASTER_CSV, transform=data_transforms['val'], species_to_id=species_to_id)
    val_subset = Subset(val_full_dataset, val_subset.indices)

    def collate_fn(batch):
        valid_items = [item for item in batch if item['has_species_label']]
        if not valid_items:
            return torch.empty(0, 3, IMG_SIZE, IMG_SIZE), torch.empty(0, dtype=torch.long)
        images = torch.stack([item['image'] for item in valid_items])
        labels = torch.stack([item['species_label'] for item in valid_items])
        return images, labels

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    
    num_species = len(species_to_id)
    print(f"Found {num_species} species.")

    # --- 2. Mô hình, Loss, Optimizer ---
    print("Initializing model...")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_species).to(DEVICE)
    
    # --- TẮT HOÀN TOÀN CLASS WEIGHTS ---
    class_weights = None
    print("!!! DEBUG MODE: Training is running WITHOUT any class weights. !!!")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler()

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    best_val_acc = 0.0

    # --- 3. Vòng lặp Huấn luyện ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}"):
            if images.shape[0] == 0: continue
            
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=DEVICE.split(':')[0]):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected at epoch {epoch+1}. Skipping batch.")
                continue

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch {epoch+1} - Average Train Loss: {avg_train_loss:.4f}")

        # --- Đánh giá trên tập Validation ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                if images.shape[0] == 0: continue
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                with torch.amp.autocast(device_type=DEVICE.split(':')[0]):
                    outputs = model(images)
                all_preds.append(outputs.argmax(dim=1).cpu())
                all_labels.append(labels.cpu())
        
        if not all_labels:
            print("Validation set seems to be empty or contains no valid labels.")
            continue
            
        val_acc = accuracy_score(torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy())
        print(f"Epoch {epoch+1} - Validation Accuracy: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME.replace('.', '_')}_best.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved to {checkpoint_path} with accuracy: {best_val_acc:.4f}")

    print("--- Training for classification-only model (no weights) complete ---")

if __name__ == "__main__":
    main()