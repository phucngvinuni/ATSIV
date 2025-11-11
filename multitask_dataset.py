import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
from PIL import Image, UnidentifiedImageError
import numpy as np
import os
import torchvision # <<<<<<< THÊM DÒNG NÀY

class FishVistaMultiTaskDataset(Dataset):
    def __init__(self, csv_file, transform=None, species_to_id=None):
        self.master_df = pd.read_csv(csv_file, low_memory=False)
        self.transform = transform
        
        if species_to_id is None:
            all_species = sorted(self.master_df['standardized_species'].dropna().unique())
            self.species_to_id = {species: i for i, species in enumerate(all_species)}
        else:
            self.species_to_id = species_to_id
        
        self.num_species = len(self.species_to_id)
        
        self.trait_columns = ['adipose_fin', 'pelvic_fin', 'barbel', 'multiple_dorsal_fin']

    def __len__(self):
        return len(self.master_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.master_df.iloc[idx]

        image_path = row['image_path']

        if pd.isna(image_path) or not isinstance(image_path, str):
            img_size = 224
            if self.transform and hasattr(self.transform, 'transforms'):
                for t in self.transform.transforms:
                    # SỬA LỖI Ở ĐÂY: Dùng torchvision.transforms thay vì torch.vision.transforms
                    if isinstance(t, (torchvision.transforms.Resize, torchvision.transforms.CenterCrop)):
                        img_size = t.size[0] if isinstance(t.size, tuple) else t.size
                        break

            # print(f"Warning: Invalid or missing image_path at index {idx}. Path: {image_path}. Returning dummy data.")
            image = torch.zeros((3, img_size, img_size))
            mask = torch.zeros((img_size, img_size), dtype=torch.long)
            return {
                'image': image, 'species_label': torch.tensor(-1, dtype=torch.long),
                'trait_labels': torch.full((len(self.trait_columns),), -1.0, dtype=torch.float),
                'segmentation_mask': mask, 'has_species_label': False,
                'has_trait_labels': False, 'has_mask': False, 'filename': "invalid_path"
            }

        try:
            image = Image.open(image_path).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Warning: Cannot load image {image_path}. Error: {e}. Returning dummy data.")
            img_size = 224
            if self.transform and hasattr(self.transform, 'transforms'):
                 for t in self.transform.transforms:
                    if isinstance(t, (torchvision.transforms.Resize, torchvision.transforms.CenterCrop)):
                        img_size = t.size[0] if isinstance(t.size, tuple) else t.size
                        break
            image_tensor = torch.zeros((3, img_size, img_size))
            mask = torch.zeros((img_size, img_size), dtype=torch.long)
            return {
                'image': image_tensor, 'species_label': torch.tensor(-1, dtype=torch.long),
                'trait_labels': torch.full((len(self.trait_columns),), -1.0, dtype=torch.float),
                'segmentation_mask': mask, 'has_species_label': False,
                'has_trait_labels': False, 'has_mask': False, 'filename': "load_error"
            }
        
        img_size_for_mask_tuple = image.size # (width, height)
        if self.transform:
            image_tensor = self.transform(image)
            img_size_for_mask_tuple = (image_tensor.shape[2], image_tensor.shape[1]) # (width, height)
        else:
            # Nếu không có transform, tự chuyển sang tensor
            image_tensor = torchvision.transforms.ToTensor()(image)

        species_name = row['standardized_species']
        species_label = self.species_to_id.get(species_name, -1)
        
        trait_labels = [row[col] for col in self.trait_columns]
        trait_labels = np.nan_to_num(trait_labels, nan=-1.0).astype(np.float32)
        
        mask_path = row['mask_path']
        has_mask = pd.notna(mask_path) and isinstance(mask_path, str)
        if has_mask:
            try:
                mask = Image.open(mask_path)
                mask = mask.resize(img_size_for_mask_tuple, Image.NEAREST)
                mask = torch.from_numpy(np.array(mask)).long()
            except (FileNotFoundError, UnidentifiedImageError):
                mask = torch.zeros(img_size_for_mask_tuple[::-1], dtype=torch.long) # size is (h, w)
                has_mask = False
        else:
            mask = torch.zeros(img_size_for_mask_tuple[::-1], dtype=torch.long)

        sample = {
            'image': image_tensor,
            'species_label': torch.tensor(species_label, dtype=torch.long),
            'trait_labels': torch.tensor(trait_labels, dtype=torch.float),
            'segmentation_mask': mask,
            'has_species_label': species_label != -1,
            'has_trait_labels': not np.all(trait_labels == -1),
            'has_mask': has_mask,
            'filename': row['filename'] if isinstance(row['filename'], str) else "unknown"
        }

        return sample


def get_datasets(master_csv_path):
    master_df = pd.read_csv(master_csv_path, low_memory=False)
    
    train_indices = master_df[master_df['split'] == 'train'].index.tolist()
    val_indices = master_df[master_df['split'] == 'val'].index.tolist()
    test_indices = master_df[master_df['split'] == 'test'].index.tolist()
    
    full_dataset = FishVistaMultiTaskDataset(csv_file=master_csv_path)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    return train_dataset, val_dataset, test_dataset, full_dataset.species_to_id