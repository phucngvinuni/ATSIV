# hierarchical_dataset.py

import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
from PIL import Image, UnidentifiedImageError
import numpy as np
import os
import torchvision
import json

class FishVistaHierarchicalDataset(Dataset):
    """
    Dataset cho Fish-Vista, hỗ trợ đa tác vụ và học thứ bậc (loài và họ).
    """
    def __init__(self, csv_file, transform=None, species_to_id=None, family_to_id=None):
        """
        Khởi tạo dataset.
        Args:
            csv_file (str): Đường dẫn tới file master_dataset.csv.
            transform (callable, optional): Transform áp dụng cho ảnh.
            species_to_id (dict, optional): Mapping từ tên loài sang ID.
            family_to_id (dict, optional): Mapping từ tên họ sang ID.
        """
        self.master_df = pd.read_csv(csv_file, low_memory=False)
        self.transform = transform
        
        # --- Xử lý Mapping cho Loài ---
        if species_to_id is None:
            all_species = sorted(self.master_df['standardized_species'].dropna().unique())
            self.species_to_id = {species: i for i, species in enumerate(all_species)}
        else:
            self.species_to_id = species_to_id
        self.num_species = len(self.species_to_id)
        
        # --- THÊM XỬ LÝ MAPPING CHO HỌ ---
        if family_to_id is None:
            all_families = sorted(self.master_df['family'].dropna().unique())
            self.family_to_id = {fam: i for i, fam in enumerate(all_families)}
        else:
            self.family_to_id = family_to_id
        self.num_families = len(self.family_to_id)
        # ------------------------------------

        self.trait_columns = ['adipose_fin', 'pelvic_fin', 'barbel', 'multiple_dorsal_fin']

    def __len__(self):
        return len(self.master_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.master_df.iloc[idx]
        image_path = row['image_path']

        # Xử lý các trường hợp ảnh bị lỗi hoặc đường dẫn không hợp lệ
        try:
            image = Image.open(image_path).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError, TypeError, AttributeError):
            # print(f"Warning: Cannot load image at index {idx}, path: {image_path}. Returning dummy data.")
            # Tạo dữ liệu giả để DataLoader không bị lỗi
            img_size = 224 # Mặc định
            if self.transform and hasattr(self.transform, 'transforms'):
                 for t in self.transform.transforms:
                    if isinstance(t, (torchvision.transforms.Resize, torchvision.transforms.CenterCrop)):
                        img_size = t.size[0] if isinstance(t.size, tuple) else t.size
                        break
            return self._get_dummy_item(img_size)

        # Áp dụng transform
        img_size_for_mask_tuple = image.size
        if self.transform:
            image_tensor = self.transform(image)
            img_size_for_mask_tuple = (image_tensor.shape[2], image_tensor.shape[1])
        else:
            image_tensor = torchvision.transforms.ToTensor()(image)

        # --- Lấy các nhãn ---
        # 1. Nhãn Loài
        species_name = row['standardized_species']
        species_label = self.species_to_id.get(species_name, -1)
        
        # 2. THÊM NHÃN HỌ
        family_name = row['family']
        family_label = self.family_to_id.get(family_name, -1)
        
        # 3. Nhãn Đặc điểm
        trait_labels = [row[col] for col in self.trait_columns]
        trait_labels = np.nan_to_num(trait_labels, nan=-1.0).astype(np.float32)
        
        # 4. Nhãn Phân đoạn
        mask_path = row['mask_path']
        has_mask = pd.notna(mask_path) and isinstance(mask_path, str)
        if has_mask:
            try:
                mask = Image.open(mask_path)
                mask = mask.resize(img_size_for_mask_tuple, Image.NEAREST)
                mask = torch.from_numpy(np.array(mask)).long()
            except (FileNotFoundError, UnidentifiedImageError):
                mask = torch.zeros(img_size_for_mask_tuple[::-1], dtype=torch.long)
                has_mask = False
        else:
            mask = torch.zeros(img_size_for_mask_tuple[::-1], dtype=torch.long)

        # --- Gộp thành dictionary trả về ---
        sample = {
            'image': image_tensor,
            'species_label': torch.tensor(species_label, dtype=torch.long),
            'family_label': torch.tensor(family_label, dtype=torch.long), # THÊM
            'trait_labels': torch.tensor(trait_labels, dtype=torch.float),
            'segmentation_mask': mask,
            'has_species_label': species_label != -1,
            'has_family_label': family_label != -1, # THÊM
            'has_trait_labels': not np.all(trait_labels == -1),
            'has_mask': has_mask,
            'filename': row['filename'] if isinstance(row['filename'], str) else "unknown"
        }

        return sample

    def _get_dummy_item(self, img_size):
        """Tạo một item giả để tránh lỗi khi ảnh không load được."""
        image = torch.zeros((3, img_size, img_size))
        mask = torch.zeros((img_size, img_size), dtype=torch.long)
        return {
            'image': image,
            'species_label': torch.tensor(-1, dtype=torch.long),
            'family_label': torch.tensor(-1, dtype=torch.long),
            'trait_labels': torch.full((len(self.trait_columns),), -1.0, dtype=torch.float),
            'segmentation_mask': mask,
            'has_species_label': False,
            'has_family_label': False,
            'has_trait_labels': False,
            'has_mask': False,
            'filename': "dummy_item"
        }


def get_hierarchical_datasets(master_csv_path):
    """
    Hàm helper để tạo các tập train, val, test từ file master CSV.
    """
    master_df = pd.read_csv(master_csv_path, low_memory=False)
    
    # Load mappings
    try:
        with open('species_to_id.json', 'r') as f:
            species_to_id = json.load(f)
        with open('family_to_id.json', 'r') as f:
            family_to_id = json.load(f)
    except FileNotFoundError:
        print("Cảnh báo: Không tìm thấy file mapping. Dataset sẽ tự tạo mapping.")
        species_to_id, family_to_id = None, None

    # Lấy indices cho từng split
    train_indices = master_df[master_df['split'] == 'train'].index.tolist()
    val_indices = master_df[master_df['split'] == 'val'].index.tolist()
    test_indices = master_df[master_df['split'] == 'test'].index.tolist()
    
    # Tạo dataset đầy đủ một lần
    full_dataset = FishVistaHierarchicalDataset(
        csv_file=master_csv_path,
        species_to_id=species_to_id,
        family_to_id=family_to_id
    )

    # Tạo các Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    return train_dataset, val_dataset, test_dataset, species_to_id, family_to_id