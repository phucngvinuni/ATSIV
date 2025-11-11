import torch
import timm
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import os

from multitask_model import MultiTaskSwinTransformer 
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- CẤU HÌNH ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "gradcam_visualizations"
MASTER_CSV = "master_dataset.csv"
IMG_SIZE = 224
NUM_IMAGES_TO_VISUALIZE = 50

MT_MODEL_PATH = "./checkpoints_improved/best_model_improved.pth"
MT_MODEL_ARCH = MultiTaskSwinTransformer

CLS_MODEL_PATH = "./checkpoints_cls_only_noweightsbase/maxvit_base_tf_224_in1k_best.pth"
CLS_MODEL_NAME = 'maxvit_base_tf_224.in1k'

# --- Reshape transform cho Swin-B ---
def reshape_transform_swin(tensor):
    if len(tensor.shape) == 3:
        B, L, C = tensor.shape
        H = W = int(L**0.5)
        return tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)
    return tensor

# --- Hàm lấy layer theo đường dẫn ---
def get_layer_by_name(model, layer_name):
    """Lấy layer từ model bằng tên đường dẫn"""
    parts = layer_name.split('.')
    module = model
    for part in parts:
        module = getattr(module, part)
    return module

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading data...")
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    test_df = df[df['split'] == 'test'].dropna(subset=['standardized_species', 'image_path'])
    image_samples = test_df.sample(n=NUM_IMAGES_TO_VISUALIZE, random_state=42)

    preprocess = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    all_species = sorted(df['standardized_species'].dropna().unique())
    species_to_id = {s: i for i, s in enumerate(all_species)}
    id_to_species = {i: s for s, i in species_to_id.items()}
    num_species = len(species_to_id)

    print("Loading models...")
    mt_model = MT_MODEL_ARCH(num_species=num_species, num_traits=4, num_seg_classes=10)
    mt_model.load_state_dict(torch.load(MT_MODEL_PATH, map_location=DEVICE, weights_only=True))
    mt_model.to(DEVICE).eval()

    cls_model = timm.create_model(CLS_MODEL_NAME, pretrained=False, num_classes=num_species)
    cls_model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE, weights_only=True))
    cls_model.to(DEVICE).eval()

    # --- Chọn target layers cụ thể ---
    # Cho Swin-B: layer norm cuối cùng của block cuối cùng
    target_layer_mt_name = "backbone.layers_3.blocks.1.norm2"
    target_layer_mt = get_layer_by_name(mt_model, target_layer_mt_name)
    
    # Cho MaxViT: conv3_1x1 cuối cùng (projection layer của block cuối)
    target_layer_cls_name = "stages.3.blocks.1.conv.conv3_1x1"
    target_layer_cls = get_layer_by_name(cls_model, target_layer_cls_name)
    
    print(f"\nUsing target layer for MT: {target_layer_mt_name}")
    print(f"  -> {target_layer_mt}")
    print(f"Using target layer for CLS: {target_layer_cls_name}")
    print(f"  -> {target_layer_cls}")
    
    # --- Wrapper để model đa tác vụ chỉ trả về output của species ---
    class MTWrapper(torch.nn.Module):
        def __init__(self, model):
            super(MTWrapper, self).__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)['species']
    
    mt_wrapper = MTWrapper(mt_model)
    
    cam_mt = GradCAM(model=mt_wrapper, target_layers=[target_layer_mt], reshape_transform=reshape_transform_swin)
    cam_cls = GradCAM(model=cls_model, target_layers=[target_layer_cls])

    print(f"\nGenerating Grad-CAM for {NUM_IMAGES_TO_VISUALIZE} images...")
    for idx, (index, row) in enumerate(image_samples.iterrows()):
        image_path, true_species_name = row['image_path'], row['standardized_species']
        if pd.isna(image_path) or pd.isna(true_species_name): 
            continue
        
        true_species_id = species_to_id[true_species_name]
        
        try:
            rgb_img_pil = Image.open(image_path).convert('RGB')
            rgb_img = np.array(rgb_img_pil.resize((IMG_SIZE, IMG_SIZE))) / 255.0
            input_tensor = normalize(preprocess(rgb_img_pil)).unsqueeze(0).to(DEVICE)

            # Tạo CAM cho Multi-Task
            with torch.no_grad(): 
                mt_pred_id = mt_wrapper(input_tensor).argmax(dim=1).item()
            targets_mt = [ClassifierOutputTarget(mt_pred_id)]
            grayscale_cam_mt = cam_mt(input_tensor=input_tensor, targets=targets_mt)[0, :]
            visualization_mt = show_cam_on_image(rgb_img, grayscale_cam_mt, use_rgb=True)

            # Tạo CAM cho Cls-Only
            with torch.no_grad(): 
                cls_pred_id = cls_model(input_tensor).argmax(dim=1).item()
            targets_cls = [ClassifierOutputTarget(cls_pred_id)]
            grayscale_cam_cls = cam_cls(input_tensor=input_tensor, targets=targets_cls)[0, :]
            visualization_cls = show_cam_on_image(rgb_img, grayscale_cam_cls, use_rgb=True)

            # Ghép ảnh
            original_img_display = (rgb_img * 255).astype(np.uint8)
            concatenated_image = np.concatenate([original_img_display, visualization_mt, visualization_cls], axis=1)

            # Thêm text
            def get_short_name(name): 
                return ' '.join(name.split()[:2]) if isinstance(name, str) else "N/A"
            
            mt_correct = true_species_id == mt_pred_id
            cls_correct = true_species_id == cls_pred_id
            
            texts = [
                ("Original", f"GT: {get_short_name(id_to_species[true_species_id])}", (255, 255, 255)),
                ("Multi-Task", f"Pred: {get_short_name(id_to_species[mt_pred_id])}", (0, 255, 0) if mt_correct else (255, 0, 0)),
                ("Cls-Only", f"Pred: {get_short_name(id_to_species[cls_pred_id])}", (0, 255, 0) if cls_correct else (255, 0, 0)),
            ]
            
            final_image = concatenated_image.copy()
            for i, (title, pred_text, color) in enumerate(texts):
                pos_title = (i * IMG_SIZE + 5, 20)
                pos_pred = (i * IMG_SIZE + 5, 40)
                
                # Title (trắng)
                final_image = cv2.putText(final_image, title, pos_title, 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                # Prediction/GT text (màu theo đúng/sai)
                final_image = cv2.putText(final_image, pred_text, pos_pred, 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            save_path = os.path.join(OUTPUT_DIR, f"cam_{idx:03d}_{os.path.basename(image_path)}")
            Image.fromarray(final_image).save(save_path)
            
            status = f"MT:{'✓' if mt_correct else '✗'} CLS:{'✓' if cls_correct else '✗'}"
            print(f"[{idx+1}/{NUM_IMAGES_TO_VISUALIZE}] {status} | {os.path.basename(image_path)}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    print(f"\n✓ Done! Visualizations saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()