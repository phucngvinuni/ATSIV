import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms
import os

from multitask_model import MultiTaskSwinTransformer

# --- CẤU HÌNH ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "multitask_visualizations"
MASTER_CSV = "master_dataset.csv"
IMG_SIZE = 224
NUM_IMAGES = 10  # Số lượng ảnh muốn visualize
MT_MODEL_PATH = "./checkpoints_improved/best_model_improved.pth"

# Trait names (customize theo dataset của bạn)
TRAIT_NAMES = ['Elongated', 'Compressed', 'Depth', 'Length']  # Example names

# Segmentation class names
SEG_CLASSES = {
    0: 'Background',
    1: 'Body',
    2: 'Eye',
    3: 'Fin',
    4: 'Head',
    5: 'Tail',
    6: 'Mouth',
    7: 'Scale',
    8: 'Dorsal',
    9: 'Other'
}

def create_colored_segmentation(seg_mask, num_classes=10):
    """Tạo segmentation mask với màu sắc đẹp"""
    # Sử dụng colormap đẹp
    colors = plt.cm.get_cmap('tab10', num_classes)
    h, w = seg_mask.shape
    colored_mask = np.zeros((h, w, 3))
    
    for class_id in range(num_classes):
        mask = seg_mask == class_id
        colored_mask[mask] = colors(class_id)[:3]
    
    return (colored_mask * 255).astype(np.uint8)

def overlay_segmentation(image, seg_mask, alpha=0.4):
    """Overlay segmentation mask lên ảnh gốc"""
    image_np = np.array(image)
    colored_mask = create_colored_segmentation(seg_mask)
    
    # Blend
    overlay = (image_np * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    return overlay

def visualize_multitask_outputs(save_path='multitask_outputs.png', num_samples=5):
    """
    Visualize all outputs from multi-task model
    Layout: 
    - Row 1: Original images
    - Row 2: Segmentation overlay
    - Row 3: Species prediction + Traits bar chart
    """
    
    print("Loading data and model...")
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    test_df = df[df['split'] == 'test'].dropna(subset=['standardized_species', 'image_path'])
    
    all_species = sorted(df['standardized_species'].dropna().unique())
    species_to_id = {s: i for i, s in enumerate(all_species)}
    id_to_species = {i: s for s, i in species_to_id.items()}
    num_species = len(species_to_id)
    
    # Load model
    mt_model = MultiTaskSwinTransformer(num_species=num_species, num_traits=4, num_seg_classes=10)
    mt_model.load_state_dict(torch.load(MT_MODEL_PATH, map_location=DEVICE, weights_only=True))
    mt_model.to(DEVICE).eval()
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Sample images
    samples = test_df.sample(n=min(num_samples, len(test_df)), random_state=42)
    
    # Create figure
    fig = plt.figure(figsize=(5 * num_samples, 15))
    gs = fig.add_gridspec(3, num_samples, hspace=0.3, wspace=0.2)
    
    print(f"Processing {len(samples)} images...")
    
    for col_idx, (_, row) in enumerate(samples.iterrows()):
        try:
            image_path = row['image_path']
            true_species = row['standardized_species']
            
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_resized = img.resize((IMG_SIZE, IMG_SIZE))
            img_tensor = normalize(preprocess(img)).unsqueeze(0).to(DEVICE)
            
            # Inference
            with torch.no_grad():
                outputs = mt_model(img_tensor)
            
            # Parse outputs
            species_logits = outputs['species'][0]
            species_probs = torch.softmax(species_logits, dim=0)
            pred_species_id = species_probs.argmax().item()
            pred_species_name = id_to_species[pred_species_id]
            species_conf = species_probs[pred_species_id].item()
            
            traits = outputs['traits'].sigmoid().cpu().numpy()[0]
            
            seg_mask = outputs['segmentation'].argmax(dim=1)[0].cpu().numpy()
            
            # --- Row 1: Original Image ---
            ax1 = fig.add_subplot(gs[0, col_idx])
            ax1.imshow(img_resized)
            ax1.set_title(f'Original\nGT: {true_species[:25]}', fontsize=10, fontweight='bold')
            ax1.axis('off')
            
            # --- Row 2: Segmentation Overlay ---
            ax2 = fig.add_subplot(gs[1, col_idx])
            overlay = overlay_segmentation(img_resized, seg_mask, alpha=0.5)
            ax2.imshow(overlay)
            
            # Count pixels per class
            unique, counts = np.unique(seg_mask, return_counts=True)
            dominant_classes = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:3]
            seg_info = ', '.join([f"{SEG_CLASSES.get(c, 'Unknown')}" for c, _ in dominant_classes])
            
            ax2.set_title(f'Segmentation\nMain: {seg_info}', fontsize=9)
            ax2.axis('off')
            
            # --- Row 3: Species + Traits ---
            ax3 = fig.add_subplot(gs[2, col_idx])
            
            # Remove axis for custom layout
            ax3.axis('off')
            
            # Species prediction text
            correct = '✓' if pred_species_name == true_species else '✗'
            color = 'green' if pred_species_name == true_species else 'red'
            
            species_text = f"{correct} Prediction:\n{pred_species_name[:30]}\nConf: {species_conf:.2%}"
            ax3.text(0.5, 0.85, species_text, 
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                    transform=ax3.transAxes)
            
            # Traits bar chart
            trait_ax = ax3.inset_axes([0.1, 0.05, 0.8, 0.6])
            colors_bar = plt.cm.viridis(traits)
            bars = trait_ax.barh(range(len(traits)), traits, color=colors_bar, alpha=0.8)
            trait_ax.set_yticks(range(len(traits)))
            trait_ax.set_yticklabels(TRAIT_NAMES, fontsize=8)
            trait_ax.set_xlim(0, 1)
            trait_ax.set_xlabel('Value', fontsize=8)
            trait_ax.set_title('Predicted Traits', fontsize=9, fontweight='bold')
            trait_ax.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, traits)):
                trait_ax.text(val + 0.02, i, f'{val:.3f}', 
                            va='center', fontsize=7)
            
            print(f"[{col_idx+1}/{len(samples)}] Processed: {os.path.basename(image_path)}")
            
        except Exception as e:
            print(f"Error processing image {col_idx}: {e}")
            continue
    
    # Add legend for segmentation colors
    legend_elements = [mpatches.Patch(facecolor=plt.cm.tab10(i), 
                                     label=SEG_CLASSES.get(i, f'Class {i}'))
                      for i in range(len(SEG_CLASSES))]
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=5, fontsize=9, title='Segmentation Classes',
              bbox_to_anchor=(0.5, -0.02))
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {save_path}")
    plt.close()

def visualize_single_detailed(image_path, save_path='single_multitask.png'):
    """
    Detailed visualization for a single image
    """
    print("Loading model...")
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    
    all_species = sorted(df['standardized_species'].dropna().unique())
    species_to_id = {s: i for i, s in enumerate(all_species)}
    id_to_species = {i: s for s, i in species_to_id.items()}
    num_species = len(species_to_id)
    
    mt_model = MultiTaskSwinTransformer(num_species=num_species, num_traits=4, num_seg_classes=10)
    mt_model.load_state_dict(torch.load(MT_MODEL_PATH, map_location=DEVICE, weights_only=True))
    mt_model.to(DEVICE).eval()
    
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_tensor = normalize(preprocess(img)).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        outputs = mt_model(img_tensor)
    
    species_probs = torch.softmax(outputs['species'][0], dim=0)
    top5_probs, top5_ids = species_probs.topk(5)
    
    traits = outputs['traits'].sigmoid().cpu().numpy()[0]
    seg_mask = outputs['segmentation'].argmax(dim=1)[0].cpu().numpy()
    
    # Create detailed figure
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_resized)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Segmentation overlay
    ax2 = fig.add_subplot(gs[0, 1])
    overlay = overlay_segmentation(img_resized, seg_mask, alpha=0.5)
    ax2.imshow(overlay)
    ax2.set_title('Segmentation Overlay', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Pure segmentation mask
    ax3 = fig.add_subplot(gs[0, 2])
    colored_mask = create_colored_segmentation(seg_mask)
    ax3.imshow(colored_mask)
    ax3.set_title('Segmentation Mask', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Top-5 species predictions
    ax4 = fig.add_subplot(gs[1, 0])
    species_names = [id_to_species[i.item()][:30] for i in top5_ids]
    probabilities = top5_probs.cpu().numpy()
    
    colors_species = plt.cm.RdYlGn(probabilities)
    bars = ax4.barh(range(5), probabilities, color=colors_species, alpha=0.8)
    ax4.set_yticks(range(5))
    ax4.set_yticklabels(species_names, fontsize=10)
    ax4.set_xlabel('Confidence', fontsize=11)
    ax4.set_title('Top-5 Species Predictions', fontsize=14, fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, probabilities)):
        ax4.text(val + 0.01, i, f'{val:.2%}', va='center', fontsize=9)
    
    # Traits
    ax5 = fig.add_subplot(gs[1, 1])
    colors_traits = plt.cm.viridis(traits)
    bars_traits = ax5.bar(range(len(traits)), traits, color=colors_traits, alpha=0.8)
    ax5.set_xticks(range(len(traits)))
    ax5.set_xticklabels(TRAIT_NAMES, fontsize=10)
    ax5.set_ylabel('Value', fontsize=11)
    ax5.set_ylim(0, 1)
    ax5.set_title('Predicted Traits', fontsize=14, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars_traits, traits):
        ax5.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Segmentation statistics
    ax6 = fig.add_subplot(gs[1, 2])
    unique, counts = np.unique(seg_mask, return_counts=True)
    percentages = counts / counts.sum() * 100
    
    # Filter out very small classes
    significant = percentages > 1.0
    filtered_classes = unique[significant]
    filtered_percentages = percentages[significant]
    
    colors_seg = [plt.cm.tab10(c) for c in filtered_classes]
    wedges, texts, autotexts = ax6.pie(filtered_percentages, 
                                        labels=[SEG_CLASSES.get(c, f'C{c}') for c in filtered_classes],
                                        colors=colors_seg,
                                        autopct='%1.1f%%',
                                        startangle=90)
    ax6.set_title('Segmentation Distribution', fontsize=14, fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print("="*60)
    print("Multi-Task Model Output Visualization")
    print("="*60)
    
    # Grid visualization
    print("\n[1/2] Creating multi-sample grid...")
    visualize_multitask_outputs(
        save_path=os.path.join(OUTPUT_DIR, 'multitask_grid.png'),
        num_samples=NUM_IMAGES
    )
    
    # Single detailed visualization
    print("\n[2/2] Creating detailed single-image visualization...")
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    test_df = df[df['split'] == 'test'].dropna(subset=['image_path'])
    sample_image = test_df.sample(1, random_state=42).iloc[0]['image_path']
    
    visualize_single_detailed(
        image_path=sample_image,
        save_path=os.path.join(OUTPUT_DIR, 'multitask_detailed.png')
    )
    
    print("\n" + "="*60)
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}/")
    print("="*60)

if __name__ == "__main__":
    main()