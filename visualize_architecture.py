# visualize_architecture_diagram.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_diagram():
    """Create a beautiful custom architecture diagram"""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    color_input = '#E8F4F8'
    color_backbone = '#B8E6F0'
    color_neck = '#88D8E8'
    color_head = '#FFE6CC'
    color_output = '#FFB380'
    
    # Helper function to draw boxes
    def draw_box(x, y, w, h, text, color, fontsize=10, fontweight='normal'):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=color,
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text,
               ha='center', va='center',
               fontsize=fontsize, fontweight=fontweight,
               wrap=True)
    
    # Helper function to draw arrows
    def draw_arrow(x1, y1, x2, y2, color='black', style='->'):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style,
            color=color,
            linewidth=2,
            mutation_scale=20
        )
        ax.add_patch(arrow)
    
    # Title
    ax.text(5, 11.5, 'Multi-Task Swin Transformer Architecture',
           ha='center', fontsize=18, fontweight='bold')
    
    # INPUT LAYER
    draw_box(4, 10, 2, 0.8, 'Input Image\n(3, 224, 224)', color_input, fontsize=11, fontweight='bold')
    draw_arrow(5, 10, 5, 9.5)
    
    # BACKBONE - Swin Transformer
    backbone_y = 8.5
    draw_box(3.5, backbone_y, 3, 1, 
            'Swin Transformer Backbone\n(swin_base_patch4_window7_224)',
            color_backbone, fontsize=10, fontweight='bold')
    
    # Backbone details
    stages = [
        ('Stage 1\n56×56×128', 7.5),
        ('Stage 2\n28×28×256', 6.5),
        ('Stage 3\n14×14×512', 5.5),
        ('Stage 4\n7×7×1024', 4.5),
    ]
    
    for stage_text, y_pos in stages:
        draw_box(3.8, y_pos, 2.4, 0.6, stage_text, color_backbone, fontsize=9)
        if y_pos > 4.5:
            draw_arrow(5, y_pos, 5, y_pos - 0.4)
    
    draw_arrow(5, 8.5, 5, 7.9)
    draw_arrow(5, 4.5, 5, 3.8)
    
    # FEATURE PYRAMID NETWORK (FPN)
    fpn_y = 3
    draw_box(3.5, fpn_y, 3, 0.8,
            'Feature Pyramid Network (FPN)\nMulti-scale feature fusion',
            color_neck, fontsize=10, fontweight='bold')
    draw_arrow(5, fpn_y, 5, fpn_y - 0.5)
    
    # TASK HEADS (3 branches)
    head_y = 1.5
    head_width = 2.2
    head_height = 1.2
    spacing = 2.5
    
    # Species Classification Head
    species_x = 0.5
    draw_arrow(5, fpn_y - 0.5, species_x + head_width/2, head_y + head_height)
    draw_box(species_x, head_y, head_width, head_height,
            'Species Head\n\nGlobal Pool\n↓\nFC(1024→512)\n↓\nFC(512→N species)',
            color_head, fontsize=9, fontweight='bold')
    draw_arrow(species_x + head_width/2, head_y, species_x + head_width/2, 0.5)
    draw_box(species_x, 0.1, head_width, 0.4,
            f'Species\nLogits (N,)',
            color_output, fontsize=9, fontweight='bold')
    
    # Trait Regression Head
    trait_x = species_x + spacing
    draw_arrow(5, fpn_y - 0.5, trait_x + head_width/2, head_y + head_height)
    draw_box(trait_x, head_y, head_width, head_height,
            'Trait Head\n\nGlobal Pool\n↓\nFC(1024→512)\n↓\nFC(512→4)',
            color_head, fontsize=9, fontweight='bold')
    draw_arrow(trait_x + head_width/2, head_y, trait_x + head_width/2, 0.5)
    draw_box(trait_x, 0.1, head_width, 0.4,
            'Traits\n(4,)',
            color_output, fontsize=9, fontweight='bold')
    
    # Segmentation Head
    seg_x = trait_x + spacing
    draw_arrow(5, fpn_y - 0.5, seg_x + head_width/2, head_y + head_height)
    draw_box(seg_x, head_y, head_width, head_height,
            'Segmentation Head\n\nConv(1024→512)\n↓\nUpsample 4×\n↓\nConv(512→10)',
            color_head, fontsize=9, fontweight='bold')
    draw_arrow(seg_x + head_width/2, head_y, seg_x + head_width/2, 0.5)
    draw_box(seg_x, 0.1, head_width, 0.4,
            'Seg Mask\n(10, 224, 224)',
            color_output, fontsize=9, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=color_input, edgecolor='black', label='Input'),
        mpatches.Patch(facecolor=color_backbone, edgecolor='black', label='Backbone'),
        mpatches.Patch(facecolor=color_neck, edgecolor='black', label='Feature Fusion'),
        mpatches.Patch(facecolor=color_head, edgecolor='black', label='Task Heads'),
        mpatches.Patch(facecolor=color_output, edgecolor='black', label='Outputs'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add parameter info
    info_text = """
    Total Parameters: ~88M
    Backbone: Swin-B (pretrained)
    Input Size: 224×224×3
    Tasks: Classification + Trait + Segmentation
    """
    ax.text(0.2, 11, info_text, fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('model_architecture_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: model_architecture_diagram.png")
    plt.close()

if __name__ == "__main__":
    create_architecture_diagram()