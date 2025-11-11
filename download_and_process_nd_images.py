import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import os
from tqdm import tqdm
import json
import argparse
# import ast # Không cần thiết nữa

# --- Bắt đầu phần code đầy đủ ---

parser = argparse.ArgumentParser(description='Download, process and save ND images')
parser.add_argument('--save_dir', type=str, help='The directory where processed images will be saved', required=True)
args = parser.parse_args()

save_dir = args.save_dir

# Check for necessary directories and files
if not os.path.exists('ND_Processing_Files'):
    raise FileNotFoundError('The ND_Processing_Files directory is not found in your current working directory. Please ensure you are in the root of the fish-vista repo.')

if not os.path.exists(save_dir):
    print(f"Creating save directory at: {save_dir}")
    os.makedirs(save_dir)

# Load metadata and mappings
# --- SỬA LỖI 1: Sửa tên file CSV ---
try:
    nd_df = pd.read_csv(os.path.join('ND_Processing_Files', 'ND_data.csv'))
except FileNotFoundError:
    print("Error: 'ND_Processing_Files/ND_data.csv' not found.")
    exit()

with open(os.path.join('ND_Processing_Files', 'nd_filenames_bboxes_map.json'), 'r') as f:
    nd_filenames_bboxes_map = json.load(f)
seg_mask_dir = os.path.join('ND_Processing_Files', 'ND_background_masks')

# Start processing loop
print(f"Starting download and processing for {len(nd_df)} images...")
for i, row in tqdm(nd_df.iterrows(), total=len(nd_df)):
    
    target_filename = row['filename']
    target_filepath = os.path.join(save_dir, target_filename)

    # Bỏ qua nếu file đã tồn tại
    if os.path.exists(target_filepath):
        continue

    # Bọc toàn bộ logic trong try...except
    try:
        download_url = row['original_url']
        
        # Download image with a timeout
        response = requests.get(download_url, timeout=60)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        
        # Convert response content to a PIL Image
        image = Image.open(BytesIO(response.content))
        
        # Buộc load ảnh để phát hiện lỗi sớm
        image.load() 

        # --- SỬA LỖI 2: Bỏ ast.literal_eval ---
        # Get bounding box and padding info
        target_bbox = nd_filenames_bboxes_map[target_filename]
        left, upper, right, lower = target_bbox
        max_width, max_height = image.size
        
        try:
            padding = int(row['bbox_padding'])
        except (ValueError, TypeError):
            print(f"\nWarning: Could not parse padding for {target_filename}. Using default padding of 20.")
            padding = 20
        
        # Apply padding safely
        padded_left = max(left - padding, 0)
        padded_upper = max(upper - padding, 0)
        padded_right = min(right + padding, max_width)
        padded_lower = min(lower + padding, max_height)
        
        # Crop the image using the adjusted, padded bounding box
        cropped_image = image.crop((padded_left, padded_upper, padded_right, padded_lower))

        # Find and open the corresponding background mask
        assert target_filename.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')), f'Unexpected filename extension for {target_filename}'
        base_filename = os.path.splitext(target_filename)[0]
        target_seg_mask_file = base_filename + '.png'

        mask_path = os.path.join(seg_mask_dir, target_seg_mask_file)
        if os.path.exists(mask_path):
            mask_image = Image.open(mask_path).convert('L')
        else:
            print(f'\nWarning: Segmentation mask not found for target image {target_filename}. Skipping...')
            continue
        
        # Ensure mask is the same size as the cropped image
        if mask_image.size != cropped_image.size:
             mask_image = mask_image.resize(cropped_image.size, Image.NEAREST)

        # Create a white background and composite the final image
        white_image = Image.new("RGB", cropped_image.size, (255, 255, 255))
        result_image = Image.composite(cropped_image.convert("RGB"), white_image, mask_image)
        
        # Save the final processed image
        result_image.save(target_filepath)

    # Catch specific errors related to download, image processing, and data format
    except (OSError, requests.exceptions.RequestException, UnidentifiedImageError, KeyError, TypeError, ValueError) as e:
        print(f"\nSkipping {target_filename} due to an error: {e}")
        # 'continue' will move to the next item in the loop
        continue

print("Processing complete.")
# --- Kết thúc phần code đầy đủ ---