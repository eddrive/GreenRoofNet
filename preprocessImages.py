import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# Path to files in Google Drive
image_folder =  './dataset/images'
mask_folder = './dataset/masks'

# List of image files
image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')])
mask_files = sorted([os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.endswith('.png')])



def preprocess_image_and_mask(image_path, mask_path, target_size=(600, 600)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    magenta = np.array([255, 0, 255])
    binary_mask = np.all(mask_rgb == magenta, axis=-1).astype(np.uint8)
    binary_mask = (1 - binary_mask)  # Invert values: magenta=0, everything else=1
    binary_mask = cv2.resize(binary_mask, target_size, interpolation=cv2.INTER_NEAREST)
    binary_mask = np.expand_dims(binary_mask, axis=-1)  # Add channel dimension
    
    return image, binary_mask

# Initialize lists to hold all images and masks
all_images = []
all_masks = []

# Process all images and masks and store them in the lists
for i, (image_path, mask_path) in enumerate(zip(image_files, mask_files)):
    image, mask = preprocess_image_and_mask(image_path, mask_path)
    all_images.append(image)
    all_masks.append(mask)
    print(f'Processed image {i + 1}/{len(image_files)}')

# Convert lists to numpy arrays
all_images = np.array(all_images)
all_masks = np.array(all_masks)

# Save the numpy arrays as a single .npz file
# Ensure the directory exists
os.makedirs('./dataset/preprocessed', exist_ok=True)

# Save the numpy arrays
np.savez('./dataset/preprocessed/dataset_compressed.npz', images=all_images, masks=all_masks)
