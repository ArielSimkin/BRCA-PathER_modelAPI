#imports

import os
import openslide
from PIL import Image
import numpy as np
import splitfolders
import random

#building the DataSet

EPSILON = 0.9

print("finished setup")

# Set Up Directory Paths
base_dir = "/mnt/c/Users/owner/System_Progect/Data" # Path to the directory containing WSI folders
output_dir = "/mnt/c/Users/owner/System_Progect/Processed_Data"  # Output directory for processed tiles

# Identify ER Status from Folder Names
folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

# Dictionary to map folder paths to their ER status
er_status_map = {}

for folder in folders:
    if 'ER+' in folder:
        er_status_map[os.path.join(base_dir, folder)] = 'ER+'
    elif 'ER-' in folder:
        er_status_map[os.path.join(base_dir, folder)] = 'ER-'

# Function to normalize a tile
def normalize_tile(tile):
    tile_array = np.array(tile).astype(np.float32)
    tile_array /= 255.0  # Normalize to [0, 1]
    return Image.fromarray((tile_array * 255).astype(np.uint8))

# Function to save a tile
def save_tile(tile, er_status, output_dir, wsi_name, tile_index):

    rnd = np.random.rand()

    if rnd > EPSILON:
        er_dir = os.path.join(output_dir, er_status)
        os.makedirs(er_dir, exist_ok=True)
        tile_filename = f"{wsi_name}_tile_{tile_index}.png"
        tile.save(os.path.join(er_dir, tile_filename))

# Function to check if a tile is more than 80% white
def is_tile_not_white(tile, threshold=0.8):
    tile_array = np.array(tile)
    # Convert the tile to grayscale and check the percentage of white pixels
    gray_tile = np.dot(tile_array[...,:3], [0.2989, 0.587, 0.114])  # Convert to grayscale
    white_pixels = np.sum(gray_tile > 200)  # Pixels close to white (255)
    total_pixels = gray_tile.size
    white_ratio = white_pixels / total_pixels
    return white_ratio <= threshold

# Tile size
tile_size = 299

# Process Each WSI
for folder_path, er_status in er_status_map.items():
    # List all WSI files in the folder
    wsi_files = [f for f in os.listdir(folder_path) if f.endswith('.svs')]

    for wsi_file in wsi_files:
        wsi_path = os.path.join(folder_path, wsi_file)
        wsi_name = os.path.splitext(wsi_file)[0]

        # Open the WSI
        slide = openslide.OpenSlide(wsi_path)

        # Get the dimensions of the WSI
        width, height = slide.dimensions

        tile_index = 0

        # Iterate over the WSI to extract tiles
        for x in range(0, width, tile_size):
            for y in range(0, height, tile_size):
                # Ensure the tile is within the slide dimensions
                if x + tile_size <= width and y + tile_size <= height:
                    tile = slide.read_region((x, y), 0, (tile_size, tile_size))
                    tile = tile.convert('RGB')  # Convert to RGB
                    if is_tile_not_white(tile):  # Filter out tiles that are too white
                        tile = normalize_tile(tile)  # Normalize the tile
                        save_tile(tile, er_status, output_dir, wsi_name, tile_index)
                        tile_index += 1

# Split Tiles into Train, Validation, and Test Sets
split_ratio = (0.70, 0.15, 0.15)  # 70% train, 15% val, 15% test

# Perform the split using the splitfolders package
splitfolders.ratio(output_dir, output=output_dir + '_split', seed=1337, ratio=split_ratio)

print("Tiles have been processed and dataset split successfully.")

#Augmentation

def balance_classes(directory, augmentation_fn):
    """
    Balances the number of images in each class folder within the specified directory.

    Parameters:
    - directory (str): Path to the directory containing class subdirectories.
    - augmentation_fn (function): Function to apply for data augmentation.
    """
    # Step 1: Calculate Class Imbalance
    class_counts = {}
    class_files = {}

    for pack_role in os.listdir(directory):
        pack_path = os.path.join(directory, pack_role)
        for class_name in os.listdir(pack_path):
            class_path = os.path.join(pack_path, class_name)
            if os.path.isdir(class_path):
                files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                class_counts[class_name] = len(files)
                class_files[class_name] = files

        print(class_counts)

        if len(class_counts) != 2:
            raise ValueError("This script supports datasets with exactly two classes.")

        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)

        majority_count = class_counts[majority_class]
        minority_count = class_counts[minority_class]

        # Step 2: Compute Target Class Sizes
        gap = majority_count - minority_count
        undersample_count = int(gap * 0.7)
        augment_count = gap - undersample_count

        # Step 3: Undersample the Majority Class
        majority_files = class_files[majority_class]
        files_to_remove = random.sample(majority_files, undersample_count)
        for file in files_to_remove:
            os.remove(file)

        # Step 4: Augment the Minority Class
        minority_files = class_files[minority_class]
        augmented_images = 0
        while augmented_images < augment_count:
            for file in minority_files:
                if augmented_images >= augment_count:
                    break
                with Image.open(file) as img:
                    augmented_image = augmentation_fn(img)
                    augmented_image.save(os.path.join(pack_path, minority_class, f"aug_{augmented_images}.png"))
                    augmented_images += 1

def rotate_image(image, angle=90):
    """
    Rotates the input image by the specified angle.

    Parameters:
    - image (PIL.Image): Image to rotate.
    - angle (int): Angle by which to rotate the image.

    Returns:
    - PIL.Image: Rotated image.
    """
    return image.rotate(angle)

# Example usage
dataset_directory = output_dir + "_split"
balance_classes(dataset_directory, rotate_image)
