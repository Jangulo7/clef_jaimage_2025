# prepare_images.py

import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# --- Configuration ---
# Base directory for data
DATA_BASE_DIR = "./caption_prediction/data"

# Metadata directories and files
METADATA_DIR = "./metadata"
TRAIN_CAPTIONS_FILE = os.path.join(METADATA_DIR, "train_captions.csv")
TRAIN_CONCEPTS_FILE = os.path.join(METADATA_DIR, "train_concepts.csv")
VALID_CAPTIONS_FILE = os.path.join(METADATA_DIR, "valid_captions.csv")
VALID_CONCEPTS_FILE = os.path.join(METADATA_DIR, "valid_concepts.csv")

# Image directories
TRAIN_IMAGE_DIR = os.path.join(DATA_BASE_DIR, "train", "images")
VALID_IMAGE_DIR = os.path.join(DATA_BASE_DIR, "valid", "images")

# Input image extension
INPUT_IMAGE_EXTENSION = "jpg"

# Output directories for processed images
TRAIN_OUTPUT_IMAGE_DIR = os.path.join(DATA_BASE_DIR, "train", "processed_images")
VALID_OUTPUT_IMAGE_DIR = os.path.join(DATA_BASE_DIR, "valid", "processed_images")

# Optional: Target size for pre-resizing
# If None, the script will only validate and convert the format
TARGET_SIZE = (512, 512)  # Example: resize to 512x512. Set to None to skip.

# Image quality settings
RESIZE_FILTER = Image.Resampling.LANCZOS
OUTPUT_FORMAT = "JPEG"  # "PNG" or "JPEG"
JPEG_QUALITY = 90  # Quality for saving JPEG (1-95, where 95 is high quality)

def create_output_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created: {dir_path}")

def load_metadata_from_csv(captions_file, concepts_file):
    """
    Load image IDs from local CSV files containing captions and concepts
    """
    image_ids = set()
    
    try:
        # Load captions CSV
        if os.path.exists(captions_file):
            captions_df = pd.read_csv(captions_file)
            if 'image_id' in captions_df.columns:
                image_ids.update(captions_df['image_id'].unique())
            else:
                print(f"WARNING: 'image_id' column not found in {captions_file}")
        else:
            print(f"WARNING: File not found: {captions_file}")
            
        # Load concepts CSV
        if os.path.exists(concepts_file):
            concepts_df = pd.read_csv(concepts_file)
            if 'image_id' in concepts_df.columns:
                image_ids.update(concepts_df['image_id'].unique())
            else:
                print(f"WARNING: 'image_id' column not found in {concepts_file}")
        else:
            print(f"WARNING: File not found: {concepts_file}")
            
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        
    return list(image_ids)

def process_and_validate_images(all_image_ids, input_dir, output_dir):
    """
    Validates listed images (by image_id), converts them to RGB,
    optionally resizes them, and saves them to the output directory.
    """
    create_output_directory(output_dir)
    
    processed_count = 0
    error_count = 0
    missing_files = []
    corrupt_files = []

    print(f"\nStarting image processing. Output directory: {output_dir}")
    print(f"Target size for pre-resizing: {TARGET_SIZE if TARGET_SIZE else 'Not defined (original size maintained if possible)'}")
    print(f"Output format: {OUTPUT_FORMAT}")
    if OUTPUT_FORMAT == "JPEG":
        print(f"JPEG quality: {JPEG_QUALITY}")

    # Iterate through the unique image_ids obtained from metadata
    for img_id in tqdm(all_image_ids, desc="Processing Images"):
        # Assume image_id is the base name of the file
        input_image_filename = f"{img_id}.{INPUT_IMAGE_EXTENSION}"
        input_image_path = os.path.join(input_dir, input_image_filename)
        
        output_image_filename = f"{img_id}.{OUTPUT_FORMAT.lower()}"
        output_image_path = os.path.join(output_dir, output_image_filename)

        if not os.path.exists(input_image_path):
            missing_files.append(input_image_path)
            error_count += 1
            continue

        try:
            img = Image.open(input_image_path)

            # 1. Ensure the image is in RGB format
            # Radiology images are often 'L' (grayscale) or 'LA' (grayscale with alpha)
            if img.mode not in ["RGB", "RGBA"]:  # If it's not RGB or RGBA
                img = img.convert("RGB")
            elif img.mode == "RGBA":  # If it's RGBA, convert to RGB to discard alpha channel
                img = img.convert("RGB")

            # 2. Optional: Resize image
            if TARGET_SIZE:
                # Check if the image already has the desired size to avoid unnecessary resizing
                if img.size != TARGET_SIZE:
                    img = img.resize(TARGET_SIZE, RESIZE_FILTER)

            # 3. Save the processed image
            if OUTPUT_FORMAT == "JPEG":
                img.save(output_image_path, format=OUTPUT_FORMAT, quality=JPEG_QUALITY)
            else:  # For PNG or other formats that don't use 'quality'
                img.save(output_image_path, format=OUTPUT_FORMAT)
            processed_count += 1

        except FileNotFoundError:  # Should be caught by os.path.exists, but as a safeguard
            missing_files.append(input_image_path)
            error_count += 1
        except UnidentifiedImageError:  # Specifically for files that PIL cannot identify as an image
            corrupt_files.append(f"{input_image_path} (Error: Unrecognized image format or corrupt)")
            error_count += 1
        except Exception as e:
            corrupt_files.append(f"{input_image_path} (Error: {e})")
            error_count += 1
            
    print("\n--- Image Processing Summary ---")
    print(f"Images successfully processed: {processed_count}")
    print(f"Total errors (missing or corrupt): {error_count}")
    if missing_files:
        print(f"\nMissing files ({len(missing_files)}):")
        for f_path in missing_files[:10]:  # Show the first 10
            print(f"  {f_path}")
        if len(missing_files) > 10:
            print(f"  ...and {len(missing_files) - 10} more.")
    if corrupt_files:
        print(f"\nCorrupt or unprocessable files ({len(corrupt_files)}):")
        for f_info in corrupt_files[:10]:  # Show the first 10
             print(f"  {f_info}")
        if len(corrupt_files) > 10:
            print(f"  ...and {len(corrupt_files) - 10} more.")

    if error_count > 0:
        print("\nWARNING: Some images could not be processed. Review the list above.")
    else:
        print("All images found were successfully processed!")

def process_dataset_split(split_name, image_dir, output_dir, captions_file, concepts_file):
    """
    Process a specific dataset split (train or valid)
    """
    print(f"\n{'='*50}")
    print(f"Processing {split_name.upper()} dataset")
    print(f"{'='*50}")
    
    print(f"Loading metadata from CSV files for {split_name}...")
    image_ids = load_metadata_from_csv(captions_file, concepts_file)
    
    if not image_ids:
        print(f"No image_ids found in the metadata for {split_name}. Cannot process images.")
        return
        
    print(f"Found {len(image_ids)} unique image_ids in {split_name} metadata.")
    
    if not os.path.isdir(image_dir):
        print(f"Error: The input image directory '{image_dir}' does not exist.")
        print("Please check the path.")
        return
        
    process_and_validate_images(image_ids, image_dir, output_dir)
    
    print(f"\n{split_name.capitalize()} image processing and validation completed.")
    print(f"If successful, you can now update the `image_dir` path in your fine-tuning script to:")
    print(f"`image_dir = \"{output_dir}\"`")

if __name__ == "__main__":
    # Process training set
    process_dataset_split(
        "train", 
        TRAIN_IMAGE_DIR, 
        TRAIN_OUTPUT_IMAGE_DIR,
        TRAIN_CAPTIONS_FILE,
        TRAIN_CONCEPTS_FILE
    )
    
    # Process validation set
    process_dataset_split(
        "valid", 
        VALID_IMAGE_DIR, 
        VALID_OUTPUT_IMAGE_DIR,
        VALID_CAPTIONS_FILE,
        VALID_CONCEPTS_FILE
    )
    
    print("\nPreliminary image processing and validation completed for all datasets.")
    print("Remember: The AutoProcessor in your fine-tuning script will still perform model-specific transformations.")
    
    