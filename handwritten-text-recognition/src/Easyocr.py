"""
EasyOCR Baseline Model - Inference Script
Generates OCR predictions on handwritten text images using EasyOCR
and saves predictions to CSV for evaluation.

Author: Member 1
Date: Day 1
"""

import easyocr
import cv2
import os
import csv
from tqdm import tqdm
import sys
from context_correction import correct_text

# Configuration
DATA_PATH = "data/raw/"
LABEL_FILE = "data/labels.txt"
OUTPUT_DIR = "results/"
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "easyocr_predictions.csv")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize EasyOCR reader
print("🔄 Initializing EasyOCR reader...")
reader = easyocr.Reader(['en'])
print("✅ EasyOCR reader initialized\n")

# Track statistics
processed = 0
failed = 0
predictions_list = []

# Read labels and process images
print("📖 Reading labels from:", LABEL_FILE)
try:
    with open(LABEL_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"📊 Total samples to process: {len(lines)}\n")
except Exception as e:
    print(f"❌ Error reading label file: {e}")
    sys.exit(1)

# Process each image with EasyOCR
print("🖼️  Processing images with EasyOCR...")
for line in tqdm(lines, desc="Progress"):
    try:
        filename, ground_truth = line.split(',', 1)
        filename = filename.strip()
        ground_truth = ground_truth.strip()
    except:
        print(f"⚠️  Skipping malformed line: {line}")
        failed += 1
        continue

    img_path = os.path.join(DATA_PATH, filename)

    # Validate image path
    if not os.path.exists(img_path):
        print(f"⚠️  Image not found: {img_path}")
        failed += 1
        continue

    # Load image
    try:
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError("Failed to load image")
    except Exception as e:
        print(f"⚠️  Error loading image {filename}: {e}")
        failed += 1
        continue

    # Perform OCR inference
    try:
        result = reader.readtext(image)
        
        if len(result) == 0:
            predicted_text = ""
        else:
            # Extract text from results (each result is [bbox, text, confidence])
            predicted_text = " ".join([res[1] for res in result])
    except Exception as e:
        print(f"⚠️  OCR failed for {filename}: {e}")
        failed += 1
        continue

    # Store prediction
    predictions_list.append({
        'filename': filename,
        'ground_truth': ground_truth,
        'predicted_text': predicted_text
    })
    processed += 1

# Save predictions to CSV
print(f"\n💾 Saving predictions to: {PREDICTIONS_FILE}")
try:
    with open(PREDICTIONS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'ground_truth', 'predicted_text'])
        writer.writeheader()
        writer.writerows(predictions_list)
    print(f"✅ Predictions saved successfully\n")
except Exception as e:
    print(f"❌ Error saving predictions: {e}")
    sys.exit(1)

# Print summary statistics
print("="*50)
print("📈 EASYOCR INFERENCE SUMMARY")
print("="*50)
print(f"✅ Successfully processed: {processed}")
print(f"❌ Failed to process: {failed}")
print(f"📊 Total samples: {len(lines)}")
print(f"💾 Output file: {PREDICTIONS_FILE}")
print("="*50)
print("\n➡️  Next step: Run evaluate.py to compute metrics")