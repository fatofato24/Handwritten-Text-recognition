"""
Multi-Engine OCR Ensemble - Advanced Post-Processing
Combines predictions from EasyOCR, Tesseract, and TrOCR with intelligent voting
and confidence-weighted fusion for improved accuracy.

Research-backed approach:
- Ensemble voting with Levenshtein distance
- Confidence-weighted combination
- Character-level error correction
- Advanced spell-checking

Author: Research Team
Date: May 2, 2026
"""

import easyocr
import pytesseract
import cv2
import os
import csv
from tqdm import tqdm
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from context_correction import (
    clean_text, correct_text, ensemble_correct, 
    ensemble_vote, apply_char_replacements
)
import torch

# Configuration
DATA_PATH = "data/raw/"
LABEL_FILE = "data/labels.txt"
OUTPUT_DIR = "results/"
ENSEMBLE_PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "ensemble_predictions.csv")
ENSEMBLE_COMPARISON_FILE = os.path.join(OUTPUT_DIR, "ensemble_comparison.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Using device: {device}")

# ===========================
# INITIALIZE OCR ENGINES
# ===========================
print("🔄 Initializing OCR engines...")

# EasyOCR
print("  Loading EasyOCR...")
easyocr_reader = easyocr.Reader(['en'], gpu=(device == "cuda"))
print("  ✅ EasyOCR ready")

# TrOCR
print("  Loading TrOCR...")
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
if device == "cuda":
    trocr_model = trocr_model.to("cuda")
print("  ✅ TrOCR ready")

print("✅ All engines initialized\n")

# ===========================
# SINGLE ENGINE INFERENCE
# ===========================
def easyocr_inference(image):
    """Run EasyOCR on image"""
    try:
        result = easyocr_reader.readtext(image)
        if len(result) == 0:
            return "", 0.0
        
        # Extract text and average confidence
        texts = [res[1] for res in result]
        confidences = [res[2] for res in result]
        
        predicted_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return predicted_text, avg_confidence
    except Exception as e:
        print(f"    ⚠️  EasyOCR error: {e}")
        return "", 0.0


def tesseract_inference(image):
    """Run Tesseract on image"""
    try:
        # Convert BGR to GRAY
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Preprocess for better results
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        
        # Run Tesseract
        text = pytesseract.image_to_string(thresh)
        
        # Tesseract doesn't provide confidence easily, estimate from text quality
        # Heuristic: no special chars + normal length = higher confidence
        if text and len(text) > 3:
            confidence = 0.75
        elif text:
            confidence = 0.50
        else:
            confidence = 0.0
        
        return text.strip(), confidence
    except Exception as e:
        print(f"    ⚠️  Tesseract error: {e}")
        return "", 0.0


def trocr_inference(image):
    """Run TrOCR on image"""
    try:
        # Convert BGR to RGB and create PIL Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Run inference
        pixel_values = trocr_processor(images=pil_image, return_tensors="pt").pixel_values
        
        if device == "cuda":
            pixel_values = pixel_values.to("cuda")
        
        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)
        
        predicted_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # TrOCR is generally reliable, assign high confidence
        confidence = 0.85
        
        return predicted_text.strip(), confidence
    except Exception as e:
        print(f"    ⚠️  TrOCR error: {e}")
        return "", 0.0


# ===========================
# ENSEMBLE PROCESSING
# ===========================
def process_with_ensemble(image_path):
    """
    Process image with all three engines and combine results using ensemble voting.
    """
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return {}, {}
    
    # Run individual engines
    easyocr_text, easyocr_conf = easyocr_inference(image)
    tesseract_text, tesseract_conf = tesseract_inference(image)
    trocr_text, trocr_conf = trocr_inference(image)
    
    # Store individual results
    individual_results = {
        'easyocr': easyocr_text,
        'tesseract': tesseract_text,
        'trocr': trocr_text,
    }
    
    individual_conf = {
        'easyocr': easyocr_conf,
        'tesseract': tesseract_conf,
        'trocr': trocr_conf,
    }
    
    # Ensemble voting
    predictions_dict = {
        'easyocr': (easyocr_text, easyocr_conf),
        'tesseract': (tesseract_text, tesseract_conf),
        'trocr': (trocr_text, trocr_conf),
    }
    
    ensemble_text, ensemble_conf = ensemble_correct(predictions_dict)
    
    return {
        'easyocr': easyocr_text,
        'tesseract': tesseract_text,
        'trocr': trocr_text,
        'ensemble': ensemble_text,
    }, {
        'easyocr': easyocr_conf,
        'tesseract': tesseract_conf,
        'trocr': trocr_conf,
        'ensemble': ensemble_conf,
    }


# ===========================
# MAIN PROCESSING
# ===========================
print("📖 Reading labels from:", LABEL_FILE)

try:
    with open(LABEL_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"📊 Total samples to process: {len(lines)}\n")
except Exception as e:
    print(f"❌ Error reading label file: {e}")
    exit(1)

# Process images
predictions_list = []
comparison_list = []

print("🖼️  Processing images with ensemble OCR...\n")

for line in tqdm(lines, desc="Progress"):
    try:
        filename, ground_truth = line.split(',', 1)
        filename = filename.strip()
        ground_truth = ground_truth.strip()
    except:
        continue
    
    img_path = os.path.join(DATA_PATH, filename)
    
    if not os.path.exists(img_path):
        continue
    
    try:
        predictions, confidences = process_with_ensemble(img_path)
        
        if not predictions:
            continue
        
        # Store ensemble prediction
        predictions_list.append({
            'filename': filename,
            'ground_truth': ground_truth,
            'predicted_text': predictions.get('ensemble', ''),
            'ensemble_confidence': confidences.get('ensemble', 0.0),
        })
        
        # Store comparison of all engines
        comparison_list.append({
            'filename': filename,
            'ground_truth': ground_truth,
            'easyocr': predictions.get('easyocr', ''),
            'easyocr_conf': confidences.get('easyocr', 0.0),
            'tesseract': predictions.get('tesseract', ''),
            'tesseract_conf': confidences.get('tesseract', 0.0),
            'trocr': predictions.get('trocr', ''),
            'trocr_conf': confidences.get('trocr', 0.0),
            'ensemble': predictions.get('ensemble', ''),
            'ensemble_conf': confidences.get('ensemble', 0.0),
        })
        
    except Exception as e:
        print(f"⚠️  Error processing {filename}: {e}")
        continue

# ===========================
# SAVE RESULTS
# ===========================
print(f"\n💾 Saving ensemble predictions to: {ENSEMBLE_PREDICTIONS_FILE}")

if predictions_list:
    with open(ENSEMBLE_PREDICTIONS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'filename', 'ground_truth', 'predicted_text', 'ensemble_confidence'
        ])
        writer.writeheader()
        writer.writerows(predictions_list)
    print(f"✅ Saved {len(predictions_list)} predictions")

print(f"\n📊 Saving engine comparison to: {ENSEMBLE_COMPARISON_FILE}")

if comparison_list:
    with open(ENSEMBLE_COMPARISON_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'filename', 'ground_truth',
            'easyocr', 'easyocr_conf',
            'tesseract', 'tesseract_conf',
            'trocr', 'trocr_conf',
            'ensemble', 'ensemble_conf'
        ])
        writer.writeheader()
        writer.writerows(comparison_list)
    print(f"✅ Saved {len(comparison_list)} comparisons")

print("\n✅ Ensemble OCR processing complete!")
