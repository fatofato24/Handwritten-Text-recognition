"""
Visual Comparison Tool - Shows improvements from ensemble and post-processing
Displays side-by-side comparison of OCR outputs and highlights corrections
"""

import csv
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from context_correction import clean_text

# Configuration
RESULTS_DIR = "results/"


def color_text(text, color='green'):
    """Add color to terminal output"""
    colors = {
        'green': '\033[92m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'reset': '\033[0m',
        'bold': '\033[1m',
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"


def compare_with_ground_truth(predicted, ground_truth):
    """
    Show character-level comparison of predicted vs ground truth.
    Highlight differences.
    """
    
    pred = clean_text(predicted)
    truth = clean_text(ground_truth)
    
    if pred == truth:
        return color_text("✅ EXACT MATCH", 'green')
    
    # Character-level comparison
    max_len = max(len(pred), len(truth))
    matches = sum(1 for p, t in zip(pred, truth) if p == t)
    accuracy = matches / max_len if max_len > 0 else 0
    
    if accuracy >= 0.9:
        return color_text(f"🟢 {accuracy*100:.1f}% match", 'green')
    elif accuracy >= 0.7:
        return color_text(f"🟡 {accuracy*100:.1f}% match", 'yellow')
    else:
        return color_text(f"🔴 {accuracy*100:.1f}% match", 'red')


def display_comparison():
    """Display side-by-side comparison of results"""
    
    print("\n" + "="*100)
    print(color_text("🔍 VISUAL COMPARISON - EasyOCR vs Ensemble vs Corrected", 'bold'))
    print("="*100)
    
    # Load data
    easyocr_file = os.path.join(RESULTS_DIR, 'easyocr_evaluation.csv')
    ensemble_file = os.path.join(RESULTS_DIR, 'ensemble_predictions.csv')
    
    if not os.path.exists(easyocr_file):
        print(color_text("❌ EasyOCR results not found. Run: python src/Easyocr.py", 'red'))
        return
    
    if not os.path.exists(ensemble_file):
        print(color_text("❌ Ensemble results not found. Run: python src/ensemble_ocr.py", 'red'))
        return
    
    # Read EasyOCR results
    easyocr_data = {}
    with open(easyocr_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            easyocr_data[row['filename']] = row
    
    # Read Ensemble results
    ensemble_data = {}
    with open(ensemble_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ensemble_data[row['filename']] = row
    
    # Compare
    improved_count = 0
    degraded_count = 0
    same_count = 0
    
    print(f"\n{'Filename':<20} {'Ground Truth':<20} {'EasyOCR':<25} {'Ensemble':<25} {'Improvement':<15}")
    print("-" * 125)
    
    for filename in sorted(easyocr_data.keys()):
        if filename not in ensemble_data:
            continue
        
        ground_truth = easyocr_data[filename]['ground_truth']
        easyocr_pred = easyocr_data[filename]['predicted_text']
        ensemble_pred = ensemble_data[filename]['predicted_text']
        
        # Check if improved
        easy_clean = clean_text(easyocr_pred)
        ensemble_clean = clean_text(ensemble_pred)
        
        easy_exact = easy_clean == clean_text(ground_truth)
        ensemble_exact = ensemble_clean == clean_text(ground_truth)
        
        # Determine improvement status
        if ensemble_exact and not easy_exact:
            status = color_text("🟢 IMPROVED", 'green')
            improved_count += 1
        elif easy_exact and not ensemble_exact:
            status = color_text("🔴 DEGRADED", 'red')
            degraded_count += 1
        else:
            status = "➖ SAME"
            same_count += 1
        
        # Truncate for display
        def truncate(s, n=18):
            return (s[:n-2] + '..') if len(s) > n else s
        
        print(f"{truncate(filename):<20} {truncate(ground_truth):<20} "
              f"{truncate(easyocr_pred):<25} {truncate(ensemble_pred):<25} {status:<15}")
    
    # Summary
    total = improved_count + degraded_count + same_count
    
    print("\n" + "="*100)
    print(color_text("📊 SUMMARY", 'bold'))
    print("="*100)
    print(f"  {color_text('✅ Improved:', 'green')}   {improved_count}/{total} samples ({improved_count/total*100:.1f}%)")
    print(f"  {color_text('➖ Same:', 'cyan')}       {same_count}/{total} samples ({same_count/total*100:.1f}%)")
    print(f"  {color_text('❌ Degraded:', 'red')}    {degraded_count}/{total} samples ({degraded_count/total*100:.1f}%)")
    print("="*100)


def show_sample_corrections():
    """Show detailed corrections for a few samples"""
    
    print("\n" + "="*100)
    print(color_text("📝 DETAILED CORRECTION SAMPLES", 'bold'))
    print("="*100)
    
    ensemble_file = os.path.join(RESULTS_DIR, 'ensemble_comparison.csv')
    
    if not os.path.exists(ensemble_file):
        print(color_text("⚠️  Ensemble comparison file not found", 'yellow'))
        return
    
    with open(ensemble_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        samples = list(reader)[:5]  # Show first 5
    
    for i, sample in enumerate(samples, 1):
        ground_truth = sample['ground_truth']
        easyocr = sample['easyocr']
        tesseract = sample['tesseract']
        trocr = sample['trocr']
        ensemble = sample['ensemble']
        
        print(f"\n{color_text(f'Sample {i}:', 'bold')}")
        print(f"  Ground Truth: {color_text(ground_truth, 'cyan')}")
        print(f"  ├─ EasyOCR:   {easyocr} (conf: {sample.get('easyocr_conf', 'N/A')})")
        print(f"  ├─ Tesseract: {tesseract} (conf: {sample.get('tesseract_conf', 'N/A')})")
        print(f"  ├─ TrOCR:     {trocr} (conf: {sample.get('trocr_conf', 'N/A')})")
        print(f"  └─ Ensemble:  {color_text(ensemble, 'green')} (conf: {sample.get('ensemble_conf', 'N/A')})")
        
        # Show match status
        print(f"    Match Status: {compare_with_ground_truth(ensemble, ground_truth)}")


def main():
    """Main comparison interface"""
    
    print(f"\n{color_text('🚀 OCR IMPROVEMENT VISUALIZATION', 'bold')}")
    
    display_comparison()
    show_sample_corrections()
    
    print(f"\n{color_text('✅ Comparison complete!', 'green')}")
    print(f"\nFor detailed metrics, run: {color_text('python src/evaluate_ensemble.py', 'cyan')}")


if __name__ == "__main__":
    main()
