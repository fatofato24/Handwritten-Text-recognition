"""
Evaluation Script for Handwritten Text Recognition - EasyOCR Baseline
Computes comprehensive metrics comparing predictions with ground truth.

This script:
- Reads EasyOCR predictions from CSV
- Calculates word-level accuracy, Levenshtein distance, CER
- Generates detailed per-sample metrics
- Computes aggregate statistics
- Saves results to structured CSV format
- Provides analysis and observations

Author: Member 3
Date: Day 1
"""

import csv
import os
import sys
from pathlib import Path
from typing import List, Dict
import statistics

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    normalize_text,
    calculate_levenshtein_ratio,
    calculate_word_accuracy,
    calculate_character_error_rate,
    is_correct_match,
    get_detailed_metrics
)

# Configuration
PREDICTIONS_FILE = "results/easyocr_predictions.csv"
EVALUATION_FILE = "results/easyocr_evaluation.csv"
SUMMARY_FILE = "results/easyocr_summary.txt"

# Create results directory if needed
os.makedirs("results", exist_ok=True)


def load_predictions(predictions_file: str) -> List[Dict]:
    """Load predictions from CSV file."""
    predictions = []
    try:
        with open(predictions_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                predictions.append(row)
        print(f"✅ Loaded {len(predictions)} predictions from {predictions_file}")
        return predictions
    except FileNotFoundError:
        print(f"❌ Error: Predictions file not found: {predictions_file}")
        print(f"   Please run Easyocr.py first to generate predictions")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading predictions file: {e}")
        sys.exit(1)


def evaluate_predictions(predictions: List[Dict]) -> Dict:
    """
    Evaluate each prediction and compute metrics.
    
    Returns:
        Dictionary containing:
        - evaluation_results: List of per-sample metrics
        - aggregate_stats: Overall statistics
    """
    print("\n📊 Evaluating predictions...")
    evaluation_results = []
    
    for i, pred in enumerate(predictions):
        filename = pred.get('filename', '')
        ground_truth = pred.get('ground_truth', '')
        predicted_text = pred.get('predicted_text', '')
        
        # Calculate metrics
        metrics = get_detailed_metrics(predicted_text, ground_truth)
        
        # Build evaluation row
        eval_row = {
            'index': i + 1,
            'filename': filename,
            'ground_truth': ground_truth,
            'predicted_text': predicted_text,
            'levenshtein_ratio': f"{metrics['levenshtein_ratio']:.4f}",
            'word_accuracy': f"{metrics['word_accuracy']:.4f}",
            'character_error_rate': f"{metrics['character_error_rate']:.4f}",
            'exact_match': metrics['exact_match'],
            'correct_70': metrics['is_correct_70'],
            'correct_80': metrics['is_correct_80'],
            'correct_90': metrics['is_correct_90'],
        }
        
        evaluation_results.append({
            'row': eval_row,
            'metrics': metrics
        })
    
    return evaluation_results


def compute_aggregate_stats(evaluation_results: List[Dict]) -> Dict:
    """
    Compute aggregate statistics across all predictions.
    """
    if not evaluation_results:
        print("❌ No evaluation results to aggregate")
        return {}
    
    n = len(evaluation_results)
    
    # Extract metric values
    lev_ratios = [r['metrics']['levenshtein_ratio'] for r in evaluation_results]
    word_accs = [r['metrics']['word_accuracy'] for r in evaluation_results]
    cers = [r['metrics']['character_error_rate'] for r in evaluation_results]
    exact_matches = sum(1 for r in evaluation_results if r['metrics']['exact_match'])
    correct_70 = sum(1 for r in evaluation_results if r['metrics']['is_correct_70'])
    correct_80 = sum(1 for r in evaluation_results if r['metrics']['is_correct_80'])
    correct_90 = sum(1 for r in evaluation_results if r['metrics']['is_correct_90'])
    
    stats = {
        'total_samples': n,
        'exact_match_count': exact_matches,
        'exact_match_pct': (exact_matches / n * 100) if n > 0 else 0,
        'correct_70_count': correct_70,
        'correct_70_pct': (correct_70 / n * 100) if n > 0 else 0,
        'correct_80_count': correct_80,
        'correct_80_pct': (correct_80 / n * 100) if n > 0 else 0,
        'correct_90_count': correct_90,
        'correct_90_pct': (correct_90 / n * 100) if n > 0 else 0,
        'avg_levenshtein': statistics.mean(lev_ratios),
        'median_levenshtein': statistics.median(lev_ratios),
        'min_levenshtein': min(lev_ratios),
        'max_levenshtein': max(lev_ratios),
        'stdev_levenshtein': statistics.stdev(lev_ratios) if n > 1 else 0,
        'avg_word_accuracy': statistics.mean(word_accs),
        'median_word_accuracy': statistics.median(word_accs),
        'min_word_accuracy': min(word_accs),
        'max_word_accuracy': max(word_accs),
        'stdev_word_accuracy': statistics.stdev(word_accs) if n > 1 else 0,
        'avg_cer': statistics.mean(cers),
        'median_cer': statistics.median(cers),
        'min_cer': min(cers),
        'max_cer': max(cers),
        'stdev_cer': statistics.stdev(cers) if n > 1 else 0,
    }
    
    return stats


def save_evaluation_csv(evaluation_results: List[Dict], output_file: str):
    """Save detailed evaluation results to CSV."""
    print(f"\n💾 Saving evaluation results to: {output_file}")
    
    try:
        fieldnames = [
            'index', 'filename', 'ground_truth', 'predicted_text',
            'levenshtein_ratio', 'word_accuracy', 'character_error_rate',
            'exact_match', 'correct_70', 'correct_80', 'correct_90'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in evaluation_results:
                writer.writerow(result['row'])
        
        print(f"✅ Evaluation results saved ({len(evaluation_results)} samples)")
    except Exception as e:
        print(f"❌ Error saving evaluation CSV: {e}")


def save_summary_report(stats: Dict, output_file: str):
    """Save summary statistics report to text file."""
    print(f"💾 Saving summary report to: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("EASYOCR BASELINE - EVALUATION SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Overall Statistics
            f.write("📊 DATASET STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Samples Evaluated: {stats['total_samples']}\n\n")
            
            # Accuracy Metrics
            f.write("✅ ACCURACY METRICS (Threshold-based)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Exact Match (100%):        {stats['exact_match_count']:4d}/{stats['total_samples']} "
                    f"({stats['exact_match_pct']:6.2f}%)\n")
            f.write(f"≥70% Match (Levenshtein):  {stats['correct_70_count']:4d}/{stats['total_samples']} "
                    f"({stats['correct_70_pct']:6.2f}%)\n")
            f.write(f"≥80% Match (Levenshtein):  {stats['correct_80_count']:4d}/{stats['total_samples']} "
                    f"({stats['correct_80_pct']:6.2f}%)\n")
            f.write(f"≥90% Match (Levenshtein):  {stats['correct_90_count']:4d}/{stats['total_samples']} "
                    f"({stats['correct_90_pct']:6.2f}%)\n\n")
            
            # Levenshtein Similarity
            f.write("📏 LEVENSHTEIN SIMILARITY (0.0-1.0, higher is better)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Average:                   {stats['avg_levenshtein']:.4f}\n")
            f.write(f"Median:                    {stats['median_levenshtein']:.4f}\n")
            f.write(f"Min:                       {stats['min_levenshtein']:.4f}\n")
            f.write(f"Max:                       {stats['max_levenshtein']:.4f}\n")
            f.write(f"Std Dev:                   {stats['stdev_levenshtein']:.4f}\n\n")
            
            # Word-Level Accuracy
            f.write("🔤 WORD-LEVEL ACCURACY (0.0-1.0, higher is better)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Average:                   {stats['avg_word_accuracy']:.4f}\n")
            f.write(f"Median:                    {stats['median_word_accuracy']:.4f}\n")
            f.write(f"Min:                       {stats['min_word_accuracy']:.4f}\n")
            f.write(f"Max:                       {stats['max_word_accuracy']:.4f}\n")
            f.write(f"Std Dev:                   {stats['stdev_word_accuracy']:.4f}\n\n")
            
            # Character Error Rate
            f.write("❌ CHARACTER ERROR RATE - CER (0.0-1.0, lower is better)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Average:                   {stats['avg_cer']:.4f}\n")
            f.write(f"Median:                    {stats['median_cer']:.4f}\n")
            f.write(f"Min:                       {stats['min_cer']:.4f}\n")
            f.write(f"Max:                       {stats['max_cer']:.4f}\n")
            f.write(f"Std Dev:                   {stats['stdev_cer']:.4f}\n\n")
            
            # Key Observations
            f.write("🔍 KEY OBSERVATIONS & INSIGHTS\n")
            f.write("-" * 70 + "\n")
            
            observations = generate_observations(stats)
            for i, obs in enumerate(observations, 1):
                f.write(f"{i}. {obs}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("📁 Output Files Generated:\n")
            f.write(f"   - {EVALUATION_FILE}\n")
            f.write(f"   - {SUMMARY_FILE}\n")
            f.write("="*70 + "\n")
        
        print(f"✅ Summary report saved")
    except Exception as e:
        print(f"❌ Error saving summary report: {e}")


def generate_observations(stats: Dict) -> List[str]:
    """
    Generate key observations about model performance.
    """
    observations = []
    
    # Accuracy observation
    if stats['correct_70_pct'] >= 90:
        observations.append(
            f"Strong baseline performance: {stats['correct_70_pct']:.1f}% of samples have ≥70% match. "
            "EasyOCR provides a solid foundation for handwritten text recognition."
        )
    elif stats['correct_70_pct'] >= 70:
        observations.append(
            f"Moderate performance: {stats['correct_70_pct']:.1f}% of samples achieve ≥70% similarity. "
            "Preprocessing and model improvements could help."
        )
    else:
        observations.append(
            f"Lower accuracy: Only {stats['correct_70_pct']:.1f}% of samples have ≥70% match. "
            "Significant model improvements or preprocessing optimization may be needed."
        )
    
    # Consistency observation
    if stats['stdev_levenshtein'] < 0.15:
        observations.append(
            f"Consistent predictions: Low standard deviation ({stats['stdev_levenshtein']:.4f}) "
            "indicates stable model behavior across samples."
        )
    else:
        observations.append(
            f"Variable performance: High standard deviation ({stats['stdev_levenshtein']:.4f}) "
            "suggests model performs differently on certain types of samples."
        )
    
    # Error rate observation
    if stats['avg_cer'] < 0.2:
        observations.append(
            f"Low character error rate: Average CER of {stats['avg_cer']:.4f} indicates "
            "most character-level predictions are accurate."
        )
    elif stats['avg_cer'] < 0.5:
        observations.append(
            f"Moderate CER: Average CER of {stats['avg_cer']:.4f} suggests room for improvement "
            "at character and word levels."
        )
    else:
        observations.append(
            f"High CER: Average CER of {stats['avg_cer']:.4f} indicates significant character-level errors. "
            "Investigation of failure modes recommended."
        )
    
    # Word accuracy observation
    if stats['avg_word_accuracy'] >= stats['avg_levenshtein'] + 0.1:
        observations.append(
            f"Word-level accuracy ({stats['avg_word_accuracy']:.4f}) is substantially higher than "
            f"character-level similarity ({stats['avg_levenshtein']:.4f}), suggesting errors are concentrated "
            "in specific words rather than scattered."
        )
    
    # Exact match observation
    if stats['exact_match_pct'] > 10:
        observations.append(
            f"High exact matches: {stats['exact_match_count']} samples ({stats['exact_match_pct']:.1f}%) "
            "are perfectly recognized. This indicates EasyOCR handles many straightforward cases well."
        )
    
    # Range observation
    if stats['max_levenshtein'] - stats['min_levenshtein'] > 0.7:
        observations.append(
            f"Wide performance range: Similarity scores vary from {stats['min_levenshtein']:.4f} to "
            f"{stats['max_levenshtein']:.4f}, indicating diverse sample difficulty or quality levels."
        )
    
    return observations


def print_summary_console(stats: Dict):
    """Print summary statistics to console."""
    print("\n" + "="*70)
    print("EASYOCR BASELINE - EVALUATION RESULTS")
    print("="*70)
    print(f"\n📊 Total Samples: {stats['total_samples']}")
    print(f"\n✅ Accuracy (Levenshtein-based):")
    print(f"   Exact Match:    {stats['exact_match_pct']:6.2f}% ({stats['exact_match_count']})")
    print(f"   ≥70% Match:     {stats['correct_70_pct']:6.2f}% ({stats['correct_70_count']})")
    print(f"   ≥80% Match:     {stats['correct_80_pct']:6.2f}% ({stats['correct_80_count']})")
    print(f"   ≥90% Match:     {stats['correct_90_pct']:6.2f}% ({stats['correct_90_count']})")
    print(f"\n📏 Levenshtein Similarity (0.0-1.0):")
    print(f"   Mean:           {stats['avg_levenshtein']:.4f}")
    print(f"   Median:         {stats['median_levenshtein']:.4f}")
    print(f"   Std Dev:        {stats['stdev_levenshtein']:.4f}")
    print(f"\n🔤 Word-Level Accuracy (0.0-1.0):")
    print(f"   Mean:           {stats['avg_word_accuracy']:.4f}")
    print(f"   Median:         {stats['median_word_accuracy']:.4f}")
    print(f"   Std Dev:        {stats['stdev_word_accuracy']:.4f}")
    print(f"\n❌ Character Error Rate (0.0-1.0):")
    print(f"   Mean:           {stats['avg_cer']:.4f}")
    print(f"   Median:         {stats['median_cer']:.4f}")
    print(f"   Std Dev:        {stats['stdev_cer']:.4f}")
    print("\n" + "="*70)


def main():
    """Main evaluation pipeline."""
    print("🚀 Starting EasyOCR Evaluation Pipeline")
    print("="*70)
    
    # Load predictions
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"❌ Error: Predictions file not found: {PREDICTIONS_FILE}")
        print(f"\n   Please run the following command first:")
        print(f"   python src/Easyocr.py")
        sys.exit(1)
    
    predictions = load_predictions(PREDICTIONS_FILE)
    
    # Evaluate
    evaluation_results = evaluate_predictions(predictions)
    
    # Compute statistics
    stats = compute_aggregate_stats(evaluation_results)
    
    # Save results
    save_evaluation_csv(evaluation_results, EVALUATION_FILE)
    save_summary_report(stats, SUMMARY_FILE)
    
    # Print to console
    print_summary_console(stats)
    
    print(f"\n✅ Evaluation complete!")
    print(f"\n📁 Output files:")
    print(f"   📊 Detailed results: {EVALUATION_FILE}")
    print(f"   📈 Summary report:   {SUMMARY_FILE}")
    print(f"\n💡 Next step: Run Tesseract.py and Trocr.py, then compare results")


if __name__ == "__main__":
    main()
