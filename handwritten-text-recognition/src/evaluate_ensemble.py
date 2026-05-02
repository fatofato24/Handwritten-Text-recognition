"""
Enhanced Evaluation Framework for OCR Post-Processing
Compares EasyOCR baseline vs. Ensemble vs. Improved Context Correction

Metrics:
- Character Error Rate (CER)
- Word Accuracy
- Levenshtein Distance
- Exact Match Rate
- Semantic Similarity
"""

import csv
import os
from pathlib import Path
import numpy as np
import Levenshtein as leven
from difflib import SequenceMatcher
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from context_correction import (
    correct_text, apply_char_replacements, clean_text
)

# Configuration
RESULTS_DIR = "results/"
OUTPUT_ANALYSIS = os.path.join(RESULTS_DIR, "analysis_results.csv")

# ===========================
# EVALUATION METRICS
# ===========================

def character_error_rate(reference, hypothesis):
    """Calculate Character Error Rate (CER)"""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    
    distance = leven.distance(reference, hypothesis)
    return distance / len(reference)


def word_error_rate(reference, hypothesis):
    """Calculate Word Error Rate (WER)"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    
    distance = leven.distance(reference, hypothesis)
    return distance / len(reference)


def word_accuracy(reference, hypothesis):
    """Exact word-level accuracy"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) == 0 else 0.0
    
    matches = sum(1 for r, h in zip(ref_words, hyp_words) if r == h)
    return matches / len(ref_words)


def semantic_similarity(reference, hypothesis):
    """Calculate semantic similarity using sequence matching"""
    return SequenceMatcher(None, reference, hypothesis).ratio()


def levenshtein_ratio(reference, hypothesis):
    """Levenshtein ratio (0 = completely different, 1 = identical)"""
    if len(reference) == 0 and len(hypothesis) == 0:
        return 1.0
    
    max_len = max(len(reference), len(hypothesis), 1)
    distance = leven.distance(reference, hypothesis)
    return 1.0 - (distance / max_len)


# ===========================
# ANALYZE PREDICTIONS
# ===========================

def analyze_predictions(csv_file):
    """
    Analyze predictions from a CSV file and calculate all metrics.
    """
    
    results = []
    
    if not os.path.exists(csv_file):
        print(f"⚠️  File not found: {csv_file}")
        return results
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            ground_truth = row.get('ground_truth', '').lower().strip()
            
            # Handle different column names
            if 'predicted_text' in row:
                predicted = row.get('predicted_text', '').lower().strip()
            elif 'ensemble' in row:
                predicted = row.get('ensemble', '').lower().strip()
            else:
                continue
            
            if not ground_truth or not predicted:
                continue
            
            # Calculate metrics
            result = {
                'filename': row.get('filename', ''),
                'ground_truth': ground_truth,
                'predicted': predicted,
                'cer': character_error_rate(ground_truth, predicted),
                'wer': word_error_rate(ground_truth, predicted),
                'word_accuracy': word_accuracy(ground_truth, predicted),
                'semantic_sim': semantic_similarity(ground_truth, predicted),
                'levenshtein_ratio': levenshtein_ratio(ground_truth, predicted),
                'exact_match': 1 if ground_truth == predicted else 0,
            }
            
            results.append(result)
    
    return results


def print_summary_statistics(results, method_name):
    """Print summary statistics for a method"""
    
    if not results:
        print(f"\n⚠️  No results for {method_name}")
        return {}
    
    cer_values = [r['cer'] for r in results]
    wer_values = [r['wer'] for r in results]
    wa_values = [r['word_accuracy'] for r in results]
    sim_values = [r['semantic_sim'] for r in results]
    lev_values = [r['levenshtein_ratio'] for r in results]
    exact_match_count = sum(r['exact_match'] for r in results)
    
    summary = {
        'method': method_name,
        'total_samples': len(results),
        'exact_match_rate': exact_match_count / len(results) if results else 0.0,
        'avg_cer': np.mean(cer_values),
        'avg_wer': np.mean(wer_values),
        'avg_word_accuracy': np.mean(wa_values),
        'avg_semantic_sim': np.mean(sim_values),
        'avg_levenshtein_ratio': np.mean(lev_values),
    }
    
    print(f"\n{'='*60}")
    print(f"📊 {method_name}")
    print(f"{'='*60}")
    print(f"  Total Samples:          {len(results)}")
    print(f"  Exact Match Rate:       {exact_match_count}/{len(results)} ({summary['exact_match_rate']*100:.1f}%)")
    print(f"  Avg Character Error Rate (CER):     {summary['avg_cer']:.4f} (lower is better)")
    print(f"  Avg Word Error Rate (WER):          {summary['avg_wer']:.4f}")
    print(f"  Avg Word Accuracy:      {summary['avg_word_accuracy']:.4f} ({summary['avg_word_accuracy']*100:.1f}%)")
    print(f"  Avg Semantic Similarity: {summary['avg_semantic_sim']:.4f}")
    print(f"  Avg Levenshtein Ratio:   {summary['avg_levenshtein_ratio']:.4f}")
    
    return summary


# ===========================
# MAIN COMPARISON
# ===========================

def run_comprehensive_evaluation():
    """Run comprehensive evaluation comparing all methods"""
    
    print("\n" + "="*60)
    print("🔍 COMPREHENSIVE OCR EVALUATION")
    print("="*60)
    
    methods = {
        'easyocr': os.path.join(RESULTS_DIR, 'easyocr_evaluation.csv'),
        'ensemble': os.path.join(RESULTS_DIR, 'ensemble_predictions.csv'),
    }
    
    all_summaries = []
    
    for method_name, csv_file in methods.items():
        print(f"\n📖 Analyzing {method_name}...")
        results = analyze_predictions(csv_file)
        summary = print_summary_statistics(results, method_name.upper())
        all_summaries.append(summary)
    
    # ===========================
    # COMPARISON & IMPROVEMENT
    # ===========================
    
    if len(all_summaries) >= 2:
        print(f"\n{'='*60}")
        print("📈 IMPROVEMENT ANALYSIS")
        print(f"{'='*60}")
        
        baseline = all_summaries[0]
        improved = all_summaries[1]
        
        # Calculate improvements
        exact_match_improvement = (improved['exact_match_rate'] - baseline['exact_match_rate']) * 100
        cer_improvement = (baseline['avg_cer'] - improved['avg_cer']) / max(baseline['avg_cer'], 0.001) * 100
        wa_improvement = (improved['avg_word_accuracy'] - baseline['avg_word_accuracy']) * 100
        
        print(f"\n  Exact Match Rate Improvement:  {exact_match_improvement:+.1f}%")
        print(f"  Character Error Rate Improvement: {cer_improvement:+.1f}%")
        print(f"  Word Accuracy Improvement:    {wa_improvement:+.1f}%")
        
        if exact_match_improvement > 0:
            print(f"\n  ✅ IMPROVEMENT DETECTED: {exact_match_improvement:.1f}% better exact match rate!")
        elif exact_match_improvement < 0:
            print(f"\n  ⚠️  Performance decreased: {abs(exact_match_improvement):.1f}%")
        else:
            print(f"\n  ➖  No change in performance")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    run_comprehensive_evaluation()
