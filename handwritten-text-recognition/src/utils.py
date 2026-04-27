"""
Utility Functions for Handwritten Text Recognition Project
Provides text normalization, similarity metrics, and evaluation helpers.

Author: Shared utility module
"""

import re
from typing import Tuple, List, Dict
import difflib

try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False


def normalize_text(text: str) -> str:
    """
    Normalize text for fair comparison:
    - Convert to lowercase
    - Strip leading/trailing whitespace
    - Remove extra whitespace
    - Normalize punctuation
    
    Args:
        text: Input text string
        
    Returns:
        Normalized text string
    """
    text = text.lower().strip()
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text


def calculate_levenshtein_ratio(predicted: str, ground_truth: str) -> float:
    """
    Calculate Levenshtein similarity ratio (0.0 to 1.0).
    Uses python-Levenshtein if available for efficiency, else difflib.
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        Similarity ratio (0.0 = completely different, 1.0 = identical)
    """
    predicted = normalize_text(predicted)
    ground_truth = normalize_text(ground_truth)
    
    if predicted == ground_truth:
        return 1.0
    
    if HAS_LEVENSHTEIN:
        return Levenshtein.ratio(predicted, ground_truth)
    else:
        # Fallback to difflib
        return difflib.SequenceMatcher(None, predicted, ground_truth).ratio()


def calculate_word_accuracy(predicted: str, ground_truth: str) -> float:
    """
    Calculate word-level accuracy.
    Splits text into words and computes matching ratio.
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        Word accuracy ratio (0.0 to 1.0)
    """
    predicted_words = normalize_text(predicted).split()
    ground_truth_words = normalize_text(ground_truth).split()
    
    if len(ground_truth_words) == 0:
        return 1.0 if len(predicted_words) == 0 else 0.0
    
    # Use SequenceMatcher for word-level comparison
    matcher = difflib.SequenceMatcher(None, predicted_words, ground_truth_words)
    matching_blocks = sum(block.size for block in matcher.get_matching_blocks())
    
    return matching_blocks / len(ground_truth_words)


def calculate_character_error_rate(predicted: str, ground_truth: str) -> float:
    """
    Calculate Character Error Rate (CER).
    CER = (I + D + S) / N where I=insertions, D=deletions, S=substitutions, N=total ground truth chars
    Returns error rate (lower is better). 0.0 = perfect, can be > 1.0 for very different texts.
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        Character Error Rate
    """
    predicted = normalize_text(predicted)
    ground_truth = normalize_text(ground_truth)
    
    if len(ground_truth) == 0:
        return 0.0 if len(predicted) == 0 else 1.0
    
    if HAS_LEVENSHTEIN:
        distance = Levenshtein.distance(predicted, ground_truth)
    else:
        # Calculate edit distance using SequenceMatcher
        matcher = difflib.SequenceMatcher(None, predicted, ground_truth)
        distance = len(predicted) + len(ground_truth) - 2 * sum(
            block.size for block in matcher.get_matching_blocks()
        )
    
    return distance / len(ground_truth)


def is_correct_match(predicted: str, ground_truth: str, threshold: float = 0.7) -> bool:
    """
    Determine if prediction is correct based on similarity threshold.
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        threshold: Similarity threshold (default 0.7 = 70%)
        
    Returns:
        True if similarity >= threshold, False otherwise
    """
    similarity = calculate_levenshtein_ratio(predicted, ground_truth)
    return similarity >= threshold


def get_case_sensitivity_errors(predicted: str, ground_truth: str) -> int:
    """
    Count characters that differ only in case.
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        Number of case-only mismatches
    """
    if len(predicted) != len(ground_truth):
        return 0
    
    count = 0
    for p, g in zip(predicted, ground_truth):
        if p.lower() == g.lower() and p != g:
            count += 1
    
    return count


def get_detailed_metrics(predicted: str, ground_truth: str) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for a single prediction.
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        Dictionary with various metrics
    """
    return {
        'levenshtein_ratio': calculate_levenshtein_ratio(predicted, ground_truth),
        'word_accuracy': calculate_word_accuracy(predicted, ground_truth),
        'character_error_rate': calculate_character_error_rate(predicted, ground_truth),
        'exact_match': normalize_text(predicted) == normalize_text(ground_truth),
        'is_correct_70': is_correct_match(predicted, ground_truth, 0.7),
        'is_correct_80': is_correct_match(predicted, ground_truth, 0.8),
        'is_correct_90': is_correct_match(predicted, ground_truth, 0.9),
    }


def print_comparison(predicted: str, ground_truth: str, filename: str = "") -> None:
    """
    Pretty print comparison between predicted and ground truth text.
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        filename: Optional filename to print
    """
    pred_norm = normalize_text(predicted)
    truth_norm = normalize_text(ground_truth)
    metrics = get_detailed_metrics(predicted, ground_truth)
    
    print(f"\n📄 Filename: {filename}")
    print(f"   Ground Truth: {truth_norm}")
    print(f"   Predicted:    {pred_norm}")
    print(f"   Levenshtein:  {metrics['levenshtein_ratio']:.3f}")
    print(f"   Word Acc:     {metrics['word_accuracy']:.3f}")
    print(f"   CER:          {metrics['character_error_rate']:.3f}")
    print(f"   Exact Match:  {metrics['exact_match']}")
