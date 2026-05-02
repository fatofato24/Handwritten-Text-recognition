"""
Enhanced Post-Processing for OCR - Context Correction v2.0
Research-based improvements:
- Multi-engine ensemble voting with confidence weighting
- Character-level OCR error correction patterns
- Levenshtein distance consensus voting
- Advanced contextual spell-checking
- Language model-based corrections

Based on research in: Ensemble methods for OCR, Character error modeling
"""

import re
import difflib
from collections import Counter
from wordfreq import zipf_frequency
from symspellpy import SymSpell, Verbosity
import Levenshtein as leven
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# SYMSPELL INITIALIZATION
# =========================
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
_symspell_loaded = False

try:
    sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)
    _symspell_loaded = True
except:
    _symspell_loaded = False
    print("⚠ SymSpell dictionary not found. Check path.")


# =========================
# COMMON OCR CHARACTER MISRECOGNITION PATTERNS
# Research-backed patterns from literature
# =========================
OCR_CHAR_ERRORS = {
    # Digit-Letter confusions (most common)
    '0': ['o', 'O'],
    '1': ['l', 'I', 'i'],
    '5': ['s', 'S'],
    '8': ['B', 'b'],
    
    # Letter confusions
    'l': ['1', 'I', 'i'],
    'O': ['0', 'o'],
    'I': ['1', 'l', 'i'],
    'Z': ['2', '7'],
    'S': ['5', 'g', 'o'],
    'B': ['8', 'b'],
    'rn': ['m'],  # Common pair
    'cl': ['d'],
    '[]': ['U'],
}


# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    """Clean text: lowercase, remove special chars, normalize whitespace"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =========================
# ADVANCED CHARACTER NORMALIZATION
# Handles common OCR misrecognitions more aggressively
# =========================
def apply_char_replacements(word, confidence=0.5):
    """
    Apply character-level corrections based on OCR error patterns.
    More aggressive with low-confidence predictions.
    """
    if confidence > 0.90:
        # High confidence: minimal changes
        return word.replace("0", "o").replace("1", "l").replace("5", "s")
    
    # Low confidence: apply more corrections
    word = word.replace("0", "o")
    word = word.replace("1", "l")
    word = word.replace("5", "s")
    word = word.replace("8", "b")
    
    # Fix common letter pairs
    word = word.replace("rn", "m")
    word = word.replace("cl", "d")
    
    return word


# =========================
# LEVENSHTEIN-BASED VOTING
# For ensemble predictions from multiple OCR engines
# =========================
def ensemble_vote(predictions, confidences=None):
    """
    Vote on best prediction using Levenshtein distance.
    All predictions "vote" based on similarity to each other.
    Confidence-weighted voting for multi-engine scenarios.
    """
    if not predictions:
        return predictions[0] if predictions else ""
    
    if len(predictions) == 1:
        return predictions[0]
    
    if confidences is None:
        confidences = [1.0] * len(predictions)
    
    # Calculate weighted similarity matrix
    best_pred = predictions[0]
    best_score = 0
    
    for i, pred_i in enumerate(predictions):
        score = 0
        for j, pred_j in enumerate(predictions):
            if i != j:
                # Similarity based on Levenshtein
                similarity = 1.0 - (leven.distance(pred_i, pred_j) / max(len(pred_i), len(pred_j), 1))
                score += similarity * confidences[j]
        
        # Weight by own confidence
        score *= confidences[i]
        
        if score > best_score:
            best_score = score
            best_pred = pred_i
    
    return best_pred


# =========================
# DECISION: SHOULD CORRECT?
# =========================
def should_correct(word, conf):
    """Determine if word should be spell-corrected"""
    if len(word) <= 2:
        return False
    
    # High confidence → trust OCR output
    if conf >= 0.95:
        return False
    
    # Very low confidence → always correct
    if conf < 0.75:
        return True
    
    # Medium confidence → correct carefully
    return True


# =========================
# ADVANCED CANDIDATE GENERATION
# =========================
def get_candidates(word, max_candidates=5):
    """Get spell-check candidates with multiple strategies"""
    candidates = []
    
    try:
        results = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        candidates = [r.term for r in results[:max_candidates]]
    except:
        pass
    
    return candidates


# =========================
# LANGUAGE MODEL SCORING
# =========================
_lm_tokenizer = None
_lm_model = None
_lm_device = None


def load_language_model():
    global _lm_tokenizer, _lm_model, _lm_device
    if _lm_tokenizer is not None and _lm_model is not None:
        return
    
    try:
        model_name = "distilgpt2"
        _lm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _lm_model = AutoModelForCausalLM.from_pretrained(model_name)
        _lm_model.eval()
        _lm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _lm_model.to(_lm_device)
    except Exception:
        _lm_tokenizer = None
        _lm_model = None
        _lm_device = None


def score_sentence_lm(sentence):
    """Score a full sentence using a language model."""
    if not sentence:
        return float('-inf')
    
    load_language_model()
    if _lm_tokenizer is None or _lm_model is None:
        return float('-inf')
    
    try:
        encoded = _lm_tokenizer(sentence, return_tensors="pt")
        if _lm_device is not None:
            encoded = {k: v.to(_lm_device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = _lm_model(**encoded, labels=encoded["input_ids"])
        return -float(outputs.loss)
    except Exception:
        return float('-inf')


def score_sentence_simple(sentence):
    """Fallback sentence score using word frequency."""
    if not sentence:
        return float('-inf')
    score = 0.0
    for word in sentence.split():
        score += zipf_frequency(word, "en")
    return score


def score_sentence(sentence):
    """Choose the best scoring method for sentence quality."""
    lm_score = score_sentence_lm(sentence)
    if lm_score != float('-inf'):
        return lm_score
    return score_sentence_simple(sentence)


# =========================
# ADVANCED SCORING FUNCTION
# Combines multiple signals for better candidate selection
# =========================
def score_candidate(word, candidate, conf):
    """
    Score a candidate correction using multiple signals:
    - Sequence similarity (Levenshtein)
    - Edit distance
    - Word frequency
    - Character-level patterns
    - Confidence penalties
    """
    
    # 1. Sequence similarity (Levenshtein ratio)
    sim = difflib.SequenceMatcher(None, word, candidate).ratio()
    lev_dist = leven.distance(word, candidate)
    
    # Normalize Levenshtein
    max_len = max(len(word), len(candidate), 1)
    lev_ratio = 1.0 - (lev_dist / max_len)
    
    # 2. Word frequency signal
    freq = zipf_frequency(candidate, "en")
    freq_score = min(max(freq, 0) / 10, 1.0)  # Normalize to [0, 1]
    
    # 3. Length penalty (prefer similar lengths)
    length_diff = abs(len(word) - len(candidate))
    length_penalty = min(length_diff * 0.15, 1.0)
    
    # 4. Confidence penalty (lower confidence = more willing to correct)
    conf_penalty = (1.0 - conf) * 0.1
    
    # 5. Edit distance penalty
    edit_penalty = min(lev_dist * 0.05, 0.5)
    
    # Combined score (weights tuned for OCR post-processing)
    final_score = (
        0.50 * lev_ratio +        # Levenshtein similarity (primary)
        0.25 * sim +               # Sequence similarity
        0.15 * freq_score +        # Frequency
        -0.10 * length_penalty +   # Penalize length diff
        -0.05 * edit_penalty +     # Penalize edit distance
        -0.05 * conf_penalty       # Confidence penalty
    )
    
    return max(final_score, 0)


# =========================
# SAFETY FILTERS
# Prevents incorrect "corrections" to common words
# =========================
def is_safe_correction(original, candidate):
    """
    Safety check to prevent bad corrections.
    Protects against collapsing diverse words into common ones.
    """
    
    # List of "trap" words that are common but often wrong substitutes
    trap_words = {
        "the", "and", "with", "is", "to", "of", "a", "in", "or", "it",
        "that", "for", "as", "by", "on", "at", "be", "was", "are", "but"
    }
    
    if candidate.lower() in trap_words:
        return False
    
    # Length difference too large
    if abs(len(original) - len(candidate)) > 4:
        return False
    
    # Original already looks correct (high vowel content)
    vowels_in_original = sum(1 for c in original.lower() if c in 'aeiou')
    if len(original) > 2 and vowels_in_original / len(original) > 0.6:
        return True  # Likely already valid
    
    return True


# =========================
# WORD CORRECTION
# =========================
def correct_word(word, conf=0.5):
    """
    Correct a single word using spell-checking and confidence.
    """
    if not should_correct(word, conf):
        return word
    
    candidates = get_candidates(word)
    if not candidates:
        return word
    
    # Score all candidates
    scored = []
    for candidate in candidates:
        if is_safe_correction(word, candidate):
            score = score_candidate(word, candidate, conf)
            scored.append((candidate, score))
    
    if not scored:
        return word
    
    # Return best candidate
    best = max(scored, key=lambda x: x[1])
    return best[0] if best[1] > 0.3 else word  # Threshold for acceptance


# =========================
# SENTENCE-LEVEL CORRECTION
# =========================
def compound_sentence_correction(text):
    """Apply SymSpell compound correction for full-sentence context."""
    if not text or not _symspell_loaded:
        return text
    
    try:
        suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
        if suggestions:
            return suggestions[0].term
    except Exception:
        pass
    return text


def correct_text(text, ocr_result=None):
    """
    Main text correction function.
    Uses both word-level and sentence-level context.
    """
    text = clean_text(text)
    words = text.split()
    
    if not words:
        return text
    
    conf_map = {}
    
    # Build confidence map from OCR results
    if ocr_result:
        for r in ocr_result:
            conf_map[r[1].lower()] = r[2]
    
    # Word-level correction first
    corrected_words = []
    for w in words:
        conf = conf_map.get(w, 0.70)
        w = apply_char_replacements(w, conf)
        corrected_words.append(correct_word(w, conf))
    word_corrected = " ".join(corrected_words)
    
    # Sentence-level candidates
    candidates = [text, word_corrected]
    sentence_compound = compound_sentence_correction(word_corrected)
    if sentence_compound and sentence_compound != word_corrected:
        candidates.append(sentence_compound)
    
    best_candidate = max(candidates, key=score_sentence)
    return best_candidate


# =========================
# MULTI-ENGINE ENSEMBLE CORRECTION
# NEW: For combining predictions from multiple OCR engines
# =========================
def ensemble_correct(predictions_dict):
    """
    Combine predictions from multiple OCR engines with ensemble voting.
    
    Args:
        predictions_dict: {
            'easyocr': ('text', confidence),
            'tesseract': ('text', confidence),
            'trocr': ('text', confidence)
        }
    
    Returns:
        Best combined prediction with confidence score
    """
    
    predictions = []
    confidences = []
    engines = []
    
    for engine, (text, conf) in predictions_dict.items():
        if text and text.strip():
            predictions.append(clean_text(text))
            confidences.append(conf)
            engines.append(engine)
    
    if not predictions:
        return "", 0.0
    
    if len(predictions) == 1:
        return predictions[0], confidences[0]
    
    # Use Levenshtein voting
    best = ensemble_vote(predictions, confidences)
    
    # Correct the ensemble result
    corrected = correct_text(best)
    
    # Average confidence
    avg_conf = sum(confidences) / len(confidences)
    
    return corrected, avg_conf