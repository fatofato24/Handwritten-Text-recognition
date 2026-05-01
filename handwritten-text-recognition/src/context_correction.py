from textblob import TextBlob
from wordfreq import zipf_frequency
import re

def clean_text(text):
    # Remove unwanted characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.lower()


def is_valid_word(word):
    # Check if it's a common English word
    return zipf_frequency(word, 'en') > 3


def correct_word(word):
    blob = TextBlob(word)
    corrected = str(blob.correct())

    # Accept only meaningful corrections
    if is_valid_word(corrected):
        return corrected
    return word


def correct_text(text):
    text = clean_text(text)
    words = text.split()
    corrected_words = [correct_word(w) for w in words]
    return " ".join(corrected_words)