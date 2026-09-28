"""
TravelMate — NLP Preprocessing Module
======================================
Reusable NLP preprocessing pipeline for travel queries and documents.
Includes text normalization, travel-aware tokenization, stopword filtering
(preserving travel-critical prepositions and operators), lemmatization,
and stemming.
"""

import re
import string
from typing import List, Dict, Any, Optional

# Lazy loading of spacy / nltk to ensure fast module import
_SPACY_NLP = None
_STEMMER = None

# Custom Travel Stopwords list:
# Standard stopwords MINUS critical travel interrogatives, relations, and qualifiers
TRAVEL_PRESERVED_WORDS = {
    "not", "no", "never", "without", "with", "near", "around", "from", "to",
    "between", "under", "above", "over", "how", "what", "where", "which", "why",
    "in", "on", "at", "by", "for", "few", "more", "less", "top", "best"
}

STANDARD_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "is", "am", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "about", "against", "into", "through", "during", "before",
    "after", "then", "once", "here", "there", "when", "all", "any", "both",
    "each", "other", "some", "such", "only", "own", "same", "so", "than",
    "too", "very", "can", "will", "just", "don", "should", "now"
}

TRAVEL_STOPWORDS = STANDARD_STOPWORDS - TRAVEL_PRESERVED_WORDS


def get_spacy_nlp():
    """Lazy loader for spaCy English pipeline."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            import spacy
            _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser"])
        except Exception:
            try:
                import spacy
                _SPACY_NLP = spacy.blank("en")
            except Exception:
                _SPACY_NLP = None
    return _SPACY_NLP


def get_stemmer():
    """Lazy loader for PorterStemmer."""
    global _STEMMER
    if _STEMMER is None:
        try:
            from nltk.stem import PorterStemmer
            _STEMMER = PorterStemmer()
        except Exception:
            _STEMMER = None
    return _STEMMER


def clean_text(text: str) -> str:
    """
    Cleans raw text by normalizing whitespace, expanding common contractions,
    and preserving travel symbols (currency, hyphens, numbers).
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Expand contractions
    contractions = {
        r"can\'t": "can not",
        r"won\'t": "will not",
        r"n\'t": " not",
        r"\'re": " are",
        r"\'s": " is",
        r"\'d": " would",
        r"\'ll": " will",
        r"\'t": " not",
        r"\'ve": " have",
        r"\'m": " am",
    }
    cleaned = text
    for pattern, replacement in contractions.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # 2. Normalize whitespace
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize(text: str, keep_punctuation: bool = False) -> List[str]:
    """
    Tokenizes text while preserving travel tokens like currency ('₹8000', 'rs.500'),
    durations ('3-day', '4d'), and hyphenated compounds.
    """
    if not text:
        return []
    
    cleaned = clean_text(text)
    
    if keep_punctuation:
        # Simple whitespace tokenization with boundary splitting
        tokens = re.findall(r"[\w₹$€]+|[^\w\s]", cleaned)
    else:
        # Match words, currency-prefixed numbers, hyphenated words, alphanumeric numbers
        tokens = re.findall(r"[₹$€]?\d+(?:,\d+)*(?:\.\d+)?k?|\b\w+(?:-\w+)*\b", cleaned)
        
    return [t.strip() for t in tokens if t.strip()]


def remove_stopwords(tokens: List[str], preserve_travel_context: bool = True) -> List[str]:
    """
    Removes common stopwords. If preserve_travel_context is True,
    critical navigation and constraint words (not, with, without, near, under) are retained.
    """
    stop_set = TRAVEL_STOPWORDS if preserve_travel_context else STANDARD_STOPWORDS
    filtered = []
    for token in tokens:
        lower_token = token.lower()
        if lower_token not in stop_set or token.startswith(("₹", "$", "€")) or token.isdigit():
            filtered.append(token)
    return filtered


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Lemmatizes tokens to base forms using spaCy if available,
    with a reliable fallback lookup.
    """
    nlp = get_spacy_nlp()
    if nlp is not None:
        # Feed joined tokens to spaCy
        doc = nlp(" ".join(tokens))
        lemmas = []
        for token in doc:
            lemma = token.lemma_.lower() if token.lemma_ != "-PRON-" else token.text.lower()
            # If lemma is just symbol or empty, retain original
            lemmas.append(lemma if lemma.strip() else token.text)
        return lemmas
    
    # Fallback rule-based lemmatizer
    suffix_rules = [
        ("beaches", "beach"),
        ("churches", "church"),
        ("cities", "city"),
        ("attractions", "attraction"),
        ("destinations", "destination"),
        ("activities", "activity"),
        ("places", "place"),
        ("waterfalls", "waterfall"),
        ("mountains", "mountain"),
        ("lakes", "lake"),
        ("hotels", "hotel"),
        ("bazaars", "bazaar"),
        ("traveling", "travel"),
        ("travelling", "travel"),
        ("visited", "visit"),
        ("visiting", "visit"),
        ("suggesting", "suggest"),
        ("recommending", "recommend"),
    ]
    lemmas = []
    for token in tokens:
        lower = token.lower()
        matched = False
        for sfx, root in suffix_rules:
            if lower == sfx:
                lemmas.append(root)
                matched = True
                break
        if not matched:
            if lower.endswith("ies") and len(lower) > 4:
                lemmas.append(lower[:-3] + "y")
            elif lower.endswith("ing") and len(lower) > 5:
                lemmas.append(lower[:-3])
            elif lower.endswith("ed") and len(lower) > 4:
                lemmas.append(lower[:-2])
            elif lower.endswith("s") and not lower.endswith("ss") and len(lower) > 3:
                lemmas.append(lower[:-1])
            else:
                lemmas.append(lower)
    return lemmas


def stem_tokens(tokens: List[str]) -> List[str]:
    """Applies Porter Stemmer to a list of tokens."""
    stemmer = get_stemmer()
    if stemmer:
        return [stemmer.stem(t) for t in tokens]
    return [t.lower() for t in tokens]


def preprocess_query(query: str, for_classification: bool = False) -> Dict[str, Any]:
    """
    Comprehensive query preprocessing pipeline returning all stages:
    original, cleaned, tokens, tokens without stopwords, lemmas, and normalized text.
    """
    original = str(query or "").strip()
    cleaned = clean_text(original)
    tokens = tokenize(cleaned, keep_punctuation=False)
    tokens_no_stop = remove_stopwords(tokens, preserve_travel_context=True)
    lemmas = lemmatize_tokens(tokens_no_stop)
    stems = stem_tokens(tokens_no_stop)
    
    # Normalized representation ideal for ML / TF-IDF
    normalized_for_model = " ".join(lemmas)
    
    return {
        "original_query": original,
        "cleaned_text": cleaned,
        "tokens": tokens,
        "tokens_without_stopwords": tokens_no_stop,
        "lemmas": lemmas,
        "stems": stems,
        "normalized_text": normalized_for_model
    }
