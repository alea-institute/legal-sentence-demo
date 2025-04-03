"""Tokenizer implementations for various sentence boundary detection libraries."""

from app.tokenizers.nltk import NLTKTokenizer
from app.tokenizers.spacy import SpacyTokenizer
from app.tokenizers.pysbd import PySBDTokenizer
from app.tokenizers.nupunkt import NupunktTokenizer
from app.tokenizers.charboundary import CharBoundaryTokenizer

# Dictionary of available tokenizers for easy access
TOKENIZERS = {
    "nltk": NLTKTokenizer,
    "spacy": SpacyTokenizer,
    "pysbd": PySBDTokenizer,
    "nupunkt": NupunktTokenizer,
    "charboundary": CharBoundaryTokenizer,
}

# Configuration for UI display using custom theme colors
TOKENIZER_CONFIG = {
    "nltk": {"display_name": "NLTK Punkt", "color": "pink"},
    "spacy": {"display_name": "spaCy", "color": "green"},
    "pysbd": {"display_name": "PySBD", "color": "yellow"},
    "nupunkt": {"display_name": "nupunkt", "color": "brown"},
    "charboundary": {"display_name": "CharBoundary", "color": "lime"},
}