"""CharBoundary implementation of SentenceTokenizer."""

import time
import os
from typing import List, Dict, Any, Optional, Tuple

from app.core.tokenizer import SentenceTokenizer
from app.logger import logger


class CharBoundaryTokenizer(SentenceTokenizer):
    """SentenceTokenizer implementation for CharBoundary.
    
    CharBoundary is a machine learning-based sentence boundary detector
    optimized for legal and scientific texts.
    """
    
    def __init__(self, name: str = "charboundary"):
        """Initialize the CharBoundary tokenizer.
        
        Args:
            name: Name identifier for the tokenizer
        """
        super().__init__(name)
        self._model = None
        
    def initialize(self, **kwargs) -> None:
        """Initialize the CharBoundary tokenizer.
        
        Args:
            **kwargs: Initialization options:
                - size: Model size to use ('small', 'medium', 'large', default: 'small')
                - model_path: Optional path to a custom model file
        """
        try:
            import charboundary
            
            start_time = time.time()
            
            # Use large model by default
            size = kwargs.get("size", "large")
            model_path = kwargs.get("model_path")
            
            logger.info(f"Initializing CharBoundary with size={size}")
            
            if model_path and os.path.exists(model_path):
                # Load a custom model from path if feature is added in future versions
                logger.error("Custom model loading not supported in current charboundary version")
                raise ValueError("Custom model loading not supported in current charboundary version")
            else:
                # Load a pre-trained ONNX model based on size
                if size == "small":
                    logger.info("Loading small CharBoundary model")
                    self._model = charboundary.get_small_onnx_segmenter()
                elif size == "large":
                    logger.info("Loading large CharBoundary model")
                    self._model = charboundary.get_large_onnx_segmenter()
                else:  # medium or default
                    logger.info("Loading medium CharBoundary model")
                    self._model = charboundary.get_medium_onnx_segmenter()
            
            end_time = time.time()
            self._initialization_time = end_time - start_time
            self._initialization_options = kwargs.copy()
            self._is_initialized = True
            logger.info(f"CharBoundary tokenizer initialized in {self._initialization_time:.2f}s")
            
        except ImportError as e:
            logger.error(f"CharBoundary import error: {str(e)}")
            raise RuntimeError(f"Could not import CharBoundary: {str(e)}. Please install with 'pip install charboundary[onnx]'.")
    
    def tokenize(self, text: str, threshold: Optional[float] = None) -> List[str]:
        """Tokenize text into sentences using CharBoundary.
        
        Args:
            text: Input text to tokenize
            threshold: Optional probability threshold (0.0-1.0) for sentence boundaries
                      Higher values are more conservative (fewer sentence breaks)
            
        Returns:
            List of sentence strings
        """
        if not self.is_initialized:
            raise RuntimeError("Tokenizer must be initialized before use.")
        
        # Log a hash of the input text for debugging
        import hashlib
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        logger.info(f"CharBoundary tokenizing text with hash: {text_hash}, length: {len(text)}")
        
        try:
            # Using the preferred segment_to_sentences method from README
            try:
                # First try the preferred method from documentation
                # Pass threshold parameter if provided
                if threshold is not None:
                    logger.info(f"Using CharBoundary with threshold: {threshold}")
                    try:
                        sentences = self._model.segment_to_sentences(text, threshold=threshold)
                    except TypeError:
                        # If the library doesn't support threshold parameter yet
                        logger.warning("CharBoundary doesn't support threshold parameter, using default")
                        sentences = self._model.segment_to_sentences(text)
                else:
                    sentences = self._model.segment_to_sentences(text)
                
                # Log unexpected sentences for debugging
                for i, sentence in enumerate(sentences):
                    if "Brown v. Board of Education" in sentence and sentence not in text:
                        logger.warning(f"CharBoundary produced sentence not in original text: {sentence}")
                        logger.warning(f"Original text contains 'Brown v. Board'? {'Brown v. Board' in text}")
                
                # Validate that sentences are actually from the input text and filter out hallucinations
                valid_sentences = []
                for i, sentence in enumerate(sentences):
                    stripped_sentence = sentence.strip()
                    
                    # Skip empty sentences
                    if not stripped_sentence:
                        continue
                        
                    # Check if the sentence is actually in the input text
                    if stripped_sentence in text:
                        valid_sentences.append(sentence)
                    else:
                        logger.warning(f"Filtering out hallucinated sentence {i} from CharBoundary not found in original text: '{sentence}'")
                        if len(sentence) > 50:
                            # Log just the start of long sentences
                            logger.warning(f"Hallucinated sentence start: '{sentence[:50]}...'")
                
                logger.info(f"CharBoundary originally returned {len(sentences)} sentences, {len(valid_sentences)} were valid")
                return valid_sentences
            except AttributeError:
                # Fallback to segment_text if segment_to_sentences is not available
                # This handles any version differences
                logger.warning("segment_to_sentences not available, falling back to segment_text")
                segmented = self._model.segment_text(text)
                
                # Check if the result is a string with <|sentence|> markers
                if isinstance(segmented, str):
                    # Split by the sentence marker and filter out any empty strings
                    sentences = [s for s in segmented.split("<|sentence|>") if s]
                    
                    # Validate that sentences are actually from the input text and filter out hallucinations
                    valid_sentences = []
                    for i, sentence in enumerate(sentences):
                        stripped_sentence = sentence.strip()
                        
                        # Skip empty sentences
                        if not stripped_sentence:
                            continue
                            
                        # Check if the sentence is actually in the input text
                        if stripped_sentence in text:
                            valid_sentences.append(sentence)
                        else:
                            logger.warning(f"Filtering out hallucinated sentence {i} from segmented text not found in original: '{sentence}'")
                    
                    logger.info(f"CharBoundary segment_text originally returned {len(sentences)} sentences, {len(valid_sentences)} were valid")
                    return valid_sentences
                elif isinstance(segmented, list):
                    # Handle possible list return format
                    return segmented
                else:
                    # Fallback to original text
                    logger.warning(f"Unexpected CharBoundary output type: {type(segmented)}")
                    return [text]
        except Exception as e:
            logger.error(f"CharBoundary tokenization error: {str(e)}")
            return [text]
        
    def get_spans(self, text: str) -> List[Tuple[int, int]]:
        """Get the character spans for each sentence using CharBoundary.
        
        For CharBoundary, we need to determine spans from the tokenization results
        since the API doesn't directly provide spans.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of (start, end) tuples representing sentence spans
        """
        if not self.is_initialized:
            raise RuntimeError("Tokenizer must be initialized before use.")
        
        try:
            # Use the base implementation to calculate spans from sentences
            return super().get_spans(text)
        except Exception as e:
            logger.error(f"CharBoundary span detection error: {str(e)}")
            return []