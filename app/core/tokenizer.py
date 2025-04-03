"""Abstract base class for sentence tokenizers."""

import abc
import time
from typing import Dict, List, Any, Optional, Tuple
import statistics


class SentenceTokenizer(abc.ABC):
    """Abstract base class for sentence tokenizers.
    
    This interface defines a standard way to interact with different sentence
    boundary detection algorithms. Implementations should handle the specifics
    of each library while conforming to this common interface.
    """

    # Standard two-sentence example for warmup and quick testing
    WARMUP_TEXT = "This is a simple first sentence. And here is the second sentence."

    def __init__(self, name: str):
        """Initialize the tokenizer.
        
        Args:
            name: Name identifier for the tokenizer
        """
        self.name = name
        self._is_initialized = False
        self._initialization_time = 0.0
        self._initialization_options = {}
        
    @abc.abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the underlying tokenizer with any required setup.
        
        This method should handle any one-time setup required before tokenization,
        such as loading models, resources, or configuring options.
        
        Args:
            **kwargs: Library-specific initialization options
        """
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if the tokenizer is initialized.
        
        Returns:
            True if the tokenizer is initialized, False otherwise
        """
        return self._is_initialized
    
    @property
    def initialization_time(self) -> float:
        """Get the time taken to initialize the tokenizer.
        
        Returns:
            Time in seconds taken to initialize the tokenizer
        """
        return self._initialization_time
    
    @property
    def initialization_options(self) -> Dict[str, Any]:
        """Get the options used to initialize the tokenizer.
        
        Returns:
            Dictionary of initialization options
        """
        return self._initialization_options.copy()
    
    def warmup(self) -> List[str]:
        """Perform a warmup tokenization on a simple two-sentence example.
        
        This method is useful to ensure that all dependencies are loaded
        and any one-time initializations are complete before benchmarking.
        It can also be used for quick testing of tokenizer functionality.
        
        Returns:
            List of tokenized sentences from the warmup text
        """
        if not self.is_initialized:
            raise RuntimeError("Tokenizer must be initialized before warmup.")
            
        # Run tokenization on the standard example
        sentences = self.tokenize(self.WARMUP_TEXT)
        
        # Verify we got at least one sentence
        if not sentences:
            raise RuntimeError(f"Warmup failed: Tokenizer {self.name} returned no sentences.")
            
        # Return the sentences for potential testing
        return sentences
    
    @abc.abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into sentences.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of sentence strings
        """
        pass
    
    def get_spans(self, text: str) -> List[Tuple[int, int]]:
        """Get the character spans for each sentence.
        
        This is a derived method that uses tokenize() to determine sentence boundaries
        and then calculates the spans. Implementations can override this for more
        efficient or accurate span detection.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of (start, end) tuples representing sentence spans
        """
        if not self.is_initialized:
            raise RuntimeError("Tokenizer must be initialized before use.")
        
        # Use tokenize to get sentences
        sentences = self.tokenize(text)
        
        # Calculate spans
        spans = []
        start = 0
        
        for sentence in sentences:
            # Find the sentence in the text starting from the current position
            sentence_start = text.find(sentence, start)
            if sentence_start == -1:
                # This should not happen with correct tokenization
                continue
                
            sentence_end = sentence_start + len(sentence)
            spans.append((sentence_start, sentence_end))
            
            # Update start position for next search
            start = sentence_end
        
        return spans
    
    def __str__(self) -> str:
        """Get a string representation of the tokenizer.
        
        Returns:
            String representation of the tokenizer
        """
        status = "initialized" if self.is_initialized else "not initialized"
        return f"{self.name} ({status})"