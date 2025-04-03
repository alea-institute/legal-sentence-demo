"""nupunkt implementation of SentenceTokenizer."""

import time
from typing import List, Dict, Any, Optional

from app.core.tokenizer import SentenceTokenizer
from app.logger import logger


class NupunktTokenizer(SentenceTokenizer):
    """SentenceTokenizer implementation for nupunkt.
    
    nupunkt is a next-generation implementation of the Punkt algorithm
    for sentence boundary detection with zero runtime dependencies.
    """
    
    def __init__(self, name: str = "nupunkt"):
        """Initialize the nupunkt tokenizer.
        
        Args:
            name: Name identifier for the tokenizer
        """
        super().__init__(name)
        self._tokenizer = None
        self._tokenizer_instance = None
        
    def initialize(self, **kwargs) -> None:
        """Initialize the nupunkt tokenizer.
        
        Args:
            **kwargs: Initialization options:
                - model_path: Optional path to a custom trained model
                - parameters: Optional custom parameters
        """
        try:
            import nupunkt
            
            start_time = time.time()
            
            model_path = kwargs.get("model_path")
            parameters = kwargs.get("parameters")
            
            if parameters:
                # If custom parameters are provided, use them
                logger.info("Initializing nupunkt with custom parameters")
                self._tokenizer_instance = nupunkt.PunktSentenceTokenizer(parameters)
                self._tokenizer = self._tokenizer_instance.tokenize
            elif model_path:
                # If a model path is provided, load from it
                logger.info(f"Loading nupunkt model from {model_path}")
                from nupunkt import PunktSentenceTokenizer
                self._tokenizer_instance = PunktSentenceTokenizer.load(model_path)
                self._tokenizer = self._tokenizer_instance.tokenize
            else:
                # Load default model once and cache it
                logger.info("Loading default nupunkt model")
                from nupunkt import load_default_model
                self._tokenizer_instance = load_default_model()
                self._tokenizer = self._tokenizer_instance.tokenize
            
            end_time = time.time()
            self._initialization_time = end_time - start_time
            self._initialization_options = kwargs.copy()
            self._is_initialized = True
            logger.info(f"nupunkt tokenizer initialized in {self._initialization_time:.2f}s")
            
        except ImportError as e:
            logger.error(f"nupunkt import error: {str(e)}")
            raise RuntimeError(f"Could not import nupunkt: {str(e)}. Please install with 'pip install nupunkt'.")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into sentences using nupunkt.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of sentence strings
        """
        if not self.is_initialized:
            raise RuntimeError("Tokenizer must be initialized before use.")
        
        try:
            # We now always use the tokenize method directly from a cached instance
            return self._tokenizer(text)
        except Exception as e:
            logger.error(f"nupunkt tokenization error: {str(e)}")
            return []