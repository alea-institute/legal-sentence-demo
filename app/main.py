from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional, Dict, Any
import os
import pathlib
import time
import traceback
import uuid
import json
import base64
import statistics
import re
from contextlib import asynccontextmanager
from collections import defaultdict

# Import custom logger
from app.logger import logger

# Import tokenizer implementations
from app.tokenizers import TOKENIZER_CONFIG, TOKENIZERS

# Constants
MAX_TEXT_LENGTH = 50000  # Maximum allowed text length
SHARED_ANALYSES = {}  # In-memory storage for shared analyses


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application startup and shutdown events.
    Initialize tokenizers during startup to reduce latency on first request.
    """
    # Startup: Initialize tokenizers
    logger.info("Application starting up...")
    
    # Initialize all tokenizers at startup
    try:
        logger.info("Pre-initializing tokenizers...")
        
        # Create tokenizer instances if they don't exist yet
        for name, cls in TOKENIZERS.items():
            if name not in tokenizer_instances:
                tokenizer_instances[name] = cls()
                
        # Initialize all tokenizers concurrently
        for name, tokenizer in tokenizer_instances.items():
            try:
                if not tokenizer.is_initialized:
                    logger.info(f"Initializing tokenizer: {name}")
                    start_time = time.time()
                    # Use large model for CharBoundary
                    if name == "charboundary":
                        # Initialize with default settings in startup
                        tokenizer.initialize(size="large")
                    else:
                        tokenizer.initialize()
                    end_time = time.time()
                    logger.info(f"Tokenizer {name} initialized in {end_time - start_time:.2f}s")
            except Exception as e:
                logger.error(f"Error initializing tokenizer {name}: {str(e)}")
                logger.error(traceback.format_exc())
        
        logger.info("All tokenizers initialized")
    except Exception as e:
        logger.error(f"Error during tokenizer initialization: {str(e)}")
        logger.error(traceback.format_exc())
    
    yield
    
    # Shutdown
    logger.info("Application shutting down...")


app = FastAPI(
    title="Legal Sentence Boundary Detection",
    description="Compare sentence boundary detection algorithms on legal text",
    version="0.1.0",
    lifespan=lifespan
)

# Setup templates and static files
base_dir = pathlib.Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")

# Dictionary to hold tokenizer instances - reset to use new model sizes
tokenizer_instances = {}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request details and timing."""
    start_time = time.time()
    path = request.url.path
    query_params = str(request.query_params)
    
    request_id = str(time.time())
    logger.info(f"[{request_id}] Request started: {request.method} {path} {query_params}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"[{request_id}] Request completed: {request.method} {path} - Status: {response.status_code} - Time: {process_time:.4f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"[{request_id}] Request failed: {request.method} {path} - Error: {str(e)} - Time: {process_time:.4f}s")
        logger.error(traceback.format_exc())
        raise


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page with the form."""
    logger.debug("Rendering index page")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request, 
    text: str = Form(...), 
    tokenizer_names: Optional[List[str]] = Form(None, alias="tokenizers"),
    generate_share_link: Optional[bool] = Form(False),
    charboundary_threshold: Optional[float] = Form(None)
):
    """Process text with selected tokenizers and return visualization."""
    if not text:
        logger.warning("Empty text submitted for analysis")
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Check text length limitation
    if len(text) > MAX_TEXT_LENGTH:
        logger.warning(f"Text exceeds maximum length: {len(text)} > {MAX_TEXT_LENGTH}")
        raise HTTPException(
            status_code=400, 
            detail=f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters. Please submit a shorter text."
        )
    
    if not tokenizer_names:
        logger.info("No tokenizers selected, defaulting to all")
        tokenizer_names = list(TOKENIZER_CONFIG.keys())
    
    logger.info(f"Analyzing text (length: {len(text)}) with tokenizers: {', '.join(tokenizer_names)}")
    
    # Generate share ID if requested
    share_id = None
    if generate_share_link:
        share_id = str(uuid.uuid4())
        logger.info(f"Generated share ID: {share_id}")
    
    # Initialize selected tokenizers if not already initialized
    results = {}
    tokenizer_info = []
    
    for tokenizer_name in tokenizer_names:
        if tokenizer_name in TOKENIZER_CONFIG:
            try:
                # Get or create tokenizer instance
                if tokenizer_name not in tokenizer_instances:
                    logger.debug(f"Creating tokenizer instance: {tokenizer_name}")
                    tokenizer_instances[tokenizer_name] = TOKENIZERS[tokenizer_name]()
                
                tokenizer = tokenizer_instances[tokenizer_name]
                if not tokenizer.is_initialized:
                    logger.debug(f"Initializing tokenizer: {tokenizer_name}")
                    # Initialize tokenizers
                    if tokenizer_name == "charboundary":
                        tokenizer.initialize(size="large")
                    else:
                        tokenizer.initialize()
                
                # Tokenize the text
                start_time = time.time()
                
                # Pass threshold directly to the tokenize method for CharBoundary
                if tokenizer_name == "charboundary" and charboundary_threshold is not None:
                    logger.info(f"Using CharBoundary threshold: {charboundary_threshold}")
                    sentences = tokenizer.tokenize(text, threshold=charboundary_threshold)
                else:
                    sentences = tokenizer.tokenize(text)
                    
                spans = tokenizer.get_spans(text)
                process_time = time.time() - start_time
                
                logger.debug(f"Tokenizer {tokenizer_name} found {len(sentences)} sentences in {process_time:.4f}s")
                
                results[tokenizer_name] = {
                    "sentences": sentences,
                    "spans": spans,
                    "count": len(sentences),
                    "process_time": process_time
                }
                
                # Get model details for display
                model_details = ""
                if tokenizer_name == "charboundary":
                    # Get CharBoundary model size
                    cb_size = tokenizer._initialization_options.get("size", "small")
                    model_details = f"({cb_size})"
                elif tokenizer_name == "spacy":
                    # Get spaCy model name
                    spacy_model = tokenizer._initialization_options.get("model", "en_core_web_sm")
                    # Fix duplicate underscores if present
                    spacy_model = spacy_model.replace("__", "_")
                    model_details = f"({spacy_model})"
                
                # Prepare tokenizer info for template
                tokenizer_info.append({
                    "name": tokenizer_name,
                    "display_name": TOKENIZER_CONFIG[tokenizer_name]["display_name"],
                    "model_details": model_details,
                    "color": TOKENIZER_CONFIG[tokenizer_name]["color"],
                    "sentence_count": len(sentences),
                    "process_time": process_time
                })
            except Exception as e:
                logger.error(f"Error processing with tokenizer {tokenizer_name}: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error processing with tokenizer {tokenizer_name}: {str(e)}"
                )
    
    try:
        # Calculate which tokenizer has the fewest sentences and which is fastest
        min_sentences = min(result_data["count"] for result_data in results.values())
        min_time = min(result_data["process_time"] for result_data in results.values())
        
        # Update tokenizer info with these highlights and add throughput
        for tokenizer_info_item in tokenizer_info:
            name = tokenizer_info_item["name"]
            count = tokenizer_info_item["sentence_count"]
            process_time = tokenizer_info_item["process_time"]
            
            # Calculate throughput in characters per second
            chars_per_second = len(text) / process_time if process_time > 0 else 0
            if chars_per_second >= 1_000_000:
                throughput = f"{chars_per_second / 1_000_000:.2f}M char/s"
            elif chars_per_second >= 1_000:
                throughput = f"{chars_per_second / 1_000:.2f}K char/s"
            else:
                throughput = f"{chars_per_second:.0f} char/s"
                
            tokenizer_info_item["throughput"] = throughput
            tokenizer_info_item["is_fewest_sentences"] = (count == min_sentences)
            tokenizer_info_item["is_fastest"] = (process_time == min_time)
        
        # Prepare data for color-coded inline markers view
        logger.debug("Generating inline markers visualization")
        inline_markers = generate_inline_markers(text, results)
        logger.debug(f"Generated {len(inline_markers)} markers")
        
        # Log first few markers for debugging
        for i, marker in enumerate(inline_markers[:5]):
            logger.debug(f"Marker {i}: tokenizer={marker.get('tokenizer')}, is_boundary={marker.get('is_boundary')}, text={marker.get('text')[:20]}...")
        
        # Prepare data for interactive sentence table view
        logger.debug("Generating sentence table visualization")
        sentence_table = generate_sentence_table(text, results)
        
        logger.info(f"Successfully analyzed text with {len(tokenizer_info)} tokenizers")
        
        # Calculate additional text statistics
        text_stats = calculate_text_statistics(text, results)
        
        # Generate sentence length distribution data
        length_distribution = generate_length_distribution(results)
        
        # Store shared analysis if requested
        if generate_share_link:
            SHARED_ANALYSES[share_id] = {
                "text": text,
                "tokenizer_names": tokenizer_names,
                "timestamp": time.time()
            }
            # Clean up old shared analyses (keep for 24 hours)
            cleanup_old_shared_analyses(24 * 60 * 60)
        
        return templates.TemplateResponse(
            "results.html", 
            {
                "request": request,
                "text": text,
                "tokenizers": tokenizer_info,
                "inline_markers": inline_markers,
                "sentence_table": sentence_table,
                "text_stats": text_stats,
                "length_distribution": length_distribution,
                "share_id": share_id,
                "base_url": str(request.base_url).rstrip('/')
            }
        )
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")


def generate_inline_markers(text: str, results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate data for the color-coded inline markers visualization."""
    # Create a list of all boundary positions from all tokenizers
    boundaries = {}
    for tokenizer_name, result in results.items():
        for start, end in result["spans"]:
            if end not in boundaries:
                boundaries[end] = []
            boundaries[end].append({
                "tokenizer": tokenizer_name,
                "color": TOKENIZER_CONFIG[tokenizer_name]["color"]
            })
    
    # Create segments between boundaries
    markers = []
    last_pos = 0
    positions = sorted(boundaries.keys())
    
    for pos in positions:
        # Add the text segment before this boundary
        if pos > last_pos:
            segment_text = text[last_pos:pos]
            markers.append({
                "text": segment_text,
                "tokenizer": None,
                "color": None,
                "is_boundary": False
            })
        
        # Add markers for each tokenizer at this boundary
        for boundary in boundaries[pos]:
            # Usually this would be a special character or HTML tag,
            # but for simplicity we'll just add a newline character
            markers.append({
                "text": "▪",  # Or any marker you prefer
                "tokenizer": boundary["tokenizer"],
                "color": boundary["color"],
                "is_boundary": True
            })
        
        last_pos = pos
    
    # Add any remaining text after the last boundary
    if last_pos < len(text):
        markers.append({
            "text": text[last_pos:],
            "tokenizer": None,
            "color": None,
            "is_boundary": False
        })
    
    return markers


def generate_sentence_table(text: str, results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate data for the interactive sentence table visualization with each tokenizer's sentences in original order."""
    # Create a structure to hold each tokenizer's sentences in their original order
    tokenizer_sentences = {}
    max_sentences = 0
    
    for tokenizer_name, result in results.items():
        # Filter out empty sentences or those that are just whitespace
        valid_sentences = [s.strip() for s in result["sentences"] if s.strip()]
        tokenizer_sentences[tokenizer_name] = valid_sentences
        
        # Track maximum number of sentences for any tokenizer
        max_sentences = max(max_sentences, len(valid_sentences))
    
    # Log sentence data
    logger.debug(f"Max sentence count: {max_sentences}")
    for name, sentences in tokenizer_sentences.items():
        logger.debug(f"{name}: {len(sentences)} sentences")
    
    # Prepare table data for template
    table_data = {
        "tokenizer_names": list(tokenizer_sentences.keys()),
        "tokenizer_colors": {name: TOKENIZER_CONFIG[name]["color"] for name in tokenizer_sentences.keys()},
        "tokenizer_display_names": {name: TOKENIZER_CONFIG[name]["display_name"] for name in tokenizer_sentences.keys()},
        "max_rows": max_sentences,
        "sentences": tokenizer_sentences
    }
    
    return table_data


def calculate_text_statistics(text: str, results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate various statistics about the text and tokenization results."""
    # Count words (simple split by whitespace)
    words = text.split()
    word_count = len(words)
    
    # Count tokens (simple approximation)
    token_count = len(re.findall(r'\b\w+\b', text))
    
    # Calculate average word length
    avg_word_length = sum(len(word) for word in words) / max(1, word_count)
    
    # Calculate average sentence length (words per sentence) for each tokenizer
    tokenizer_stats = {}
    for name, result in results.items():
        sentences = result["sentences"]
        sentence_count = len(sentences)
        
        # Words per sentence
        words_per_sentence = []
        chars_per_sentence = []
        
        for sentence in sentences:
            sentence_words = sentence.split()
            words_per_sentence.append(len(sentence_words))
            chars_per_sentence.append(len(sentence))
        
        avg_words_per_sentence = sum(words_per_sentence) / max(1, len(words_per_sentence))
        avg_chars_per_sentence = sum(chars_per_sentence) / max(1, len(chars_per_sentence))
        
        # Add to tokenizer stats
        tokenizer_stats[name] = {
            "avg_words_per_sentence": round(avg_words_per_sentence, 1),
            "avg_chars_per_sentence": round(avg_chars_per_sentence, 1),
            "min_sentence_length": min(chars_per_sentence) if chars_per_sentence else 0,
            "max_sentence_length": max(chars_per_sentence) if chars_per_sentence else 0,
            "median_sentence_length": statistics.median(chars_per_sentence) if chars_per_sentence else 0
        }
    
    return {
        "word_count": word_count,
        "token_count": token_count,
        "avg_word_length": round(avg_word_length, 1),
        "character_count": len(text),
        "tokenizer_stats": tokenizer_stats
    }


def generate_length_distribution(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate data for visualizing sentence length distribution."""
    distribution_data = {}
    
    for name, result in results.items():
        # Calculate sentence lengths
        sentence_lengths = [len(s) for s in result["sentences"]]
        
        if not sentence_lengths:
            distribution_data[name] = []
            continue
        
        # Create histogram data - divide into 10 bins
        min_length = min(sentence_lengths)
        max_length = max(sentence_lengths)
        
        # Ensure reasonable bin size
        bin_size = max(10, (max_length - min_length) // 10)
        
        # Create bins and count sentences in each bin
        bins = defaultdict(int)
        for length in sentence_lengths:
            bin_index = (length - min_length) // bin_size
            bins[bin_index] += 1
        
        # Find the maximum count for normalization
        max_count = max(bins.values()) if bins else 1
        
        # Convert to sorted list of {length, count} for template
        distribution = []
        for bin_idx, count in sorted(bins.items()):
            # Calculate normalized percentage (for more visible differences)
            normalized_percentage = round((count / max_count) * 100)
            
            # Ensure small values are still visible
            display_percentage = max(5, normalized_percentage) if count > 0 else 0
            
            distribution.append({
                "min_length": min_length + (bin_idx * bin_size),
                "max_length": min_length + ((bin_idx + 1) * bin_size) - 1,
                "count": count,
                "percentage": display_percentage,  # Normalized percentage for bar width
                "actual_percentage": round((count / len(sentence_lengths)) * 100)  # Actual percentage for label
            })
        
        distribution_data[name] = distribution
    
    return distribution_data


def cleanup_old_shared_analyses(max_age_seconds: int) -> None:
    """Remove old shared analyses to prevent memory buildup."""
    current_time = time.time()
    expired_ids = []
    
    for share_id, data in SHARED_ANALYSES.items():
        if current_time - data["timestamp"] > max_age_seconds:
            expired_ids.append(share_id)
    
    for share_id in expired_ids:
        del SHARED_ANALYSES[share_id]
    
    if expired_ids:
        logger.info(f"Removed {len(expired_ids)} expired shared analyses")


@app.get("/share/{share_id}", response_class=HTMLResponse)
async def view_shared_analysis(request: Request, share_id: str):
    """View a shared analysis by ID."""
    if share_id not in SHARED_ANALYSES:
        logger.warning(f"Shared analysis not found: {share_id}")
        raise HTTPException(status_code=404, detail="Shared analysis not found or has expired")
    
    # Get the shared data
    shared_data = SHARED_ANALYSES[share_id]
    
    # Render the index page with the shared text and tokenizers pre-populated
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "shared_text": shared_data["text"],
            "shared_tokenizers": shared_data["tokenizer_names"],
            "is_shared": True
        }
    )


@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc):
    """Handle 404 errors with a custom template."""
    logger.warning(f"404 Not Found: {request.url.path}")
    return templates.TemplateResponse(
        "error.html", 
        {"request": request, "status_code": 404, "detail": "Page not found"}, 
        status_code=404
    )


@app.exception_handler(500)
async def server_error_exception_handler(request: Request, exc):
    """Handle 500 errors with a custom template."""
    logger.error(f"500 Server Error: {str(exc)}")
    logger.error(traceback.format_exc())
    return templates.TemplateResponse(
        "error.html", 
        {"request": request, "status_code": 500, "detail": str(exc)}, 
        status_code=500
    )


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting application via __main__")
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8080, 
        reload=True,
        log_level="info"
    )