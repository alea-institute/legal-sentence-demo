# Legal Sentence Boundary Detection

A web application for demonstrating and comparing different sentence boundary detection algorithms on legal text.

## Features

- Multiple sentence tokenization algorithms:
  - NLTK
  - SpaCy
  - PySD
  - NUPunkt
  - Character Boundary-based

- Web interface for interactive testing
- REST API for programmatic access

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/legal-sentence-demo.git
cd legal-sentence-demo

# Setup the environment
./setup.sh
```

## Usage

```bash
# Run the application
python run.py
```

Then open your browser to http://localhost:8080

## Configuration

Edit `static/js/presets.json` to add common legal text examples.

## License

MIT