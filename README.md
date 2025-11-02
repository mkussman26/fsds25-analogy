# fsds25-analogy

Repository for considering analogies using language models (Word2Vec and GloVe).

## Overview

This project provides tools and datasets for exploring and evaluating word analogies using pre-trained word embedding models. Word analogies follow the pattern: "A is to B as C is to D" (e.g., "man is to woman as king is to queen").

## Project Structure

```
fsds25-analogy/
├── data/               # Data files and models
│   ├── analogies.csv   # Standard word analogies dataset
│   └── models/         # Downloaded word embedding models (created on first download)
├── output/             # Output files from analysis
├── figures/            # Generated figures and visualizations
├── download_models.py  # Script to download Word2Vec and GloVe models
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Setup

### 1. Create a Virtual Environment

It's recommended to use a Python virtual environment to manage dependencies:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Download Word Embedding Models

Use the provided downloader script to fetch pre-trained models:

```bash
# Download both Word2Vec and GloVe models (default: GloVe 100d)
python download_models.py

# Download only Word2Vec
python download_models.py --model word2vec

# Download only GloVe with specific dimensions
python download_models.py --model glove --glove-dim 300

# Specify custom models directory
python download_models.py --models-dir /path/to/models
```

**Note:** Models are large files (>1GB). Ensure you have sufficient disk space and a stable internet connection.

## Available Models

### Word2Vec
- **Model:** GoogleNews-vectors-negative300
- **Size:** ~1.5 GB
- **Dimensions:** 300
- **Vocabulary:** 3 million words and phrases
- **Training Data:** Google News corpus (~100 billion words)

### GloVe
- **Model:** glove.6B (Wikipedia 2014 + Gigaword 5)
- **Size:** ~800 MB (for all dimensions)
- **Dimensions:** 50, 100, 200, or 300
- **Vocabulary:** 400K words
- **Training Data:** 6 billion tokens

## Dataset

The `data/analogies.csv` file contains standard word analogies organized by category:

- **Gender:** man/woman, king/queen relationships
- **Capital-Country:** Paris/France, London/England relationships
- **Grammar:** verb forms, adjective comparatives
- **Animal-Young:** dog/puppy, cat/kitten relationships
- **Math:** mathematical operations and concepts
- **Ordinal:** number ordering relationships

Format: `word1,word2,word3,word4,category`

Example: `man,woman,king,queen,gender`

## Usage

### Loading Models

```python
from gensim.models import KeyedVectors

# Load Word2Vec model
word2vec_model = KeyedVectors.load_word2vec_format(
    'data/models/GoogleNews-vectors-negative300.bin',
    binary=True
)

# Load GloVe model
glove_model = KeyedVectors.load_word2vec_format(
    'data/models/glove.6B.100d.txt',
    binary=False,
    no_header=True
)
```

### Testing Analogies

```python
# Test an analogy: man is to woman as king is to ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)  # Should return 'queen' with high similarity
```

### Loading the Analogies Dataset

```python
import pandas as pd

# Load analogies
analogies = pd.read_csv('data/analogies.csv')
print(analogies.head())
```

## Dependencies

- **numpy** (>=1.21.0): Numerical computing
- **pandas** (>=1.3.0): Data manipulation and analysis
- **gensim** (>=4.0.0): Word embedding models and similarity operations
- **requests** (>=2.26.0): HTTP library for downloading models
- **tqdm** (>=4.62.0): Progress bars for downloads
- **scikit-learn** (>=1.0.0): Machine learning utilities
- **matplotlib** (>=3.4.0): Plotting and visualization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See LICENSE file for details.

## References

- Word2Vec: [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- GloVe: [Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- Gensim: [https://radimrehurek.com/gensim/](https://radimrehurek.com/gensim/)
