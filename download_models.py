#!/usr/bin/env python3
"""
Downloader script for word embedding models (Word2Vec and GloVe).

This script downloads pre-trained word embedding models from public sources:
- Word2Vec: Google's pre-trained model (GoogleNews-vectors-negative300)
- GloVe: Stanford's pre-trained models (various dimensions available)

The models are saved to the 'data/models/' directory.
"""

import os
import sys
import gzip
import zipfile
import requests
from tqdm import tqdm
import argparse


def download_file(url, destination):
    """
    Download a file from a URL with progress bar.
    
    Args:
        url: URL to download from
        destination: Local path to save the file
    """
    print(f"Downloading from {url}")
    print(f"Saving to {destination}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Stream the download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            progress_bar.update(size)
    
    print(f"Download complete: {destination}")


def extract_zip(zip_path, extract_to):
    """
    Extract a zip file.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to
    """
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction complete")


def extract_gzip(gz_path, output_path):
    """
    Extract a gzip file.
    
    Args:
        gz_path: Path to the gzip file
        output_path: Path for the extracted file
    """
    print(f"Extracting {gz_path}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            f_out.write(f_in.read())
    print(f"Extraction complete: {output_path}")


def download_word2vec(models_dir):
    """
    Download Google's pre-trained Word2Vec model.
    
    This model was trained on Google News corpus (about 100 billion words).
    The model contains 300-dimensional vectors for 3 million words and phrases.
    
    Note: This is a large file (~1.5 GB).
    
    Args:
        models_dir: Directory to save the model
    """
    print("\n" + "="*60)
    print("Downloading Word2Vec Model")
    print("="*60)
    
    url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    gz_file = os.path.join(models_dir, "GoogleNews-vectors-negative300.bin.gz")
    bin_file = os.path.join(models_dir, "GoogleNews-vectors-negative300.bin")
    
    if os.path.exists(bin_file):
        print(f"Word2Vec model already exists at {bin_file}")
        return
    
    if not os.path.exists(gz_file):
        try:
            download_file(url, gz_file)
        except Exception as e:
            print(f"Error downloading Word2Vec model: {e}")
            print("You can manually download from:")
            print("https://code.google.com/archive/p/word2vec/")
            return
    
    if os.path.exists(gz_file):
        extract_gzip(gz_file, bin_file)
        # Optionally remove the gz file to save space
        # os.remove(gz_file)
    
    print(f"Word2Vec model ready at: {bin_file}")


def download_glove(models_dir, dimension=100):
    """
    Download Stanford's pre-trained GloVe model.
    
    Available dimensions: 50, 100, 200, 300
    Trained on Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab).
    
    Args:
        models_dir: Directory to save the model
        dimension: Dimension of the word vectors (50, 100, 200, or 300)
    """
    print("\n" + "="*60)
    print(f"Downloading GloVe Model ({dimension}d)")
    print("="*60)
    
    if dimension not in [50, 100, 200, 300]:
        print(f"Invalid dimension: {dimension}. Choose from 50, 100, 200, or 300.")
        return
    
    # Using the 6B token model (Wikipedia 2014 + Gigaword 5)
    zip_filename = "glove.6B.zip"
    url = f"https://nlp.stanford.edu/data/{zip_filename}"
    zip_path = os.path.join(models_dir, zip_filename)
    txt_file = os.path.join(models_dir, f"glove.6B.{dimension}d.txt")
    
    if os.path.exists(txt_file):
        print(f"GloVe model already exists at {txt_file}")
        return
    
    if not os.path.exists(zip_path):
        try:
            download_file(url, zip_path)
        except Exception as e:
            print(f"Error downloading GloVe model: {e}")
            print("You can manually download from:")
            print("https://nlp.stanford.edu/projects/glove/")
            return
    
    if os.path.exists(zip_path):
        extract_zip(zip_path, models_dir)
        # Optionally remove the zip file to save space
        # os.remove(zip_path)
    
    print(f"GloVe model ready at: {txt_file}")


def main():
    """Main function to handle command-line arguments and initiate downloads."""
    parser = argparse.ArgumentParser(
        description="Download pre-trained word embedding models (Word2Vec and GloVe)"
    )
    parser.add_argument(
        '--model',
        choices=['word2vec', 'glove', 'both'],
        default='both',
        help='Which model to download (default: both)'
    )
    parser.add_argument(
        '--glove-dim',
        type=int,
        choices=[50, 100, 200, 300],
        default=100,
        help='Dimension for GloVe vectors (default: 100)'
    )
    parser.add_argument(
        '--models-dir',
        default='data/models',
        help='Directory to save models (default: data/models)'
    )
    
    args = parser.parse_args()
    
    # Create models directory
    models_dir = args.models_dir
    os.makedirs(models_dir, exist_ok=True)
    
    print("Word Embedding Model Downloader")
    print(f"Models will be saved to: {models_dir}")
    
    # Download requested models
    if args.model in ['word2vec', 'both']:
        download_word2vec(models_dir)
    
    if args.model in ['glove', 'both']:
        download_glove(models_dir, args.glove_dim)
    
    print("\n" + "="*60)
    print("Download process complete!")
    print("="*60)
    print("\nNote: Downloaded models can be large (>1GB).")
    print("Make sure you have sufficient disk space.")
    print("\nTo use the models in your code:")
    print("  - Word2Vec: Use gensim.models.KeyedVectors.load_word2vec_format()")
    print("  - GloVe: Use gensim.models.KeyedVectors.load_word2vec_format() with binary=False")


if __name__ == "__main__":
    main()
