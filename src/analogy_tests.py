#!/usr/bin/env python3
"""
Analogy testing module for word embeddings.

This module provides functions to test and evaluate word analogies using
word embedding models, examining vector arithmetic behavior.
"""

import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
from pathlib import Path


def test_analogy(model, word_a, word_b, word_c, target_word, top_n=10, search_space=50000):
    """
    Test the analogy: word_a is to word_b as word_c is to ?
    
    Performs vector arithmetic (word_c - word_a + word_b) and finds nearest neighbors.
    
    Args:
        model: Word embedding model (KeyedVectors)
        word_a: First word in the base pair
        word_b: Second word in the base pair
        word_c: First word in the target pair
        target_word: Expected result word
        top_n: Number of nearest neighbors to return
        search_space: Number of most frequent words to search through
        
    Returns:
        tuple: (top_neighbors, target_rank) where top_neighbors is a list of
               (word, similarity) tuples and target_rank is the rank of the target word
    """
    print(f"\n=== Testing Analogy: {word_a} : {word_b} :: {word_c} : ? ===")
    
    try:
        # Get individual vectors
        vec_a = model[word_a]
        vec_b = model[word_b]
        vec_c = model[word_c]
        
        # Perform vector arithmetic: word_c - word_a + word_b
        result_vector = vec_c - vec_a + vec_b
        
        # Find nearest neighbors to the result
        # Exclude the input words from the results
        exclude_words = [word_a, word_b, word_c]
        
        nearest_neighbors = []
        search_limit = min(search_space, len(model.index_to_key))
        
        for word in model.index_to_key[:search_limit]:
            if word not in exclude_words:
                similarity = 1 - cosine(result_vector, model[word])
                nearest_neighbors.append((word, similarity))
        
        # Sort by similarity (descending)
        nearest_neighbors.sort(key=lambda x: x[1], reverse=True)
        top_neighbors = nearest_neighbors[:top_n]
        
        print(f"Vector arithmetic: {word_c} - {word_a} + {word_b}")
        print(f"Top {top_n} nearest neighbors:")
        
        target_found = False
        target_rank = None
        
        for i, (word, similarity) in enumerate(top_neighbors, 1):
            marker = "<<<< TARGET!" if word.lower() == target_word.lower() else ""
            print(f"{i:2d}. {word:<20} (similarity: {similarity:.4f}) {marker}")
            
            if word.lower() == target_word.lower():
                target_found = True
                target_rank = i
        
        # Check if target word appears further down
        if not target_found:
            for i, (word, similarity) in enumerate(nearest_neighbors, 1):
                if word.lower() == target_word.lower():
                    target_rank = i
                    target_found = True
                    break
        
        if target_found:
            print(f"\n'{target_word}' found at rank {target_rank}")
        else:
            print(f"\n'{target_word}' not found in top {len(nearest_neighbors)} results")
        
        # Also check if target word exists in vocabulary
        if target_word in model:
            target_similarity = 1 - cosine(result_vector, model[target_word])
            print(f"Direct similarity to '{target_word}': {target_similarity:.4f}")
        else:
            print(f"'{target_word}' not in model vocabulary")
        
        return top_neighbors, target_rank
        
    except KeyError as e:
        print(f"Error: Word not found in vocabulary: {e}")
        return None, None

# ORIGINAL FUNCTION. SEE REPLACED VERSION BELOW
# def run_analogy_test_suite(model, test_cases=None):
#     """
#     Run multiple analogy tests to examine vector arithmetic behavior.
    
#     Args:
#         model: Word embedding model (KeyedVectors)
#         test_cases: List of tuples (word_a, word_b, word_c, target_word).
#                    If None, uses default test cases.
                   
#     Returns:
#         dict: Results for each test case with neighbors and target rank
#     """
    
#     '''
#     if test_cases is None:
#         # Default test cases
#         test_cases = [
#             ("man", "woman", "king", "queen"),
#             ("man", "woman", "computer_programmer", "homemaker"),
#             ("Paris", "London", "France", "England"),
#             ("walking", "walked", "swimming", "swam"),
#             ("good", "better", "bad", "worse"),
#             ("Tokyo", "Japan", "Paris", "France"),
#             ("big", "bigger", "small", "smaller"),
#             ("Athens", "Greece", "Berlin", "Germany"),
#         ]
#     '''
#     csv_path = Path(__file__).parent.parent / "data" / "analogies.csv"
#     df = pd.read_csv(csv_path, header=None)
    
#     test_cases = [tuple(row) for row in df.to_numpy()]

            
#     results = {}
    
#     print("\n" + "="*70)
#     print("RUNNING ANALOGY TEST SUITE")
#     print("="*70)
    
#     for word_a, word_b, word_c, target in test_cases:
#         neighbors, rank = test_analogy(model, word_a, word_b, word_c, target)
#         results[f"{word_a}:{word_b}::{word_c}:{target}"] = {
#             'neighbors': neighbors,
#             'target_rank': rank,
#             'test_case': (word_a, word_b, word_c, target)
#         }
#         print("-" * 70)
    
#     return results

def run_analogy_test_suite(model):
    """
    Run multiple analogy tests to examine vector arithmetic behavior.

    Always loads test cases from data/analogies.csv (expects a header).
    If the CSV has more than 4 columns (e.g. a category column), the first
    four columns are used as the analogy tuple: (word_a, word_b, word_c, target).
    """
    csv_path = Path(__file__).parent.parent / "data" / "analogies.csv"

    # Read CSV (header row present in your file)
    df = pd.read_csv(csv_path, header=0, dtype=str).fillna("")

    # Drop rows that are completely empty (just in case)
    df = df.dropna(how="all")

    # If there are extra columns (like 'category'), keep only the first 4
    if df.shape[1] > 4:
        print(f"Note: {csv_path.name} has {df.shape[1]} columns — using the first 4 columns as (word1,word2,word3,word4).")
        df = df.iloc[:, :4]

    # Strip whitespace from every cell and convert to list of 4-tuples
    #df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    test_cases = [tuple(row) for row in df.values.tolist()]
    print(f"Loaded {len(test_cases)} test case(s) from {csv_path}")

    results = {}

    print("\n" + "="*70)
    print("RUNNING ANALOGY TEST SUITE")
    print("="*70)

    for word_a, word_b, word_c, target in test_cases:
        neighbors, rank = test_analogy(model, word_a, word_b, word_c, target)
        results[f"{word_a}:{word_b}::{word_c}:{target}"] = {
            'neighbors': neighbors,
            'target_rank': rank,
            'test_case': (word_a, word_b, word_c, target)
        }
        print("-" * 70)

    return results



def print_test_summary(results):
    """
    Print a summary of analogy test results.
    
    Args:
        results: Results dictionary from run_analogy_test_suite()
    """
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    successful_top5 = 0
    successful_top10 = 0
    total_analogies = len(results)
    
    for analogy, result in results.items():
        rank = result['target_rank']
        if rank and rank <= 5:
            print(f"✓ {analogy}: Target at rank {rank} (TOP 5)")
            successful_top5 += 1
            successful_top10 += 1
        elif rank and rank <= 10:
            print(f"~ {analogy}: Target at rank {rank} (TOP 10)")
            successful_top10 += 1
        elif rank:
            print(f"✗ {analogy}: Target at rank {rank} (OUTSIDE TOP 10)")
        else:
            print(f"✗ {analogy}: Target NOT FOUND")
    
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Targets in top 5:  {successful_top5}/{total_analogies} ({successful_top5/total_analogies:.1%})")
    print(f"  Targets in top 10: {successful_top10}/{total_analogies} ({successful_top10/total_analogies:.1%})")
    print(f"{'='*70}")


def explore_nearest_neighbors(model, word, n=10):
    """
    Explore the nearest neighbors of a given word.
    
    Args:
        model: Word embedding model (KeyedVectors)
        word: Word to explore
        n: Number of nearest neighbors to show
        
    Returns:
        list: List of (word, similarity) tuples
    """
    if word not in model:
        print(f"Error: '{word}' not found in vocabulary")
        return []
    
    print(f"\nNearest neighbors of '{word}':")
    neighbors = model.most_similar(word, topn=n)
    
    for i, (neighbor, similarity) in enumerate(neighbors, 1):
        print(f"{i:2d}. {neighbor:<20} (similarity: {similarity:.4f})")
    
    return neighbors


def calculate_vector_arithmetic(model, positive_words, negative_words, topn=10):
    """
    Perform custom vector arithmetic on words.
    
    Args:
        model: Word embedding model (KeyedVectors)
        positive_words: List of words to add
        negative_words: List of words to subtract
        topn: Number of results to return
        
    Returns:
        list: List of (word, similarity) tuples
    """
    try:
        results = model.most_similar(positive=positive_words, negative=negative_words, topn=topn)
        
        print(f"\nVector arithmetic:")
        print(f"  Positive: {', '.join(positive_words)}")
        print(f"  Negative: {', '.join(negative_words)}")
        print(f"\nTop {topn} results:")
        
        for i, (word, similarity) in enumerate(results, 1):
            print(f"{i:2d}. {word:<20} (similarity: {similarity:.4f})")
        
        return results
        
    except KeyError as e:
        print(f"Error: Word not found in vocabulary: {e}")
        return []
