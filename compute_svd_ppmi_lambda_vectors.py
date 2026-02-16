#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in February 2020

@author: Jakob Jungmaier

Calculates word embeddings from corpus by using PPMI, SVD,
and Dirichlet Smoothing. For more details cf. Jungmaier/Kassner/Roth(2020):
"Dirichlet-Smoothed Word Embeddings for Low-Resource Settings"

Modified February 2025:
- Incorporated NLTK for tokenization and stopword removal.
- Attempt to automatically download missing 'punkt_tab' if not found.
"""

import argparse
import math
import numpy as np
import random
import re
import nltk
###############################################################################
# BEGIN NLTK punkt_tab fix
###############################################################################
# Explanation: Some NLTK releases require 'punkt_tab' for word_tokenize() or
# related sentence tokenizers. If 'punkt_tab' is missing, you get:
#   LookupError: Resource punkt_tab not found.
# This code checks for its presence and downloads it if needed.
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
###############################################################################
# END NLTK punkt_tab fix
###############################################################################

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
from scipy.sparse import csr_matrix, dok_matrix
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def clean_and_filter_tokens(line):
    """
    Given a line of text, return a list of cleaned tokens:
    - Tokenize using NLTK
    - Lowercase
    - Remove tokens that are only punctuation or numeric
    - Remove stopwords
    """
    tokens = []
    raw_tokens = word_tokenize(line)  # Basic NLTK tokenization

    for t in raw_tokens:
        t_lower = t.lower()
        # Remove tokens that do not contain at least one alphabetic character
        if not re.search('[a-zA-Z]', t_lower):
            continue

        # Skip NLTK's English stopwords
        if t_lower in stopwords.words('english'):
            continue

        tokens.append(t_lower)

    return tokens


def file_to_cooc_matrix(file_name, chunk_size=3000000, window_size=5,
                        min_count=1, subsampling_rate=0.00001, verbose=True):
    """
    Takes a text corpus file (raw text lines) and returns
    a scipy sparse co-occurrence matrix. 
    - Tokenizes each line with NLTK
    - Removes stopwords
    - Lowercases
    - Applies a sliding window to count co-occurrences
    
    Parameters:
        chunk_size: approximate size in bytes per chunk (memory management)
        window_size: max distance from the "middle" word for context
        min_count: keep only words with frequency >= min_count
        subsampling_rate: "frequent word" downsampling (similar to word2vec)
        verbose: print progress info
    """
    word_count = defaultdict(int)

    # Counting chunks to process & computing raw vocabulary frequencies
    if verbose:
        print("Counting chunks, computing vocabulary...")

    with open(file_name, encoding='utf-8') as corpus_file:
        chunks_total = 0
        while True:
            text_chunk = corpus_file.readlines(chunk_size)
            if not text_chunk:
                break
            chunks_total += 1

            for line in text_chunk:
                tokens = clean_and_filter_tokens(line)
                for token in tokens:
                    word_count[token] += 1

        # Build vocabulary based on min_count
        vocab = [word for word, count in sorted(word_count.items(),
                                                key=lambda x: x[1],
                                                reverse=True)
                 if count >= min_count]
        vocab_set = set(vocab)

        # Subsampling prep
        if subsampling_rate:
            corpus_size = sum(word_count.values())
            subsampling_threshold = subsampling_rate * corpus_size
            subsampling_dict = {
                w: 1 - math.sqrt(subsampling_threshold / count)
                for w, count in word_count.items()
                if count > subsampling_threshold
            }
            rand = random.Random(0)
            if verbose:
                print(f"Corpus size: {corpus_size}")
                print(f"Subsampling threshold: {subsampling_threshold}")
                print(f"Words in subsampling dictionary: {len(subsampling_dict)}")

        if verbose:
            print(f"Chunks to process: {chunks_total}")
            print(f"Vocabulary size (after min_count): {len(vocab)}")
            print("Computing co-occurrence matrix:")

        corpus_file.seek(0, 0)

        word_to_id = {w: i for i, w in enumerate(vocab)}
        m = csr_matrix((len(vocab), len(vocab)), dtype=int)

        chunk_count = 0
        while True:
            text_chunk = corpus_file.readlines(chunk_size)
            if not text_chunk:
                break
            chunk_count += 1

            if verbose:
                if chunk_count != chunks_total:
                    print(f"Processing chunk {chunk_count} of {chunks_total}",
                          end="\r")
                else:
                    print(f"Processing chunk {chunk_count} of {chunks_total}")

            chunk_tokens = []
            for line in text_chunk:
                tokens = clean_and_filter_tokens(line)
                # Filter out tokens not in final vocab
                tokens = [t for t in tokens if t in vocab_set]
                # Subsampling
                if subsampling_rate:
                    tokens = [
                        t for t in tokens
                        if (t not in subsampling_dict or
                            rand.random() > subsampling_dict[t])
                    ]
                chunk_tokens.extend(tokens)

            row = []
            col = []
            data = []

            for i, middle_word in enumerate(chunk_tokens):
                mid_id = word_to_id[middle_word]
                context_start = max(0, i - window_size)
                context_end = min(len(chunk_tokens), i + window_size + 1)

                for j in range(context_start, context_end):
                    if j != i:
                        context_word = chunk_tokens[j]
                        ctx_id = word_to_id[context_word]
                        row.append(mid_id)
                        col.append(ctx_id)
                        data.append(1)

            tmp_m = csr_matrix((data, (row, col)),
                               shape=(len(vocab), len(vocab)), dtype=float)

            m = m + tmp_m

        if verbose:
            print(f"Matrix shape: {m.shape[0]} x {m.shape[1]}")
            print(f"Non-zero elements: {m.nnz}")

    return m, word_to_id


def save_word_vectors(file_name, word_vector_matrix, word_to_id, vocab,
                      verbose=True):
    """
    Saves word vectors from a word vector matrix to a text file (in word2vec
    format):

    #(vectors) #(dimensions)
    word1 dim1 dim2 dim3 ...
    word2 dim1 dim2 dim3 ...
    .     .    .    .
    .     .    .    .
    .     .    .    .

    Parameters:
        file_name: name of output file
        word_vector_matrix: matrix containing word embeddings
        word_to_id: dictionary mapping words to row indices
        vocab: list of all words in vocab (ordered)
        verbose: print progress info
    """
    if verbose:
        print(f"Saving word vectors for {len(vocab)} words:")

    with open(file_name, "w", encoding='utf-8') as vector_file:
        vector_file.write(f"{word_vector_matrix.shape[0]} {word_vector_matrix.shape[1]}\n")

        for i, word in enumerate(vocab, start=1):
            row_vec = word_vector_matrix[word_to_id[word], :]
            vector_file.write(word + " " + " ".join(map(str, row_vec)) + "\n")

            if verbose:
                if i % 1000 == 0:
                    print(f"{i} of {len(vocab)} word vectors saved.", end="\r")
                elif i == len(vocab):
                    print(f"{i} of {len(vocab)} word vectors saved.")


def pmi_weight(cooc_matrix, smoothing_factor=0, threshold=0, verbose=True):
    """
    Calculates (P)PMI matrix with optional Dirichlet smoothing

    Parameters:
        cooc_matrix: scipy sparse co-occurrence matrix
        smoothing_factor: lambda (0 = no smoothing)
        threshold: cut off for negative PMI values (PPMI if 0)
        verbose: print progress info
    """
    if verbose:
        print("PMI weighting:")
    if smoothing_factor != 0:
        if verbose:
            print(f"Smoothing with lambda={smoothing_factor}")
        sum_w = np.array(cooc_matrix.sum(axis=1))[:, 0] \
                + (smoothing_factor * cooc_matrix.shape[0])
        sum_c = np.array(cooc_matrix.sum(axis=0))[0, :] \
                + (smoothing_factor * cooc_matrix.shape[0])
    else:
        if verbose:
            print("No smoothing.")
        sum_w = np.array(cooc_matrix.sum(axis=1))[:, 0]
        sum_c = np.array(cooc_matrix.sum(axis=0))[0, :]

    sum_total = sum_c.sum()

    # Add smoothing to counts if needed
    if smoothing_factor != 0:
        cooc_matrix.data = cooc_matrix.data + smoothing_factor

    sum_w = np.reciprocal(sum_w)
    sum_c = np.reciprocal(sum_c)

    pmi = csr_matrix(cooc_matrix)
    pmi = multiply_by_rows(pmi, sum_w)
    pmi = multiply_by_columns(pmi, sum_c)
    pmi *= sum_total
    pmi.data = np.log(pmi.data)

    # Threshold for PPMI
    if threshold is not None:
        if verbose:
            print(f"Filtering values below {threshold}")
        pmi.data[pmi.data < threshold] = 0

    return pmi


def multiply_by_rows(matrix, row_coefs):
    """Multiplies rows by row_coefs."""
    normalizer = dok_matrix((len(row_coefs), len(row_coefs)))
    normalizer.setdiag(row_coefs)
    return normalizer.tocsr().dot(matrix)


def multiply_by_columns(matrix, col_coefs):
    """Multiplies columns by col_coefs."""
    normalizer = dok_matrix((len(col_coefs), len(col_coefs)))
    normalizer.setdiag(col_coefs)
    return matrix.dot(normalizer.tocsr())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculates word embeddings '
                                                 'from corpus using PPMI, SVD, '
                                                 'and Dirichlet smoothing. Cf. '
                                                 'Jungmaier/Kassner/Roth (2020).')
    parser.add_argument('corpus_file',
                        help='Text file with lines of raw text. Tokenization '
                             'and stopword removal done with NLTK.')
    parser.add_argument('word_vector_filename',
                        help='Name of the output word vector file.')
    parser.add_argument('--window_size', '-w', type=int, default=5,
                        help='Context window size (default: 5).')
    parser.add_argument('--min_count', '-m', type=int, default=1,
                        help='Minimum word frequency to keep (default: 1).')
    parser.add_argument('--subsampling', '-s', type=float, default=0.0,
                        help='Subsampling rate like word2vec (default: 0.0).')
    parser.add_argument('--chunk_size', '-c', type=int, default=3000000,
                        help='Chunk size in bytes for reading the corpus.')
    parser.add_argument('--dimensions', '-d', type=int, default=100,
                        help='Embedding dimension (default: 100).')
    parser.add_argument('--smoothing_factor', '-l', type=float, default=0.0001,
                        help='Smoothing factor lambda (default: 0.0001).')
    parser.add_argument('--threshold', '-t', type=float, default=0.0,
                        help='Threshold for PMI->PPMI (default: 0).')
    parser.add_argument('--eigenvalue_weighting', '-e', type=float, default=0.0,
                        help='Singular value weighting exponent, range [0,1], '
                             'default=0 (ignored).')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print progress info.')
    args = parser.parse_args()

    print(args)

    # 1) Build co-occurrence matrix
    m, word_to_id = file_to_cooc_matrix(
        file_name=args.corpus_file,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        min_count=args.min_count,
        subsampling_rate=args.subsampling,
        verbose=args.verbose
    )
    vocab = list(word_to_id.keys())

    # 2) Compute PPMI with Dirichlet smoothing
    m = pmi_weight(
        cooc_matrix=m,
        smoothing_factor=args.smoothing_factor,
        threshold=args.threshold,
        verbose=args.verbose
    )

    # 3) SVD
    if args.verbose:
        print("Performing SVD...", end="\r")

    from sklearn.utils.extmath import randomized_svd
    from sklearn.decomposition import TruncatedSVD

    if args.eigenvalue_weighting == 1:
        # Full SVD with Sigma in final vectors
        svd = TruncatedSVD(n_components=args.dimensions, random_state=0)
        m = svd.fit_transform(m)
    elif args.eigenvalue_weighting == 0:
        # Standard truncated SVD ignoring singular values
        u, _, _ = randomized_svd(m, n_components=args.dimensions,
                                 random_state=0)
        m = u
    else:
        # Weighted by s^alpha
        u, s, _ = randomized_svd(m, n_components=args.dimensions,
                                 random_state=0)
        sigma = np.diag(s ** args.eigenvalue_weighting)
        m = u.dot(sigma)

    if args.verbose:
        print("SVD...done.")

    # 4) Normalize vectors
    if args.verbose:
        print("Normalizing vectors...", end="\r")
    m = normalize(m, norm='l2', axis=1, copy=False)
    if args.verbose:
        print("Normalizing vectors...done.")

    # 5) Save embeddings
    save_word_vectors(
        file_name=args.word_vector_filename,
        word_vector_matrix=m,
        word_to_id=word_to_id,
        vocab=vocab,
        verbose=args.verbose
    )
