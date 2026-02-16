#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in February 2020

Modified February 2025:
- Incorporated NLTK for tokenization and stopword removal.
- Automatic download of missing 'punkt_tab'.
- Replaced Dirichlet smoothing with Jelinek–Mercer smoothing in the PMI weighting.
  For each cell (w,c) the smoothed count is computed as:
      new_count = (1 - jm_lambda) * count(w,c) + jm_lambda * (count(w) * count(c) / N)
  and then PMI is given by:
      PMI(w,c) = log( (new_count * N) / (count(w)*count(c)) )
  
Based on:
Jungmaier/Kassner/Roth (2020): "Dirichlet-Smoothed Word Embeddings for Low-Resource Settings"
and
Jelinek–Mercer smoothing as described in:
https://sigir.org/wp-content/uploads/2017/06/p268.pdf
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
      - Tokenize using NLTK.
      - Convert tokens to lowercase.
      - Remove tokens that do not contain at least one alphabetic character.
      - Remove stopwords.
    """
    tokens = []
    raw_tokens = word_tokenize(line)
    for t in raw_tokens:
        t_lower = t.lower()
        # Skip tokens without any alphabetic character.
        if not re.search('[a-zA-Z]', t_lower):
            continue
        # Skip stopwords.
        if t_lower in stopwords.words('english'):
            continue
        tokens.append(t_lower)
    return tokens


def file_to_cooc_matrix(file_name, chunk_size=3000000, window_size=5,
                        min_count=1, subsampling_rate=0.00001, verbose=True):
    """
    Reads a text corpus file (one raw text line per line) and returns a sparse
    co-occurrence matrix. It:
      - Tokenizes each line (using NLTK).
      - Removes stopwords.
      - Lowercases tokens.
      - Applies a sliding window to count co-occurrences.
    
    Parameters:
      chunk_size: approximate size in bytes per chunk (for memory management).
      window_size: max distance from the “middle” word for context.
      min_count: keep only words with frequency >= min_count.
      subsampling_rate: frequent word downsampling (similar to word2vec).
      verbose: print progress information.
    """
    word_count = defaultdict(int)

    if verbose:
        print("Counting chunks and building vocabulary...")

    with open(file_name, encoding='utf-8') as corpus_file:
        chunks_total = 0
        # First pass: count word frequencies.
        while True:
            text_chunk = corpus_file.readlines(chunk_size)
            if not text_chunk:
                break
            chunks_total += 1
            for line in text_chunk:
                tokens = clean_and_filter_tokens(line)
                for token in tokens:
                    word_count[token] += 1

        # Build vocabulary: only keep words with frequency >= min_count.
        vocab = [word for word, count in sorted(word_count.items(),
                                                 key=lambda x: x[1],
                                                 reverse=True)
                 if count >= min_count]
        vocab_set = set(vocab)

        # Prepare subsampling if requested.
        if subsampling_rate:
            corpus_size = sum(word_count.values())
            subsampling_threshold = subsampling_rate * corpus_size
            subsampling_dict = {
                w: 1 - math.sqrt(subsampling_threshold / count)
                for w, count in word_count.items() if count > subsampling_threshold
            }
            rand = random.Random(0)
            if verbose:
                print(f"Corpus size: {corpus_size}")
                print(f"Subsampling threshold: {subsampling_threshold}")
                print(f"Words in subsampling dictionary: {len(subsampling_dict)}")

        if verbose:
            print(f"Total chunks: {chunks_total}")
            print(f"Vocabulary size (after min_count): {len(vocab)}")
            print("Building co-occurrence matrix...")

        # Rewind file to beginning for the second pass.
        corpus_file.seek(0, 0)
        word_to_id = {w: i for i, w in enumerate(vocab)}
        m = csr_matrix((len(vocab), len(vocab)), dtype=float)

        chunk_count = 0
        while True:
            text_chunk = corpus_file.readlines(chunk_size)
            if not text_chunk:
                break
            chunk_count += 1
            if verbose:
                if chunk_count != chunks_total:
                    print(f"Processing chunk {chunk_count} of {chunks_total}", end="\r")
                else:
                    print(f"Processing chunk {chunk_count} of {chunks_total}")

            chunk_tokens = []
            for line in text_chunk:
                tokens = clean_and_filter_tokens(line)
                # Filter tokens to include only those in the final vocabulary.
                tokens = [t for t in tokens if t in vocab_set]
                # Apply subsampling if needed.
                if subsampling_rate:
                    tokens = [t for t in tokens
                              if (t not in subsampling_dict or rand.random() > subsampling_dict[t])]
                chunk_tokens.extend(tokens)

            row = []
            col = []
            data = []
            for i, middle_word in enumerate(chunk_tokens):
                mid_id = word_to_id[middle_word]
                context_start = max(0, i - window_size)
                context_end = min(len(chunk_tokens), i + window_size + 1)
                for j in range(context_start, context_end):
                    if j == i:
                        continue
                    context_word = chunk_tokens[j]
                    ctx_id = word_to_id[context_word]
                    row.append(mid_id)
                    col.append(ctx_id)
                    data.append(1)

            tmp_m = csr_matrix((data, (row, col)),
                               shape=(len(vocab), len(vocab)), dtype=float)
            m = m + tmp_m

        if verbose:
            print(f"Co-occurrence matrix shape: {m.shape[0]} x {m.shape[1]}")
            print(f"Non-zero entries: {m.nnz}")

    return m, word_to_id


def save_word_vectors(file_name, word_vector_matrix, word_to_id, vocab, verbose=True):
    """
    Saves word vectors to a text file in word2vec format:
       #(vectors)  #(dimensions)
       word1 dim1 dim2 dim3 ...
       word2 dim1 dim2 dim3 ...
       ...
    
    Parameters:
      file_name: name of output file.
      word_vector_matrix: matrix containing word embeddings.
      word_to_id: dictionary mapping words to row indices.
      vocab: list of all words in the vocabulary (ordered).
      verbose: if True, print progress information.
    """
    if verbose:
        print(f"Saving word vectors for {len(vocab)} words...")

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


def pmi_weight(cooc_matrix, jm_lambda=0.0, threshold=0, verbose=True):
    """
    Computes the PMI (or PPMI) matrix with optional Jelinek–Mercer smoothing.

    For each cell (w, c), the smoothed count is computed as:
      new_count = (1 - jm_lambda) * count(w,c) + jm_lambda * (count(w) * count(c) / N)
    and then the PMI is computed as:
      PMI(w,c) = log((new_count * N) / (count(w) * count(c)))

    Parameters:
      cooc_matrix: sparse co-occurrence matrix (vocabulary x vocabulary).
      jm_lambda: Jelinek–Mercer smoothing parameter (0 <= jm_lambda <= 1).
                 When set to 0, no smoothing is applied.
      threshold: if PMI values are below this threshold, they are set to 0.
      verbose:   if True, prints progress information.
    """
    if verbose:
        print("Computing PMI with Jelinek–Mercer smoothing:")

    # Compute marginal counts for words and contexts.
    row_counts = np.array(cooc_matrix.sum(axis=1))[:, 0]   # shape: (V,)
    col_counts = np.array(cooc_matrix.sum(axis=0))[0, :]    # shape: (V,)
    total_count = col_counts.sum()  # Total number of co-occurrences, N.

    # Convert the co-occurrence matrix to COO (coordinate) format.
    cooc_coo = cooc_matrix.tocoo()
    rows = cooc_coo.row
    cols = cooc_coo.col
    original_data = cooc_coo.data

    if jm_lambda != 0:
        if verbose:
            print(f"Applying Jelinek–Mercer smoothing with lambda = {jm_lambda}")
        # Compute the smoothed count for each nonzero entry.
        smoothed_data = ((1 - jm_lambda) * original_data +
                         jm_lambda * (row_counts[rows] * col_counts[cols] / total_count))
    else:
        if verbose:
            print("No smoothing applied (jm_lambda = 0).")
        smoothed_data = original_data.copy()

    # Compute PMI: PMI = log((smoothed_count * total_count) / (row_count * col_count))
    pmi_values = np.log((smoothed_data * total_count) / (row_counts[rows] * col_counts[cols]))

    # Apply thresholding (e.g., for PPMI, set negative PMI to 0).
    if threshold is not None:
        if verbose:
            print(f"Applying threshold: PMI values below {threshold} are set to 0.")
        pmi_values[pmi_values < threshold] = 0

    # Rebuild the sparse PMI matrix.
    from scipy.sparse import coo_matrix
    pmi_matrix = coo_matrix((pmi_values, (rows, cols)), shape=cooc_matrix.shape)
    pmi_matrix = pmi_matrix.tocsr()

    return pmi_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculates word embeddings from a corpus using PPMI, SVD, and Jelinek–Mercer smoothing."
    )
    parser.add_argument("corpus_file",
                        help="Text file with raw text lines. Tokenization and stopword removal are done by NLTK.")
    parser.add_argument("word_vector_filename",
                        help="Name of the output word vector file.")
    parser.add_argument("--window_size", "-w", type=int, default=5,
                        help="Context window size (default: 5).")
    parser.add_argument("--min_count", "-m", type=int, default=1,
                        help="Minimum word frequency to keep (default: 1).")
    parser.add_argument("--subsampling", "-s", type=float, default=0.0,
                        help="Subsampling rate like word2vec (default: 0.0).")
    parser.add_argument("--chunk_size", "-c", type=int, default=3000000,
                        help="Chunk size in bytes for reading the corpus.")
    parser.add_argument("--dimensions", "-d", type=int, default=100,
                        help="Embedding dimension (default: 100).")
    parser.add_argument("--jm_lambda", "-j", type=float, default=0.0,
                        help="Jelinek–Mercer smoothing parameter (0 for no smoothing, typical values between 0 and 1).")
    parser.add_argument("--threshold", "-t", type=float, default=0.0,
                        help="Threshold for PMI values (default: 0.0; negative values are set to 0).")
    parser.add_argument("--eigenvalue_weighting", "-e", type=float, default=0.0,
                        help="Singular value weighting exponent, range [0,1] (default: 0, which ignores weighting).")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print progress information.")
    args = parser.parse_args()

    if args.verbose:
        print(args)

    # 1) Build the co-occurrence matrix.
    m, word_to_id = file_to_cooc_matrix(
        file_name=args.corpus_file,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        min_count=args.min_count,
        subsampling_rate=args.subsampling,
        verbose=args.verbose
    )
    vocab = list(word_to_id.keys())

    # 2) Compute PMI with Jelinek–Mercer smoothing.
    m = pmi_weight(
        cooc_matrix=m,
        jm_lambda=args.jm_lambda,
        threshold=args.threshold,
        verbose=args.verbose
    )

    # 3) Perform SVD to reduce dimensions.
    if args.verbose:
        print("Performing SVD...", end="\r")

    if args.eigenvalue_weighting == 1:
        # Full SVD with singular values incorporated.
        svd = TruncatedSVD(n_components=args.dimensions, random_state=0)
        m = svd.fit_transform(m)
    elif args.eigenvalue_weighting == 0:
        # Standard truncated SVD (ignores singular values).
        u, _, _ = randomized_svd(m, n_components=args.dimensions, random_state=0)
        m = u
    else:
        # SVD with singular value weighting.
        u, s, _ = randomized_svd(m, n_components=args.dimensions, random_state=0)
        sigma = np.diag(s ** args.eigenvalue_weighting)
        m = u.dot(sigma)

    if args.verbose:
        print("SVD complete.")

    # 4) Normalize the word vectors.
    if args.verbose:
        print("Normalizing vectors...", end="\r")
    m = normalize(m, norm="l2", axis=1, copy=False)
    if args.verbose:
        print("Normalization complete.")

    # 5) Save the embeddings.
    save_word_vectors(
        file_name=args.word_vector_filename,
        word_vector_matrix=m,
        word_to_id=word_to_id,
        vocab=vocab,
        verbose=args.verbose
    )
