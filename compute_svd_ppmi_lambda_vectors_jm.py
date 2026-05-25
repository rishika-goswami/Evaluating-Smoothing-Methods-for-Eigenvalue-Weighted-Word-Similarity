#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rishika Goswami
goswami.rishika67@gmail.com

Builds word embeddings from a raw text corpus using PPMI, SVD, and
Jelinek-Mercer smoothing. The script writes word2vec-style vectors for
downstream word-similarity evaluation.
"""

import argparse
import math
import os
import random
import re
from collections import defaultdict
from functools import lru_cache

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd


def ensure_nltk_resources():
    """Download the small tokenizer/stopword resources if they are missing."""
    resources = (
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
    )
    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)


@lru_cache(maxsize=1)
def english_stopword_set():
    return set(stopwords.words('english'))


ensure_nltk_resources()


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
    for token in raw_tokens:
        token_lower = token.lower()
        if not re.search('[a-zA-Z]', token_lower):
            continue
        if token_lower in english_stopword_set():
            continue
        tokens.append(token_lower)
    return tokens


def file_to_cooc_matrix(file_name, chunk_size=3000000, window_size=5,
                        min_count=1, subsampling_rate=0.00001, verbose=True):
    """
    Reads a text corpus file and returns a sparse co-occurrence matrix.

    The corpus is tokenized with NLTK, lowercased, filtered for stopwords, and
    then converted into sliding-window co-occurrence counts.
    """
    word_count = defaultdict(int)

    if verbose:
        print("Counting chunks and building vocabulary...")

    with open(file_name, encoding='utf-8') as corpus_file:
        chunks_total = 0
        while True:
            text_chunk = corpus_file.readlines(chunk_size)
            if not text_chunk:
                break
            chunks_total += 1
            for line in text_chunk:
                for token in clean_and_filter_tokens(line):
                    word_count[token] += 1

        vocab = [word for word, count in sorted(word_count.items(),
                                                key=lambda x: x[1],
                                                reverse=True)
                 if count >= min_count]
        vocab_set = set(vocab)

        if subsampling_rate:
            corpus_size = sum(word_count.values())
            subsampling_threshold = subsampling_rate * corpus_size
            subsampling_dict = {
                word: 1 - math.sqrt(subsampling_threshold / count)
                for word, count in word_count.items()
                if count > subsampling_threshold
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

        corpus_file.seek(0, 0)
        word_to_id = {word: i for i, word in enumerate(vocab)}
        matrix = csr_matrix((len(vocab), len(vocab)), dtype=float)

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
                tokens = [token for token in clean_and_filter_tokens(line)
                          if token in vocab_set]
                if subsampling_rate:
                    tokens = [
                        token for token in tokens
                        if (token not in subsampling_dict or
                            rand.random() > subsampling_dict[token])
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
                    if j == i:
                        continue
                    context_word = chunk_tokens[j]
                    row.append(mid_id)
                    col.append(word_to_id[context_word])
                    data.append(1)

            chunk_matrix = csr_matrix((data, (row, col)),
                                      shape=(len(vocab), len(vocab)),
                                      dtype=float)
            matrix = matrix + chunk_matrix

        if verbose:
            print(f"Co-occurrence matrix shape: {matrix.shape[0]} x {matrix.shape[1]}")
            print(f"Non-zero entries: {matrix.nnz}")

    return matrix, word_to_id


def save_word_vectors(file_name, word_vector_matrix, word_to_id, vocab,
                      verbose=True):
    """
    Saves word vectors to a text file in word2vec format:
       #(vectors) #(dimensions)
       word1 dim1 dim2 dim3 ...
       word2 dim1 dim2 dim3 ...
    """
    if verbose:
        print(f"Saving word vectors for {len(vocab)} words...")

    output_dir = os.path.dirname(file_name)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(file_name, "w", encoding='utf-8') as vector_file:
        vector_file.write(
            f"{word_vector_matrix.shape[0]} {word_vector_matrix.shape[1]}\n"
        )
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
    Computes the PMI or PPMI matrix with optional Jelinek-Mercer smoothing.

    For each nonzero cell (w, c), the smoothed count is:
      new_count = (1 - jm_lambda) * count(w,c)
                  + jm_lambda * (count(w) * count(c) / N)
    """
    if verbose:
        print("Computing PMI with Jelinek-Mercer smoothing:")

    row_counts = np.array(cooc_matrix.sum(axis=1))[:, 0]
    col_counts = np.array(cooc_matrix.sum(axis=0))[0, :]
    total_count = col_counts.sum()

    cooc_coo = cooc_matrix.tocoo()
    rows = cooc_coo.row
    cols = cooc_coo.col
    original_data = cooc_coo.data

    if jm_lambda != 0:
        if verbose:
            print(f"Applying Jelinek-Mercer smoothing with lambda = {jm_lambda}")
        smoothed_data = ((1 - jm_lambda) * original_data +
                         jm_lambda * (row_counts[rows] *
                                      col_counts[cols] / total_count))
    else:
        if verbose:
            print("No smoothing applied (jm_lambda = 0).")
        smoothed_data = original_data.copy()

    pmi_values = np.log(
        (smoothed_data * total_count) / (row_counts[rows] * col_counts[cols])
    )

    if threshold is not None:
        if verbose:
            print(f"Applying threshold: PMI values below {threshold} are set to 0.")
        pmi_values[pmi_values < threshold] = 0

    return coo_matrix((pmi_values, (rows, cols)), shape=cooc_matrix.shape).tocsr()


def svd_components(matrix, requested_dimensions, verbose=True):
    max_components = min(matrix.shape)
    if max_components < 1:
        raise ValueError("The co-occurrence matrix is empty; check the corpus and min_count.")
    components = min(requested_dimensions, max_components)
    if verbose and components != requested_dimensions:
        print(f"Using {components} SVD components because the matrix is {matrix.shape}.")
    return components


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculates word embeddings from a corpus using PPMI, SVD, and Jelinek-Mercer smoothing."
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
                        help="Jelinek-Mercer smoothing parameter (0 for no smoothing, typical values between 0 and 1).")
    parser.add_argument("--threshold", "-t", type=float, default=0.0,
                        help="Threshold for PMI values (default: 0.0; negative values are set to 0).")
    parser.add_argument("--eigenvalue_weighting", "-e", type=float, default=0.0,
                        help="Singular value weighting exponent, range [0,1] (default: 0, which ignores weighting).")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print progress information.")
    args = parser.parse_args()

    if args.verbose:
        print(args)

    m, word_to_id = file_to_cooc_matrix(
        file_name=args.corpus_file,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        min_count=args.min_count,
        subsampling_rate=args.subsampling,
        verbose=args.verbose
    )
    vocab = list(word_to_id.keys())

    m = pmi_weight(
        cooc_matrix=m,
        jm_lambda=args.jm_lambda,
        threshold=args.threshold,
        verbose=args.verbose
    )

    dimensions = svd_components(m, args.dimensions, verbose=args.verbose)

    if args.verbose:
        print("Performing SVD...", end="\r")

    if args.eigenvalue_weighting == 1:
        svd = TruncatedSVD(n_components=dimensions, random_state=0)
        m = svd.fit_transform(m)
    elif args.eigenvalue_weighting == 0:
        u, _, _ = randomized_svd(m, n_components=dimensions, random_state=0)
        m = u
    else:
        u, s, _ = randomized_svd(m, n_components=dimensions, random_state=0)
        sigma = np.diag(s ** args.eigenvalue_weighting)
        m = u.dot(sigma)

    if args.verbose:
        print("SVD complete.")

    if args.verbose:
        print("Normalizing vectors...", end="\r")
    m = normalize(m, norm="l2", axis=1, copy=False)
    if args.verbose:
        print("Normalization complete.")

    save_word_vectors(
        file_name=args.word_vector_filename,
        word_vector_matrix=m,
        word_to_id=word_to_id,
        vocab=vocab,
        verbose=args.verbose
    )
