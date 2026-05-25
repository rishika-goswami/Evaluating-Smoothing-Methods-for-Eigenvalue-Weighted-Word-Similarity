#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rishika Goswami
goswami.rishika67@gmail.com

Creates Spearman correlation plots for five word-similarity benchmarks using
the generated word-vector outputs. The CLI can plot all datasets or one
selected dataset at a time.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


EIGENVALUE_WEIGHTS = (
    ("0", 0.0),
    ("02", 0.2),
    ("04", 0.4),
    ("06", 0.6),
    ("08", 0.8),
    ("1", 1.0),
)
JM_LAMBDAS = (0.1, 0.3, 0.5, 0.7, 0.9)
METHOD_COLORS = {
    "Bayesian": "#246BFE",
    "Dirichlet": "#D13F31",
    "Jelinek-Mercer": "#218C5A",
}
DATASETS = {
    "simlex999": {
        "title": "SimLex-999",
        "dataset_file": "simlex999_preprocessed.txt",
        "output_dir": "simlex999",
        "vector_prefix": "sl",
        "plot_file": "spearman_correlation_simlex_avg.png",
    },
    "men": {
        "title": "MEN",
        "dataset_file": "men",
        "output_dir": "men",
        "vector_prefix": "men",
        "plot_file": "spearman_correlation_men_avg.png",
    },
    "wordsim353": {
        "title": "WordSim-353",
        "dataset_file": "wordsim_similarity_goldstandard.txt",
        "output_dir": "wordsim353",
        "vector_prefix": "ws",
        "plot_file": "spearman_correlation_wordsim_avg.png",
    },
    "rg-65": {
        "title": "RG-65",
        "dataset_file": "rg_processed.txt",
        "output_dir": "rg-65",
        "vector_prefix": "rg",
        "plot_file": "spearman_correlation_rg_avg.png",
    },
    "rw": {
        "title": "Rare Words",
        "dataset_file": "rw_processed.txt",
        "output_dir": "rw",
        "vector_prefix": "rw",
        "plot_file": "spearman_correlation_rw_avg.png",
    },
}
DATASET_ALIASES = {
    "simlex": "simlex999",
    "wordsim": "wordsim353",
    "rg": "rg-65",
    "rarewords": "rw",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Spearman correlations for generated word vectors."
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        help=("Dataset(s) to plot: all, simlex999, men, wordsim353, rg-65, "
              "rw. Aliases: simlex, wordsim, rg, rarewords."),
    )
    parser.add_argument(
        "--datasets-dir",
        default="datasets",
        type=Path,
        help="Directory containing benchmark word-pair files.",
    )
    parser.add_argument(
        "--vectors-dir",
        default="outputs",
        type=Path,
        help="Directory containing generated word-vector output folders.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        type=Path,
        help="Directory where generated plot PNGs are written.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if any expected vector file is missing.",
    )
    return parser.parse_args()


def resolve_datasets(dataset_args):
    normalized = [DATASET_ALIASES.get(name.lower(), name.lower())
                  for name in dataset_args]
    if "all" in normalized:
        return list(DATASETS)

    unknown = [name for name in normalized if name not in DATASETS]
    if unknown:
        valid = ", ".join(["all"] + list(DATASETS) + list(DATASET_ALIASES))
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}. Valid: {valid}")
    return normalized


def load_pairs(dataset_path):
    pairs = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 3:
                raise ValueError(
                    f"{dataset_path}:{line_number} should contain word1 word2 score."
                )
            pairs.append((parts[0], parts[1], float(parts[2])))
    return pairs


def load_word_vectors(vector_path, needed_words):
    vectors = {}
    with vector_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split()
        if len(header) != 2:
            raise ValueError(f"{vector_path} is not in word2vec text format.")

        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            if word not in needed_words:
                continue
            vectors[word] = np.asarray(parts[1:], dtype=float)
    return vectors


def cosine_similarity(vec1, vec2):
    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denominator == 0:
        return None
    return float(np.dot(vec1, vec2) / denominator)


def vector_specs(dataset_config, jm_lambda):
    prefix = dataset_config["vector_prefix"]
    jm_tag = f"{int(jm_lambda * 10):02d}"

    for method, method_tag in (("Bayesian", "b"), ("Dirichlet", "d")):
        for eg_tag, eig_weight in EIGENVALUE_WEIGHTS:
            yield method, eig_weight, f"op_{prefix}_{method_tag}_eg{eg_tag}.txt"

    for eg_tag, eig_weight in EIGENVALUE_WEIGHTS:
        yield "Jelinek-Mercer", eig_weight, f"op_{prefix}_jm{jm_tag}_eg{eg_tag}.txt"


def compute_spearman_for_file(vector_path, pairs, needed_words):
    vectors = load_word_vectors(vector_path, needed_words)
    human_scores = []
    computed_scores = []

    for word1, word2, human_score in pairs:
        if word1 not in vectors or word2 not in vectors:
            continue
        similarity = cosine_similarity(vectors[word1], vectors[word2])
        if similarity is None:
            continue
        human_scores.append(human_score)
        computed_scores.append(similarity)

    if len(human_scores) < 2:
        return None
    correlation, _ = spearmanr(human_scores, computed_scores)
    if np.isnan(correlation):
        return None
    return float(correlation)


def plot_method_results(ax, method_results):
    for method, values in method_results.items():
        values = sorted(values, key=lambda item: item[0])
        x_values = [eig_weight for eig_weight, corr in values if corr is not None]
        y_values = [corr for eig_weight, corr in values if corr is not None]
        if not x_values:
            continue
        ax.plot(
            x_values,
            y_values,
            "o-",
            label=method,
            color=METHOD_COLORS[method],
            linewidth=2,
            markersize=5,
        )

    ax.set_xticks([eig_weight for _, eig_weight in EIGENVALUE_WEIGHTS])
    ax.set_xlabel("Eigenvalue weighting")
    ax.set_ylabel("Spearman's $\\rho$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def plot_dataset(dataset_name, datasets_dir, vectors_dir, results_dir, strict=False):
    dataset_config = DATASETS[dataset_name]
    dataset_path = datasets_dir / dataset_config["dataset_file"]
    vector_dir = vectors_dir / dataset_config["output_dir"]
    pairs = load_pairs(dataset_path)
    needed_words = {word for pair in pairs for word in pair[:2]}

    average_results = {
        method: {eig_weight: [] for _, eig_weight in EIGENVALUE_WEIGHTS}
        for method in METHOD_COLORS
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
    axes = axes.ravel()

    for index, jm_lambda in enumerate(JM_LAMBDAS):
        ax = axes[index]
        method_results = {method: [] for method in METHOD_COLORS}

        for method, eig_weight, filename in vector_specs(dataset_config, jm_lambda):
            vector_path = vector_dir / filename
            if not vector_path.exists():
                message = f"Missing {vector_path}; skipping."
                if strict:
                    raise FileNotFoundError(message)
                print(message)
                continue

            correlation = compute_spearman_for_file(vector_path, pairs, needed_words)
            method_results[method].append((eig_weight, correlation))
            if correlation is not None:
                average_results[method][eig_weight].append(correlation)

        plot_method_results(ax, method_results)
        ax.set_title(f"Jelinek-Mercer lambda={jm_lambda}")

    ax = axes[-1]
    average_method_results = {}
    for method, by_weight in average_results.items():
        average_method_results[method] = [
            (eig_weight, float(np.mean(values)) if values else None)
            for eig_weight, values in by_weight.items()
        ]
    plot_method_results(ax, average_method_results)
    ax.set_title("Average across lambda values")

    fig.suptitle(f"{dataset_config['title']} Spearman correlations", fontsize=16)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / dataset_config["plot_file"]
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def main():
    args = parse_args()
    for dataset_name in resolve_datasets(args.dataset):
        plot_dataset(
            dataset_name=dataset_name,
            datasets_dir=args.datasets_dir,
            vectors_dir=args.vectors_dir,
            results_dir=args.results_dir,
            strict=args.strict,
        )


if __name__ == "__main__":
    main()
