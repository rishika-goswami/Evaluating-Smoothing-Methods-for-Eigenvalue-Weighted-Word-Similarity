import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr

# Define the file structure for different smoothing methods
smoothing_methods_template = {
    "Bayesian": {
        "op_men_b_eg0.txt": 0.0,
        "op_men_b_eg02.txt": 0.2,
        "op_men_b_eg04.txt": 0.4,
        "op_men_b_eg06.txt": 0.6,
        "op_men_b_eg08.txt": 0.8,
        "op_men_b_eg1.txt": 1.0
    },
    "Dirichlet": {
        "op_men_d_eg0.txt": 0.0,
        "op_men_d_eg02.txt": 0.2,
        "op_men_d_eg04.txt": 0.4,
        "op_men_d_eg06.txt": 0.6,
        "op_men_d_eg08.txt": 0.8,
        "op_men_d_eg1.txt": 1.0
    }
}

# Define lambda values and corresponding filenames for Jelinek-Mercer
jm_lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]
jm_file_template = "op_men_jm{:02d}_eg{}.txt"

# Load MEN dataset
men_file = "men"
men_pairs = []

with open(men_file, "r") as f:
    for line in f:
        word1, word2, score = line.strip().split()  # Assuming space-separated
        men_pairs.append((word1, word2, float(score)))

# Prepare subplots (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# Define colors for the three methods
method_colors = {
    "Bayesian": "blue",
    "Dirichlet": "red",
    "Jelinek-Mercer": "green"
}

# Initialize average results including Jelinek-Mercer
all_methods = list(smoothing_methods_template.keys()) + ["Jelinek-Mercer"]
average_results = {method: {ew: [] for ew in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]} for method in all_methods}

# Process each lambda value and plot its graph
for idx, lambda_value in enumerate(jm_lambdas):
    smoothing_methods = smoothing_methods_template.copy()

    # Add Jelinek-Mercer smoothing with the correct lambda value
    jm_files = {
        jm_file_template.format(int(lambda_value * 10), ew): float(ew) / 10 if ew != "1" else 1.0
        for ew in ["0", "02", "04", "06", "08", "1"]
    }
    smoothing_methods["Jelinek-Mercer"] = jm_files

    results = {method: [] for method in smoothing_methods}

    # Process each smoothing method
    for method, file_info in smoothing_methods.items():
        for filename, eig_weight in file_info.items():
            if not os.path.exists(filename):
                print(f"Skipping {filename} for {method} (File not found)")
                continue

            with open(filename, "r") as f:
                lines = f.readlines()

            num_words, num_dims = map(int, lines[0].split())

            word_embeddings = {}
            for line in lines[1:num_words + 1]:
                parts = line.split()
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]])
                word_embeddings[word] = vector

            human_ranks = []
            computed_similarities = []

            for word1, word2, human_score in men_pairs:
                if word1 in word_embeddings and word2 in word_embeddings:
                    vec1 = word_embeddings[word1]
                    vec2 = word_embeddings[word2]
                    cosine_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

                    human_ranks.append(human_score)
                    computed_similarities.append(cosine_similarity)

            if len(human_ranks) > 0 and len(computed_similarities) > 0:
                spearman_corr, _ = spearmanr(human_ranks, computed_similarities)
                results[method].append((eig_weight, spearman_corr))
                average_results[method][eig_weight].append(spearman_corr)
            else:
                results[method].append((eig_weight, None))

    for method in results:
        results[method].sort(key=lambda x: x[0])

    ax = axes[idx // 3, idx % 3]

    for method, result_list in results.items():
        eigenvalue_weights = [x[0] for x in result_list]
        spearman_values = [x[1] if x[1] is not None else 0 for x in result_list]
        ax.plot(eigenvalue_weights, spearman_values, 'o-', label=method, color=method_colors[method])

    ax.set_xticks(eigenvalue_weights)
    ax.set_xticklabels([str(x) for x in eigenvalue_weights])
    ax.set_xlabel("Eigenvalue Weighting")
    ax.set_ylabel("Spearman's ρ")
    ax.set_title(f"Jelinek-Mercer λ={lambda_value}")
    ax.legend()
    ax.grid()

# Compute the **average graph** and plot it in the **6th empty space**
ax = axes[1, 2]  # Bottom-right empty space

for method in average_results:
    eigenvalue_weights = list(average_results[method].keys())
    average_spearman_values = [np.mean(average_results[method][ew]) if average_results[method][ew] else 0 for ew in eigenvalue_weights]
    ax.plot(eigenvalue_weights, average_spearman_values, 'o-', label=method, color=method_colors[method])

ax.set_xticks(eigenvalue_weights)
ax.set_xticklabels([str(x) for x in eigenvalue_weights])
ax.set_xlabel("Eigenvalue Weighting")
ax.set_ylabel("Spearman's ρ")
ax.set_title("Average of All λ Graphs")
ax.legend()
ax.grid()

# Save as a .png file
output_file = "spearman_correlation_men_avg.png"
plt.savefig(output_file)
plt.close()

print(f"Plot saved as {output_file}")
