# Evaluating Smoothing Techniques for Eigenvalue-Weighted Word Similarity

This repository contains the experiment code for my paper on smoothing methods for eigenvalue-weighted word embeddings. We compare Dirichlet, Bayesian, and Jelinek-Mercer smoothing across common word-similarity benchmarks using Spearman rank correlation, while varying the eigenvalue weighting factor and the Jelinek-Mercer interpolation parameter. The results show that smoothing can reduce sparsity effects in PPMI/SVD word embeddings, with Bayesian smoothing often giving the strongest correlations in these experiments.

## Author

- Rishika Goswami
- Email: goswami.rishika67@gmail.com
- GitHub: https://github.com/rishika-goswami
- Google Scholar: https://scholar.google.com/citations?user=zPaoDEkAAAAJ&hl=en
- Paper: https://link.springer.com/chapter/10.1007/978-3-032-13555-1_11
- DOI: https://doi.org/10.1007/978-3-032-13555-1_11

## Cite This Paper

Saha, A., Goswami, R. (2026). Evaluating Smoothing Techniques for Eigenvalue-Weighted Word Similarity. In: Subirats, L., Gurung, S., Ningombam, D., Banerji, N. (eds) Advanced Computational and Communication Paradigms. ICACCP 2025. Lecture Notes in Networks and Systems, vol 1761. Springer, Cham. https://doi.org/10.1007/978-3-032-13555-1_11

## Repository Layout

`compute_svd_ppmi_lambda_vectors.py` generates Dirichlet-smoothed PPMI/SVD vectors.

`compute_svd_ppmi_lambda_vectors_b.py` generates Bayesian-smoothed PPMI/SVD vectors.

`compute_svd_ppmi_lambda_vectors_jm.py` generates Jelinek-Mercer-smoothed PPMI/SVD vectors.

`plot_spearman_correlations.py` creates Spearman correlation plots for all five benchmarks or a selected benchmark.

`datasets/` contains the local benchmark word-pair files and is ignored by git.

`outputs/` is the generated vector-output directory and is ignored by git.

`results/` is the generated plot-output directory and is ignored by git.

## Setup

Run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
```

The embedding scripts use NLTK tokenization and stopword removal. They attempt to download the small required NLTK resources automatically if they are missing.

## Execution Order

1. Install dependencies.
2. Place the benchmark word-pair files under `datasets/` if they are not already present locally.
3. Generate vector files with one or more smoothing scripts into `outputs/<dataset>/`.
4. Generate Spearman plots from the vector files into `results/`.
5. Keep `datasets/`, `outputs/`, and `results/` out of git.

## Generate One Vector File

Dirichlet smoothing:

```bash
python compute_svd_ppmi_lambda_vectors.py datasets/men outputs/men/op_men_d_eg04.txt --eigenvalue_weighting 0.4 --smoothing_factor 0.0001 --dimensions 100 --verbose
```

Bayesian smoothing:

```bash
python compute_svd_ppmi_lambda_vectors_b.py datasets/men outputs/men/op_men_b_eg04.txt --eigenvalue_weighting 0.4 --bayesian_constant 0.0001 --dimensions 100 --verbose
```

Jelinek-Mercer smoothing:

```bash
python compute_svd_ppmi_lambda_vectors_jm.py datasets/men outputs/men/op_men_jm05_eg04.txt --eigenvalue_weighting 0.4 --jm_lambda 0.5 --dimensions 100 --verbose
```

## Generate the Full Experiment Grid

The plotter expects this file naming pattern:

`op_<prefix>_b_eg<tag>.txt` for Bayesian smoothing.

`op_<prefix>_d_eg<tag>.txt` for Dirichlet smoothing.

`op_<prefix>_jm<lambda_tag>_eg<tag>.txt` for Jelinek-Mercer smoothing.

Dataset prefixes are `sl` for SimLex-999, `men` for MEN, `ws` for WordSim-353, `rg` for RG-65, and `rw` for Rare Words.

Use this bash/WSL loop to regenerate the complete grid:

```bash
declare -A DATASET_FILES=(
  [simlex999]=datasets/simlex999_preprocessed.txt
  [men]=datasets/men
  [wordsim353]=datasets/wordsim_similarity_goldstandard.txt
  [rg-65]=datasets/rg_processed.txt
  [rw]=datasets/rw_processed.txt
)

declare -A PREFIX=(
  [simlex999]=sl
  [men]=men
  [wordsim353]=ws
  [rg-65]=rg
  [rw]=rw
)

for dataset in simlex999 men wordsim353 rg-65 rw; do
  mkdir -p "outputs/$dataset"

  for item in 0:0 02:0.2 04:0.4 06:0.6 08:0.8 1:1.0; do
    tag="${item%%:*}"
    eig="${item##*:}"

    python compute_svd_ppmi_lambda_vectors_b.py "${DATASET_FILES[$dataset]}" "outputs/$dataset/op_${PREFIX[$dataset]}_b_eg${tag}.txt" -e "$eig" -b 0.0001 -d 100
    python compute_svd_ppmi_lambda_vectors.py "${DATASET_FILES[$dataset]}" "outputs/$dataset/op_${PREFIX[$dataset]}_d_eg${tag}.txt" -e "$eig" -l 0.0001 -d 100

    for lambda_item in 01:0.1 03:0.3 05:0.5 07:0.7 09:0.9; do
      lambda_tag="${lambda_item%%:*}"
      lambda_value="${lambda_item##*:}"
      python compute_svd_ppmi_lambda_vectors_jm.py "${DATASET_FILES[$dataset]}" "outputs/$dataset/op_${PREFIX[$dataset]}_jm${lambda_tag}_eg${tag}.txt" -e "$eig" -j "$lambda_value" -d 100
    done
  done
done
```

## Generate Spearman Plots

Plot all five datasets:

```bash
python plot_spearman_correlations.py --dataset all
```

Plot one dataset:

```bash
python plot_spearman_correlations.py --dataset men
python plot_spearman_correlations.py --dataset simlex999
python plot_spearman_correlations.py --dataset wordsim353
python plot_spearman_correlations.py --dataset rg-65
python plot_spearman_correlations.py --dataset rw
```

Useful aliases are `simlex`, `wordsim`, `rg`, and `rarewords`.

Custom paths:

```bash
python plot_spearman_correlations.py --dataset all --datasets-dir datasets --vectors-dir outputs --results-dir results
```

Use `--strict` if you want the plotter to stop immediately when an expected vector file is missing.
