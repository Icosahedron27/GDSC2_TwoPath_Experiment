# GDSC2 Two-Path Feature Selection

- Two-path feature selection for drug response prediction
- SIS + CPSS for biomarker identification
- GDSC2 pharmacogenomic data

## Overview

- Multi-omics: RNA-Seq, CNV, mutations, methylation
- Drug response: IC50 values from cancer cell lines
- Linear path: marginal association via |Xᵀy|/n (X is z-scored)
- Non-linear path: Mutual information
- Data: GDSC2, GEO GSE68379

## Quick Start

# Run full pipeline (preprocessing through feature selection)
snakemake --cores 4

# Optional: Run FDR analysis on subset of drugs
snakemake -s Snakefile_fdr_simple --cores 4


## Pipeline Steps
1. Preprocessing: Transform raw data into harmonized Parquet blocks
2. Build design matrices: Create (X, y) pairs per drug with QC filtering
3. Z-score normalization: Standardize features
4. SIS: Reduce dimensions to d = floor(alpha * (n / log(n)))
5. CPSS: Stability selection over B=50 complementary subsample pairs
6. Frequent itemsets: Mine common features across drugs (min_support=2 drugs, max_size=3)
7. FDR analysis: Permutation-based false discovery rate estimation


## Requirements
python >= 3.11
pandas, numpy, scikit-learn, pyarrow
snakemake, pyyaml, scipy, diptest

## Configuration
`configs/prep/paths.yaml` (Snakemake `configfile`):
- root: base path to raw inputs
- Optional: design_drugs_subset (comma-separated string or list)

`configs/prep/global.yaml`:
- run_tag: Experiment ID (default: v1-union-na20)
- threads: 8
- target: ln_ic50
- per_drug_qc.max_na_frac: 0.2
- per_drug_qc.drop_near_zero_var: true
- sis.alpha: 4.0

`configs/prep/blocks.yaml`:
- tpm_filter.min_tpm: 1.0
- tpm_filter.min_fraction: 0.2
- mut_filter.min_fraction: 0.02
- cnv_filter.min_fraction: 0.05
- cpss.alpha_grid: [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50] (currently only the first value is used)

## Key Parameters
SIS:
- alpha=4.0 → d = floor(alpha * (n / log(n)))
- Linear: |(1/n) Xjᵀy| on z-scored X
- General: Mutual information

CPSS:
- B=50 pairs (100 models)
- Error control level: l = int(0.05 * d)
- Target q: q = int(sqrt(0.8 * l * d))
- Feature score: CPSS Score = selection_count / (2B)
- Thresholding: per-feature (unimodal τ* if diptest p>0.05 else worst-case)
- Worst-case bound: τ* = min(1, 0.5 * (1 + q̂²/(l*d)))
- Linear: ElasticNet (l1_ratio fixed to cpss.alpha_grid[0]=0.95; no l1_ratio grid search; λ via data-driven lambda grid)
- RF: RandomForest (n_estimators=100, max_depth=12, max_features='sqrt', min_samples_split=10, min_samples_leaf=5)

QC:
- max_na_frac: 0.2
- near_zero_var: σ² < 1e-6
- TPM expression filter: keep TPM__* with TPM ≥ 1.0 in ≥ 20% of cells
- Sparsity filters: keep MUT__* in ≥ 2% non-zero cells; CNV__* in ≥ 5% non-zero cells
- Missing values: per-feature median imputation (METH__* clipped to [0,1])

Frequent Itemsets:
- min_support: 2 drugs (absolute count)
- max_size: 3 features per itemset
- top_k (Snakefile): 50
