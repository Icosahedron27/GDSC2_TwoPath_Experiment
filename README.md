# GDSC2 Two-Path Feature Selection

- Two-path feature selection for drug response prediction
- SIS + CPSS for biomarker identification
- GDSC2 pharmacogenomic data

## Overview

- Multi-omics: RNA-Seq, CNV, mutations, methylation
- Drug response: IC50 values from cancer cell lines
- Linear path: Pearson-Correlation-based
- Non-linear path: Mutual information
- Data: GDSC2, GEO GSE68379

## Quick Start

# Run full pipeline (preprocessing through feature selection)
snakemake --cores 4

# Optional: Run FDR analysis on subset of drugs
snakemake -s Snakefile_fdr_simple --cores 2


## Pipeline Steps
1. Preprocessing: Transform raw data into harmonized Parquet blocks
2. Build design matrices: Create (X, y) pairs per drug with QC filtering
3. Z-score normalization: Standardize features
4. SIS: Reduce dimensions
5. CPSS: Stability selection over B=50 complementary subsample pairs
6. Frequent itemsets: Mine common features across drugs using Apriori (min_support=0.1)
7. FDR analysis: Permutation-based false discovery rate estimation


## Requirements
python >= 3.11
pandas, numpy, scikit-learn, pyarrow
snakemake, pyyaml, scipy, diptest, mlxtend
```

## Configuration
`configs/prep/global.yaml`:
- run_tag: Experiment ID (v1-union-na20)
- threads: 8
- sis.alpha: 4.0
- per_drug_qc.max_na_frac: 0.2
- design_drugs_subset: Optional drug filtering

## Key Parameters
SIS:
- alpha=4.0 → d = n/4 features
- Linear: |cor(Xj, y)|
- General: Mutual information

CPSS:
- B=50 pairs (100 models)
- Threshold: τ = sqrt(2*log(p)/B)
- Worst-case: q = 0.75*B
- Linear: ElasticNet (α=0.5, λ=0.01)
- RF: RandomForest (100 trees, depth=5)

QC:
- max_na_frac: 0.2
- near_zero_var: σ² < 1e-6
- TPM: median > 1.0
- Sparsity: min 5 non-zero
