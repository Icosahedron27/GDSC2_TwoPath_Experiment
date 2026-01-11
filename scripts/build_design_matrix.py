"""Build design matrix (X, y) for a single drug from interim blocks."""
import sys
import json
import math
import argparse
from collections import Counter
from pathlib import Path
import yaml
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts.prep.artifacts import save_table, save_manifest, save_sidecars

BLOCK_PREFIX = {
    'tpm': 'TPM',
    'cnv': 'CNV',
    'mut': 'MUT',
    'methyl': 'METH'
}


def load_interim_block(path: Path) -> pd.DataFrame:
    """Load parquet block and restore cell_id as index."""
    if not path.exists():
        return None
    df = pd.read_parquet(path, engine='pyarrow')
    if 'cell_id' in df.columns:
        df = df.set_index('cell_id')
    return df


def build_y(ic50_df: pd.DataFrame, drug: str, target: str) -> pd.DataFrame:
    """Filter IC50 for single drug and build y."""
    if drug not in ic50_df.columns:
        raise ValueError(f"Drug {drug} not found in IC50 data")
    
    y = ic50_df[[drug]].dropna()
    y.columns = [target]
    return y


def merge_blocks(blocks: dict, policy: str = 'union') -> tuple[pd.DataFrame, list[dict]]:
    """Merge omics blocks by union of features (cell intersection)."""
    if policy != 'union':
        raise NotImplementedError(f"Policy {policy} not implemented")

    valid_blocks = {k: v for k, v in blocks.items() if v is not None}
    if not valid_blocks:
        raise ValueError("No valid blocks found")

    block_iter = iter(valid_blocks.values())
    first_block = next(block_iter)
    common_cells = first_block.index
    for df in block_iter:
        common_cells = common_cells.intersection(df.index)

    feature_records: list[dict] = []
    parts = []
    for name, df in valid_blocks.items():
        df_subset = df.loc[common_cells].copy()
        prefix = BLOCK_PREFIX.get(name, name.upper())
        renamed_cols = [f"{prefix}__{col}" for col in df_subset.columns]
        df_subset.columns = renamed_cols
        parts.append(df_subset)
        feature_records.extend(
            {
                'feature': new_col,
                'block': prefix,
                'gene': orig_col
            }
            for new_col, orig_col in zip(renamed_cols, df.columns)
        )

    if not parts:
        return pd.DataFrame(index=common_cells), feature_records

    X = pd.concat(parts, axis=1)
    return X, feature_records


def qc_features(X: pd.DataFrame, block_lookup: dict, max_na_frac: float, drop_nzv: bool) -> pd.DataFrame:
    """Per-drug QC: drop high-NA and near-zero-variance features."""
    # Drop features with >max_na_frac missing
    na_frac = X.isna().mean(axis=0)
    keep_cols = na_frac[na_frac <= max_na_frac].index
    X = X[keep_cols]
    
    # Drop near-zero variance
    if drop_nzv:
        var = X.var(axis=0)
        keep_cols = var[var > 1e-6].index
        X = X[keep_cols]
    
    # Median imputation (clip methylation to [0,1])
    for col in X.columns:
        col_series = X[col]
        if col_series.isna().any():
            median_val = col_series.median()
            X[col] = col_series.fillna(median_val)
        if block_lookup.get(col) == 'METH':
            X[col] = X[col].clip(0.0, 1.0)
    
    return X


def filter_tpm_by_expression(X: pd.DataFrame, min_tpm: float, min_frac: float) -> tuple[pd.DataFrame, list[str]]:
    """Drop TPM features that fail the per-drug expression threshold."""
    if X.empty or min_frac <= 0:
        return X, []

    tpm_cols = [col for col in X.columns if col.startswith("TPM__")]
    if not tpm_cols:
        return X, []

    min_count = math.ceil(len(X) * min_frac)
    meets_threshold = (X[tpm_cols] >= min_tpm).sum(axis=0) >= min_count
    drop_cols = [col for col, keep in meets_threshold.items() if not keep]

    if drop_cols:
        X = X.drop(columns=drop_cols)

    return X, drop_cols


def filter_by_sparsity(X: pd.DataFrame, block_prefix: str, min_frac: float) -> tuple[pd.DataFrame, list[str]]:
    """Drop features with too few non-zero values (sparsity filter)."""
    if X.empty or min_frac <= 0:
        return X, []

    block_cols = [col for col in X.columns if col.startswith(f"{block_prefix}__")]
    if not block_cols:
        return X, []

    min_count = math.ceil(len(X) * min_frac)
    non_zero_counts = (X[block_cols] != 0).sum(axis=0)
    drop_cols = [col for col, count in non_zero_counts.items() if count < min_count]

    if drop_cols:
        X = X.drop(columns=drop_cols)

    return X, drop_cols


def main():
    parser = argparse.ArgumentParser(description='Build design matrix for drug')
    parser.add_argument('--drug', required=True, help='Drug name (column in IC50)')
    parser.add_argument('--run-id', required=True, help='Run identifier')
    args = parser.parse_args()
    
    # Load configs
    cfg_dir = project_root / "configs" / "prep"
    with open(cfg_dir / "global.yaml") as f:
        global_cfg = yaml.safe_load(f)
    with open(cfg_dir / "blocks.yaml") as f:
        blocks_cfg = yaml.safe_load(f)
    
    # Paths
    interim_dir = project_root / "data" / "interim" / "transformed"
    outdir = project_root / "data" / "processed" / args.drug / args.run_id / "design_matrix"
    
    # Check if output exists
    if outdir.exists():
        print(f"✗ Output directory already exists: {outdir}")
        print("  Delete it first or use a different --run-id")
        sys.exit(1)
    
    # Load interim blocks
    print("Loading interim blocks...")
    tpm = load_interim_block(interim_dir / "tpm.parquet")
    cnv = load_interim_block(interim_dir / "cnv.parquet")
    mut = load_interim_block(interim_dir / "mut.parquet")
    methyl = load_interim_block(interim_dir / "methyl.parquet")
    ic50 = load_interim_block(interim_dir / "ic50.parquet")
    
    # Build y
    print(f"Building y for drug: {args.drug}")
    target = global_cfg.get('target', 'ln_ic50')
    y = build_y(ic50, args.drug, target)
    
    # Get cell intersection with y
    print("Merging omics blocks...")
    blocks = {'tpm': tpm, 'cnv': cnv, 'mut': mut, 'methyl': methyl}
    feature_policy = global_cfg.get('feature_policy', 'union')
    X_full, feature_records = merge_blocks(blocks, policy=feature_policy)
    
    # Intersect with y
    common_cells = y.index.intersection(X_full.index)
    print(f"  Common cells: {len(common_cells)}")
    
    X = X_full.loc[common_cells]
    y = y.loc[common_cells]

    # Apply per-drug TPM expression filter before QC.
    tpm_filter_cfg = (blocks_cfg or {}).get("tpm_filter", {})
    min_tpm = float(tpm_filter_cfg.get("min_tpm", 0.0))
    min_frac = float(tpm_filter_cfg.get("min_fraction", 0.0))
    X, dropped_tpm = filter_tpm_by_expression(X, min_tpm, min_frac)
    if min_frac > 0:
        kept = X.shape[1]
        print(
            f"TPM filter removed {len(dropped_tpm)} features; "
            f"kept {kept} with TPM ≥ {min_tpm} in ≥ {min_frac:.0%} cells"
        )
    
    # Apply mutation sparsity filter
    mut_filter_cfg = (blocks_cfg or {}).get("mut_filter", {})
    mut_min_frac = float(mut_filter_cfg.get("min_fraction", 0.0))
    X, dropped_mut = filter_by_sparsity(X, "MUT", mut_min_frac)
    if mut_min_frac > 0:
        print(
            f"MUT filter removed {len(dropped_mut)} features; "
            f"kept features with ≥ {mut_min_frac:.0%} non-zero cells"
        )
    
    # Apply CNV sparsity filter
    cnv_filter_cfg = (blocks_cfg or {}).get("cnv_filter", {})
    cnv_min_frac = float(cnv_filter_cfg.get("min_fraction", 0.0))
    X, dropped_cnv = filter_by_sparsity(X, "CNV", cnv_min_frac)
    if cnv_min_frac > 0:
        print(
            f"CNV filter removed {len(dropped_cnv)} features; "
            f"kept features with ≥ {cnv_min_frac:.0%} non-zero cells"
        )
    
    # QC
    print("Running per-drug QC...")
    qc_cfg = global_cfg.get('per_drug_qc', {})
    max_na = qc_cfg.get('max_na_frac', 0.2)
    drop_nzv = qc_cfg.get('drop_near_zero_var', True)
    
    n_features_before = X.shape[1]
    block_lookup = {rec['feature']: rec['block'] for rec in feature_records}
    X = qc_features(X, block_lookup, max_na, drop_nzv)
    n_features_after = X.shape[1]
    print(f"  Features: {n_features_before} → {n_features_after}")
    
    # Feature metadata
    feature_meta = pd.DataFrame(feature_records)
    if not feature_meta.empty:
        feature_meta = feature_meta.set_index('feature').loc[X.columns].reset_index()
    else:
        feature_meta = pd.DataFrame(columns=['feature', 'block', 'gene'])
    
    # Save outputs
    print(f"Saving to {outdir}...")
    outdir.mkdir(parents=True, exist_ok=True)
    
    dtype = global_cfg.get('dtype', 'float32')
    if n_features_after:
        X = X.astype(dtype)
    y[target] = y[target].astype(dtype)

    save_table(X.reset_index(names='cell_id'), outdir / "X.parquet")
    save_table(y.reset_index(names='cell_id'), outdir / "y.parquet")
    save_table(feature_meta, outdir / "feature_meta.parquet")
    
    # Sidecars
    sidecar_dir = outdir / "sidecars"
    sidecar_dir.mkdir(exist_ok=True)
    save_sidecars(X, sidecar_dir)
    
    # Manifest
    block_counts = Counter(block_lookup.get(col, "UNKNOWN") for col in X.columns)
    manifest = {
        "drug": args.drug,
        "run_id": args.run_id,
        "target": target,
        "n_cells": len(common_cells),
        "n_features": X.shape[1],
        "qc": {
            "max_na_frac": max_na,
            "drop_near_zero_var": drop_nzv,
            "features_before": n_features_before,
            "features_after": n_features_after
        },
        "feature_policy": feature_policy,
        "blocks": {name: block_counts.get(name, 0) for name in BLOCK_PREFIX.values()}
    }
    save_manifest(manifest, outdir / "manifest.json")
    
    print(f"\n✓ Design matrix saved")
    print(f"  Cells: {manifest['n_cells']}")
    print(f"  Features: {manifest['n_features']}")
    print(f"  Blocks: {manifest['blocks']}")


if __name__ == "__main__":
    main()
