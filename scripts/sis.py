from pathlib import Path
import pandas as pd
import numpy as np
import math
import yaml
import argparse
import sys
from sklearn.feature_selection import mutual_info_regression

def sis_linear(path: Path, alpha: float = 4.0):
    """SIS Linear feature selection using marginal correlation.
    """
    X_normalized = pd.read_parquet(path / 'X_zScoreNormalized.parquet')
    y = pd.read_parquet(path / 'y.parquet')
    
    n = X_normalized.shape[0]
    scalar = float(1 / n)

    cell_ids = X_normalized['cell_id'].copy()
    X_features = X_normalized.drop(columns=['cell_id'])
    y_features = y.drop(columns=['cell_id'])

    X_arr = X_features.to_numpy()
    y_arr = y_features.to_numpy().ravel()

    product = X_arr.T @ y_arr
    product *= scalar

    sorted_idx = np.argsort(-np.abs(product))
    
    d = math.floor(alpha * (n / math.log(n)))
    keep_idx = sorted_idx[:d]

    keep_cols = X_features.columns[keep_idx]
    X_reduced = X_features[keep_cols].copy()
    X_reduced.insert(0, "cell_id", cell_ids.values)

    output_path = path / "X_linear_sis_reduced.parquet"
    X_reduced.to_parquet(output_path, index=False)
    
    print(f"SIS Linear feature selection complete")
    print(f"Original features: {X_features.shape[1]}")
    print(f"Selected features: {d}")

def sis_general(path: Path, alpha: float = 4.0):
    """SIS General feature selection using mutual information.
    """
    X_normalized = pd.read_parquet(path / 'X_zScoreNormalized.parquet')
    y = pd.read_parquet(path / 'y.parquet')
    
    n = X_normalized.shape[0]
    
    cell_ids = X_normalized['cell_id'].copy()
    X_features = X_normalized.drop(columns=['cell_id'])
    y_features = y.drop(columns=['cell_id'])
    
    X_arr = X_features.to_numpy()
    y_arr = y_features.to_numpy().ravel()
    
    mi_scores = mutual_info_regression(X_arr, y_arr, random_state=42)
    sorted_idx = np.argsort(-mi_scores)
    
    d = math.floor(alpha * (n / math.log(n)))
    keep_idx = sorted_idx[:d]
    
    keep_cols = X_features.columns[keep_idx]
    X_reduced = X_features[keep_cols].copy()
    X_reduced.insert(0, "cell_id", cell_ids.values)
    
    output_path = path / "X_general_sis_reduced.parquet"
    X_reduced.to_parquet(output_path, index=False)
    
    print(f"SIS General feature selection complete")
    print(f"Original features: {X_features.shape[1]}")
    print(f"Selected features: {d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SIS feature selection')
    parser.add_argument('--drug', required=True, help='Drug name')
    parser.add_argument('--run-id', required=True, help='Run ID')
    parser.add_argument('--alpha', type=float, help='SIS alpha hyperparameter (overrides config)')
    parser.add_argument('method', choices=['linear', 'general'], help='SIS method: linear or general')
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[1]
    design_matrix_path = project_root / f'data/processed/{args.drug}/{args.run_id}/design_matrix'
    config_path = project_root / 'configs/prep/global.yaml'
    
    if not design_matrix_path.exists():
        print(f"Path does not exist: {design_matrix_path}")
        sys.exit(1)
    
    if not (design_matrix_path / 'X_zScoreNormalized.parquet').exists():
        print(f"✗ X_zScoreNormalized.parquet not found. Run z-score normalization first.")
        sys.exit(1)
    
    alpha = args.alpha
    if alpha is None:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            alpha = config.get('sis', {}).get('alpha', 4.0)
    
    print(f"Running SIS {args.method} for {args.drug} (alpha={alpha})...")
    
    if args.method == 'linear':
        sis_linear(design_matrix_path, alpha=alpha)
    elif args.method == 'general':
        sis_general(design_matrix_path, alpha=alpha)
