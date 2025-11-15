from pathlib import Path
import pandas as pd
import math


def computeMeanDeviation(df: pd.DataFrame):
    mean = df.mean(axis=0)
    deviation = df.std(axis=0, ddof=0)
    return mean, deviation


def computeZScore(mean: float, deviation: float, xValue) -> float:
    """Standardize: (x - mean) / std"""
    if deviation == 0:
        return 0.0
    return float((xValue - mean) / deviation)


def normalizeDesignMatrix(path: Path):
    """Standardize X features"""
    X = pd.read_parquet(path / 'X.parquet')

    cell_ids = X['cell_id'].copy()
    X_features = X.drop(columns=['cell_id'])
    
    meansX, deviationX = computeMeanDeviation(X_features)
    
    X_normalized = (X_features - meansX) / deviationX
    X_normalized = X_normalized.fillna(0.0)
    X_normalized.insert(0, 'cell_id', cell_ids.values)

    output_path = path / "X_zScoreNormalized.parquet"
    X_normalized.to_parquet(output_path, index=False)
    
    print(f"Standardized X saved to: {output_path}")
    print(f"Shape: {X_normalized.shape}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Z-score normalize design matrix')
    parser.add_argument('--drug', required=True, help='Drug name')
    parser.add_argument('--run-id', required=True, help='Run ID')
    args = parser.parse_args()
    
    path = Path(f'data/processed/{args.drug}/{args.run_id}/design_matrix')
    
    if not path.exists():
        print(f"Path does not exist: {path}")
        exit(1)
    
    print(f"Normalizing design matrix for {args.drug}...")
    normalizeDesignMatrix(path)
