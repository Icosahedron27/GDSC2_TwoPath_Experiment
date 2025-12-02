"""Compare CPSS results between linear and RF paths."""
import pandas as pd
import argparse
from pathlib import Path
import json


def compare_feature_sets(linear_features: set, rf_features: set, 
                         linear_df: pd.DataFrame, rf_df: pd.DataFrame, 
                         output_dir: Path):
    
    rf_only = rf_features - linear_features
    rf_only_df = rf_df[rf_df['feature'].isin(rf_only)].copy()
    rf_only_df.to_csv(output_dir / "rf_only_features.csv", index=False)
    
    intersection = rf_features & linear_features
    intersection_df = rf_df[rf_df['feature'].isin(intersection)].copy()
    linear_scores = linear_df.set_index('feature')['cpss_score'].to_dict()
    intersection_df['cpss_score_linear'] = intersection_df['feature'].map(linear_scores)
    intersection_df.to_csv(output_dir / "intersection_features.csv", index=False)
    
    linear_only = linear_features - rf_features
    linear_only_df = linear_df[linear_df['feature'].isin(linear_only)].copy()
    linear_only_df.to_csv(output_dir / "linear_only_features.csv", index=False)
    
    union_features = rf_features | linear_features
    union_df = pd.concat([
        rf_df[rf_df['feature'].isin(union_features)],
        linear_df[linear_df['feature'].isin(linear_only)]
    ]).drop_duplicates(subset=['feature'])
    union_df['in_rf'] = union_df['feature'].isin(rf_features)
    union_df['in_linear'] = union_df['feature'].isin(linear_features)
    union_df.to_csv(output_dir / "union_features.csv", index=False)
    
    summary = {
        "n_rf_only": len(rf_only),
        "n_linear_only": len(linear_only),
        "n_intersection": len(intersection),
        "n_union": len(union_features),
        "jaccard_index": len(intersection) / len(union_features) if union_features else 0
    }
    
    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    
    results_dir = Path("results") / args.drug / args.run_id
    linear_df = pd.read_csv(results_dir / "cpss_linear_above_threshold.csv")
    rf_df = pd.read_csv(results_dir / "cpss_rf_above_threshold.csv")
    
    linear_features = set(linear_df['feature'])
    rf_features = set(rf_df['feature'])
    
    compare_feature_sets(linear_features, rf_features, linear_df, rf_df, results_dir)


if __name__ == "__main__":
    main()
