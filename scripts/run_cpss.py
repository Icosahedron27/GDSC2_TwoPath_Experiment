#!/usr/bin/env python3
"""
Snakemake wrapper for CPSS feature selection.
Runs cpss.py and saves outputs to drug-specific results directory.
"""
import argparse
from pathlib import Path
import shutil
from cpss import cpss_feature_selection

def main():
    parser = argparse.ArgumentParser(description="Run CPSS feature selection")
    parser.add_argument("--drug", required=True, help="Drug name")
    parser.add_argument("--run-id", required=True, help="Run ID")
    parser.add_argument("--method", required=True, choices=["linear", "rf"], help="Method: linear or rf")
    parser.add_argument("--B", type=int, default=10, help="Number of complementary pairs")
    args = parser.parse_args()

    linear = (args.method == "linear")
    matrix_name = "X_linear_sis_reduced.parquet" if linear else "X_general_sis_reduced.parquet"
    
    design_matrix_path = Path(f"data/processed/{args.drug}/{args.run_id}/design_matrix")
    results_path = Path(f"results/{args.drug}/{args.run_id}")
    results_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"CPSS: {args.drug} | Method: {args.method} | B={args.B}")
    print(f"{'='*80}")
    
    cpss_scores, important_features = cpss_feature_selection(
        design_matrix_path, 
        matrix_name, 
        B=args.B, 
        linear=linear
    )
    
    # Move generated files to results directory
    method_suffix = "linear" if linear else "rf"
    
    # cpss.py creates files in current working directory (repository root)
    file_mapping = {
        f"cpss_scores_{method_suffix}.csv": f"cpss_{method_suffix}_scores.csv",
        f"Significant_features{method_suffix}.csv": f"cpss_{method_suffix}_significant.csv",
        f"features_above_threshold_{method_suffix}.csv": f"cpss_{method_suffix}_above_threshold.csv",
        f"features_above_worst_case_{method_suffix}.csv": f"cpss_{method_suffix}_above_worst_case.csv",
        f"cpss_bounds_{method_suffix}.json": f"cpss_{method_suffix}_bounds.json"
    }
    
    for src_name, dst_name in file_mapping.items():
        src = Path(src_name)  # Files are in CWD, not scripts/
        dst = results_path / dst_name
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"  → {dst}")
        else:
            print(f"  ✗ {src} not found!")
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
