#!/usr/bin/env python3
"""
Analyze frequent itemsets from CPSS results:
- Compute pairwise correlations between biomarkers
- Form proxy groups based on high correlation
- Generate visualizations and summary tables
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import argparse
from typing import List, Dict, Tuple, Set
import warnings
warnings.filterwarnings('ignore')


def parse_itemset(itemset_str: str) -> Set[str]:
    """Parse itemset string into set of features."""
    if pd.isna(itemset_str):
        return set()
    # Handle both single and comma-separated itemsets
    items = [item.strip() for item in itemset_str.split(',')]
    return set(items)


def load_frequent_itemsets(filepath: Path, min_support: int = 10) -> pd.DataFrame:
    """Load and filter frequent itemsets."""
    df = pd.read_csv(filepath)
    df = df[df['support'] >= min_support].copy()
    df['features'] = df['itemset'].apply(parse_itemset)
    return df


def get_all_features(itemsets_df: pd.DataFrame) -> List[str]:
    """Extract all unique features from itemsets."""
    all_features = set()
    for features in itemsets_df['features']:
        all_features.update(features)
    return sorted(list(all_features))


def load_design_matrix_features(drug: str, run_id: str = "v1-union-na20") -> pd.DataFrame:
    """Load feature matrix for a drug to compute correlations."""
    base_path = Path("data/processed") / drug / run_id / "design_matrix"
    
    # Try to load X.parquet (original, unnormalized data)
    x_path = base_path / "X.parquet"
    if x_path.exists():
        return pd.read_parquet(x_path)
    
    # Fallback to normalized
    x_norm_path = base_path / "X_zScoreNormalized.parquet"
    if x_norm_path.exists():
        return pd.read_parquet(x_norm_path)
    
    return None


def compute_feature_correlations(features: List[str], 
                                  sample_drugs: List[str] = None,
                                  run_id: str = "v1-union-na20",
                                  max_drugs: int = 50) -> pd.DataFrame:
    """
    Compute pairwise correlations between features across multiple drugs.
    
    Args:
        features: List of feature names to analyze
        sample_drugs: List of drugs to sample from (if None, auto-detect)
        run_id: Run identifier
        max_drugs: Maximum number of drugs to sample
        
    Returns:
        Correlation matrix as DataFrame
    """
    if sample_drugs is None:
        # Auto-detect available drugs
        processed_dir = Path("data/processed")
        available_drugs = [d.name for d in processed_dir.iterdir() 
                          if d.is_dir() and (d / run_id / "design_matrix" / "X.parquet").exists()]
        sample_drugs = np.random.choice(available_drugs, 
                                       size=min(max_drugs, len(available_drugs)), 
                                       replace=False)
    
    print(f"Computing correlations across {len(sample_drugs)} drugs...")
    
    # Collect feature values across drugs
    feature_data = {feat: [] for feat in features}
    
    for drug in sample_drugs:
        X = load_design_matrix_features(drug, run_id)
        if X is None:
            continue
            
        for feat in features:
            if feat in X.columns:
                feature_data[feat].extend(X[feat].values)
    
    # Build correlation matrix
    valid_features = [feat for feat in features if len(feature_data[feat]) > 0]
    
    if len(valid_features) < 2:
        print(f"Warning: Only {len(valid_features)} valid features found")
        return pd.DataFrame()
    
    # Create dataframe with all feature values
    n_samples = min(len(feature_data[feat]) for feat in valid_features)
    df = pd.DataFrame({feat: feature_data[feat][:n_samples] for feat in valid_features})
    
    # Compute correlation matrix
    corr_matrix = df.corr(method='pearson')
    
    print(f"Correlation matrix: {corr_matrix.shape[0]} x {corr_matrix.shape[1]}")
    return corr_matrix


def form_proxy_groups(corr_matrix: pd.DataFrame, 
                      threshold: float = 0.7,
                      method: str = 'complete') -> Dict[int, List[str]]:
    """
    Form proxy groups based on correlation threshold using hierarchical clustering.
    
    Args:
        corr_matrix: Correlation matrix
        threshold: Correlation threshold for grouping
        method: Linkage method ('single', 'complete', 'average', 'ward')
        
    Returns:
        Dictionary mapping group_id -> list of features
    """
    if corr_matrix.empty:
        return {}
    
    # Convert correlation to distance (1 - |r|)
    distance_matrix = 1 - np.abs(corr_matrix.values)
    
    # Hierarchical clustering
    linkage_matrix = linkage(squareform(distance_matrix), method=method)
    
    # Form clusters at given threshold
    distance_threshold = 1 - threshold
    cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')
    
    # Group features by cluster
    groups = {}
    for idx, label in enumerate(cluster_labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(corr_matrix.index[idx])
    
    return groups


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, 
                             output_path: Path,
                             groups: Dict[int, List[str]] = None):
    """Plot correlation heatmap with optional group annotations."""
    if corr_matrix.empty:
        print("Empty correlation matrix, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(max(12, len(corr_matrix) * 0.5), 
                                    max(10, len(corr_matrix) * 0.4)))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Plot heatmap
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=len(corr_matrix) < 30,  # Annotate if not too large
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Pearson Correlation'},
                ax=ax)
    
    ax.set_title('Biomarker Correlation Matrix', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation heatmap to {output_path}")


def plot_dendrogram(corr_matrix: pd.DataFrame, 
                   output_path: Path,
                   threshold: float = 0.7,
                   method: str = 'complete'):
    """Plot hierarchical clustering dendrogram."""
    if corr_matrix.empty:
        print("Empty correlation matrix, skipping dendrogram")
        return
    
    distance_matrix = 1 - np.abs(corr_matrix.values)
    linkage_matrix = linkage(squareform(distance_matrix), method=method)
    
    fig, ax = plt.subplots(figsize=(max(12, len(corr_matrix) * 0.4), 8))
    
    dendrogram(linkage_matrix, 
               labels=corr_matrix.index.tolist(),
               leaf_rotation=90,
               leaf_font_size=8,
               ax=ax)
    
    # Add horizontal line at threshold
    ax.axhline(y=1-threshold, color='r', linestyle='--', linewidth=2, 
               label=f'Threshold (r={threshold})')
    
    ax.set_xlabel('Biomarker', fontsize=12)
    ax.set_ylabel('Distance (1 - |r|)', fontsize=12)
    ax.set_title('Hierarchical Clustering of Biomarkers', fontsize=14, pad=20)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved dendrogram to {output_path}")


def save_proxy_groups(groups: Dict[int, List[str]], 
                     corr_matrix: pd.DataFrame,
                     output_path: Path):
    """Save proxy groups to CSV with statistics."""
    if not groups:
        print("No groups to save")
        return
    
    rows = []
    for group_id, members in groups.items():
        if len(members) == 1:
            # Singleton group
            rows.append({
                'group_id': group_id,
                'size': 1,
                'representative': members[0],
                'members': members[0],
                'avg_intragroup_corr': np.nan,
                'min_intragroup_corr': np.nan,
                'max_intragroup_corr': np.nan
            })
        else:
            # Multi-member group
            # Compute intra-group correlations
            group_corr = corr_matrix.loc[members, members]
            # Extract upper triangle (excluding diagonal)
            triu_indices = np.triu_indices_from(group_corr.values, k=1)
            intra_corr_values = group_corr.values[triu_indices]
            
            # Choose representative (most central = highest avg correlation with group)
            avg_corrs = group_corr.mean(axis=1)
            representative = avg_corrs.idxmax()
            
            rows.append({
                'group_id': group_id,
                'size': len(members),
                'representative': representative,
                'members': ', '.join(sorted(members)),
                'avg_intragroup_corr': np.mean(intra_corr_values),
                'min_intragroup_corr': np.min(intra_corr_values),
                'max_intragroup_corr': np.max(intra_corr_values)
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('size', ascending=False)
    df.to_csv(output_path, index=False)
    
    print(f"Saved {len(groups)} proxy groups to {output_path}")
    print(f"  - Singletons: {sum(1 for g in groups.values() if len(g) == 1)}")
    print(f"  - Multi-member groups: {sum(1 for g in groups.values() if len(g) > 1)}")


def main():
    parser = argparse.ArgumentParser(description='Analyze frequent itemsets and form proxy groups')
    parser.add_argument('--method', type=str, default='both', 
                       choices=['rf', 'linear', 'both'],
                       help='Which method to analyze')
    parser.add_argument('--min-support', type=int, default=10,
                       help='Minimum support for itemsets')
    parser.add_argument('--corr-threshold', type=float, default=0.7,
                       help='Correlation threshold for proxy groups')
    parser.add_argument('--max-drugs', type=int, default=50,
                       help='Maximum number of drugs to sample for correlation')
    parser.add_argument('--output-dir', type=str, default='results/proxy_analysis',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    methods = ['rf', 'linear'] if args.method == 'both' else [args.method]
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Analyzing {method.upper()} method")
        print(f"{'='*80}\n")
        
        # Load frequent itemsets
        itemsets_file = Path(f"results/frequent_itemsets_{method}.csv")
        if not itemsets_file.exists():
            print(f"File not found: {itemsets_file}")
            continue
        
        itemsets_df = load_frequent_itemsets(itemsets_file, args.min_support)
        print(f"Loaded {len(itemsets_df)} itemsets with support >= {args.min_support}")
        
        # Extract all features
        features = get_all_features(itemsets_df)
        print(f"Found {len(features)} unique biomarkers")
        
        if len(features) < 2:
            print("Not enough features for correlation analysis")
            continue
        
        # Compute correlations
        corr_matrix = compute_feature_correlations(
            features, 
            max_drugs=args.max_drugs
        )
        
        if corr_matrix.empty:
            print("Could not compute correlation matrix")
            continue
        
        # Form proxy groups
        print(f"\nForming proxy groups (threshold={args.corr_threshold})...")
        groups = form_proxy_groups(corr_matrix, threshold=args.corr_threshold)
        
        # Save results
        method_dir = output_dir / method
        method_dir.mkdir(exist_ok=True, parents=True)
        
        # Save correlation matrix
        corr_matrix.to_csv(method_dir / "correlation_matrix.csv")
        print(f"Saved correlation matrix: {method_dir / 'correlation_matrix.csv'}")
        
        # Save proxy groups
        save_proxy_groups(groups, corr_matrix, method_dir / "proxy_groups.csv")
        
        # Generate plots
        plot_correlation_heatmap(corr_matrix, method_dir / "correlation_heatmap.png", groups)
        plot_dendrogram(corr_matrix, method_dir / "dendrogram.png", 
                       threshold=args.corr_threshold)
        
        # Summary statistics
        print(f"\n{method.upper()} Summary:")
        print(f"  Total biomarkers: {len(features)}")
        print(f"  Proxy groups: {len(groups)}")
        print(f"  Average group size: {np.mean([len(g) for g in groups.values()]):.2f}")
        print(f"  Largest group: {max(len(g) for g in groups.values())}")
        
        # Show highly correlated pairs
        print(f"\nTop 10 highly correlated pairs (r > 0.8):")
        high_corr = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.8:
                    high_corr.append((
                        corr_matrix.index[i],
                        corr_matrix.index[j],
                        r
                    ))
        
        high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
        for feat1, feat2, r in high_corr[:10]:
            print(f"  {feat1} <-> {feat2}: r={r:.3f}")
    
    print(f"\n{'='*80}")
    print(f"Analysis complete. Results saved to {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
