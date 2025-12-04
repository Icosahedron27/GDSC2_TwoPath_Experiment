"""
Analyze FDR validation results.
Creates publication-ready summary tables and visualizations.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load combined results
    combined_path = Path("results/fdr_combined_summary.csv")
    
    if not combined_path.exists():
        print(f"ERROR: {combined_path} not found. Run Snakefile_fdr_simple first.")
        return
    
    df = pd.read_csv(combined_path)
    
    print("\n" + "="*70)
    print("FDR VALIDATION RESULTS")
    print("="*70)
    
    # Overall statistics
    print(f"\nTotal analyses: {len(df)}")
    print(f"Drugs tested: {df['drug'].nunique()}")
    print(f"Methods: {', '.join(df['method'].unique())}")
    print(f"Permutations per analysis: {df['n_perms'].iloc[0]}")
    print(f"CPSS pairs (B): {df['B'].iloc[0]}")
    
    # FDR by method
    print("\n" + "-"*70)
    print("FDR BY METHOD")
    print("-"*70)
    
    for method in ['linear', 'rf']:
        subset = df[df['method'] == method]
        if len(subset) == 0:
            continue
            
        print(f"\n{method.upper()}:")
        print(f"  Drugs analyzed: {len(subset)}")
        print(f"  Median FDR: {subset['fdr'].median():.4f} ({subset['fdr'].median()*100:.2f}%)")
        print(f"  Mean FDR:   {subset['fdr'].mean():.4f} ({subset['fdr'].mean()*100:.2f}%)")
        print(f"  Std FDR:    {subset['fdr'].std():.4f}")
        print(f"  Min FDR:    {subset['fdr'].min():.4f} ({subset['fdr'].min()*100:.2f}%)")
        print(f"  Max FDR:    {subset['fdr'].max():.4f} ({subset['fdr'].max()*100:.2f}%)")
        print(f"  Mean E₀[V]: {subset['E0_V'].mean():.2f}")
        print(f"  Mean n_original: {subset['n_original'].mean():.1f}")
    
    # Detailed drug-level results
    print("\n" + "-"*70)
    print("DRUG-LEVEL FDR ESTIMATES")
    print("-"*70)
    print(f"\n{'Drug':<15} {'Method':<8} {'n_orig':<8} {'E₀[V]':<8} {'FDR':<10} {'Null Range'}")
    print("-"*70)
    
    for _, row in df.sort_values(['drug', 'method']).iterrows():
        print(f"{row['drug']:<15} {row['method']:<8} "
              f"{row['n_original']:<8.0f} {row['E0_V']:<8.2f} "
              f"{row['fdr']:<10.4f} [{row['null_min']:.0f}-{row['null_max']:.0f}]")
    
    # Check if FDR control is working
    print("\n" + "-"*70)
    print("FDR CONTROL ASSESSMENT")
    print("-"*70)
    
    fdr_target = 0.05  # Typical target
    
    for method in ['linear', 'rf']:
        subset = df[df['method'] == method]
        if len(subset) == 0:
            continue
        
        below_target = (subset['fdr'] <= fdr_target).sum()
        print(f"\n{method.upper()}:")
        print(f"  Analyses with FDR ≤ {fdr_target}: {below_target}/{len(subset)} ({below_target/len(subset)*100:.1f}%)")
        print(f"  Mean FDR: {subset['fdr'].mean():.4f} {'✓' if subset['fdr'].mean() <= fdr_target else '✗ (above target)'}")
    
    # Statistical comparison
    if len(df[df['method'] == 'linear']) > 0 and len(df[df['method'] == 'rf']) > 0:
        print("\n" + "-"*70)
        print("LINEAR vs RF COMPARISON")
        print("-"*70)
        
        linear_fdr = df[df['method'] == 'linear']['fdr'].values
        rf_fdr = df[df['method'] == 'rf']['fdr'].values
        
        print(f"\nLinear FDR: {np.mean(linear_fdr):.4f} ± {np.std(linear_fdr):.4f}")
        print(f"RF FDR:     {np.mean(rf_fdr):.4f} ± {np.std(rf_fdr):.4f}")
        print(f"Difference: {np.mean(linear_fdr) - np.mean(rf_fdr):.4f}")
        
        if len(linear_fdr) > 1 and len(rf_fdr) > 1:
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(linear_fdr, rf_fdr)
            print(f"t-test p-value: {p_val:.4f} {'(significant)' if p_val < 0.05 else '(n.s.)'}")
    
    # Create visualization
    print("\n" + "-"*70)
    print("CREATING VISUALIZATIONS...")
    print("-"*70)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. FDR by method
    ax = axes[0]
    df_plot = df.copy()
    df_plot['Method'] = df_plot['method'].str.upper()
    sns.boxplot(data=df_plot, x='Method', y='fdr', ax=ax, palette=['#3498db', '#e74c3c'])
    ax.axhline(0.05, color='red', linestyle='--', linewidth=1, label='Target (5%)')
    ax.set_ylabel('False Discovery Rate')
    ax.set_title('FDR Distribution by Method')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. E0[V] vs n_original
    ax = axes[1]
    for method, color in [('linear', '#3498db'), ('rf', '#e74c3c')]:
        subset = df[df['method'] == method]
        ax.scatter(subset['n_original'], subset['E0_V'], 
                  label=method.upper(), alpha=0.6, s=100, color=color)
    
    # Add diagonal reference lines
    max_n = df['n_original'].max()
    x_ref = np.linspace(0, max_n, 100)
    for fdr_line in [0.05, 0.10, 0.20]:
        ax.plot(x_ref, x_ref * fdr_line, '--', alpha=0.3, linewidth=1,
               label=f'FDR={fdr_line}')
    
    ax.set_xlabel('Original Selection (n)')
    ax.set_ylabel('Expected Null Selections E₀[V]')
    ax.set_title('Null Model: Expected vs Observed')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. FDR distribution histogram
    ax = axes[2]
    for method, color in [('linear', '#3498db'), ('rf', '#e74c3c')]:
        subset = df[df['method'] == method]
        if len(subset) > 0:
            ax.hist(subset['fdr'], bins=10, alpha=0.6, 
                   label=method.upper(), color=color, edgecolor='black')
    
    ax.axvline(0.05, color='red', linestyle='--', linewidth=2, label='Target (5%)')
    ax.set_xlabel('False Discovery Rate')
    ax.set_ylabel('Frequency')
    ax.set_title('FDR Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path("results/fdr_validation_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    # Save detailed summary table
    summary_path = Path("results/fdr_method_summary.csv")
    method_summary = df.groupby('method').agg({
        'fdr': ['mean', 'median', 'std', 'min', 'max'],
        'E0_V': ['mean', 'median'],
        'n_original': ['mean', 'median'],
        'drug': 'count'
    }).round(4)
    method_summary.to_csv(summary_path)
    print(f"✓ Saved: {summary_path}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
