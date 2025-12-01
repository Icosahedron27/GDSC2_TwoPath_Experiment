"""
Test Linear CPSS (ElasticNet) auf Erlotinib, Cisplatin, Dabrafenib
Paper-conform implementation mit method='elasticnet'
"""

import pandas as pd
import numpy as np
from pathlib import Path
from cpss_corrected import cpss_with_fixed_q

# Known biomarkers
ERLOTINIB_MARKERS = {
    'MUT__EGFR': 'EGFR mutations (primary target)',
    'CNV__EGFR': 'EGFR amplification',
    'TPM__EGFR': 'EGFR expression',
    'MUT__KRAS': 'KRAS resistance',
    'MUT__BRAF': 'BRAF resistance',
    'TPM__MET': 'MET bypass',
    'CNV__MET': 'MET amplification'
}

CISPLATIN_MARKERS = {
    'MUT__TP53': 'TP53 (apoptosis)',
    'MUT__ERCC1': 'ERCC1 (NER)',
    'MUT__ERCC2': 'ERCC2 (NER helicase)',
    'MUT__MSH2': 'MSH2 (MMR)',
    'MUT__MLH1': 'MLH1 (MMR)',
    'MUT__BRCA1': 'BRCA1 (HR)',
    'MUT__BRCA2': 'BRCA2 (HR)',
    'MUT__KRAS': 'KRAS (resistance)'
}

DABRAFENIB_MARKERS = {
    'MUT__BRAF': 'BRAF V600E (primary target)',
    'MUT__NRAS': 'NRAS (MAPK pathway)',
    'MUT__KRAS': 'KRAS (MAPK pathway)',
    'TPM__BRAF': 'BRAF expression'
}


def test_drug(drug_name, expected_markers, B=10):
    """Test Linear CPSS on one drug with biomarker validation"""
    
    print("="*90)
    print(f"DRUG: {drug_name} (Linear CPSS, paper-conform)")
    print("="*90)
    print(f"\nExpected biomarkers ({len(expected_markers)}):")
    for marker, desc in expected_markers.items():
        print(f"  - {marker}: {desc}")
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data" / "processed" / drug_name / "v1-union-na20" / "design_matrix"
    X = pd.read_parquet(data_dir / 'X_linear_sis_reduced.parquet')  # Linear: Pearson correlation SIS
    y = pd.read_parquet(data_dir / 'y.parquet')
    
    if 'cell_id' in X.columns:
        X = X.drop(columns=['cell_id'])
    if 'cell_id' in y.columns:
        y = y.drop(columns=['cell_id'])
    
    d = X.shape[1]
    l = int(0.05 * d)
    q = int(np.sqrt(0.8 * l * d))
    
    print(f"\nData: n={len(X)}, d={d}, l={l}, q={q}")
    
    # Run Linear CPSS with paper-conform implementation
    print(f"\nRunning paper-conform Linear CPSS with B={B} pairs...")
    selection_probs, feature_names = cpss_with_fixed_q(
        X, y, B=B, q=q, method='elasticnet', l1_ratio=0.9, n_lambda=100, random_state=42
    )
    
    # Compute thresholds
    q_hat = selection_probs.sum()
    theta = q_hat / d
    tau_star_wc = min(1, 0.5 * (1 + q_hat**2 / (l * d)))
    
    # Compute unimodal bound (INFORMATIONAL ONLY)
    from cpss_corrected import tau_star_unimodal
    tau_star_um = None
    if theta <= 1/np.sqrt(3):
        tau_star_um = tau_star_unimodal(B, q_hat, d, l, theta)
    
    # Create results DataFrame with threshold columns
    results = pd.DataFrame({
        'feature': feature_names,
        'selection_prob': selection_probs,
        'tau_worst_case': tau_star_wc,
        'tau_unimodal': tau_star_um if tau_star_um is not None else np.nan,
        'passes_worst_case': selection_probs >= tau_star_wc,
        'passes_unimodal': (selection_probs >= tau_star_um) if tau_star_um is not None else False
    }).sort_values('selection_prob', ascending=False)
    
    print(f"\nResults:")
    print(f"  q̂ = {q_hat:.1f} (target: {q})")
    print(f"  θ = {theta:.4f}")
    print(f"  τ* (worst-case) = {tau_star_wc:.3f}")
    if tau_star_um is not None:
        print(f"  τ* (unimodal*) = {tau_star_um:.3f} [*ASSUMPTION NOT TESTABLE]")
        print(f"  Features passing unimodal: {(selection_probs >= tau_star_um).sum()}")
    print(f"  Features passing worst-case: {(selection_probs >= tau_star_wc).sum()}")
    
    # Top 15 features
    print("\n" + "-"*90)
    print("Top 15 Features:")
    print("-"*90)
    for i, row in results.head(15).iterrows():
        marker_status = "✓" if row.feature in expected_markers else " "
        wc_status = "[WC]" if row.passes_worst_case else "    "
        um_status = "[UM]" if row.passes_unimodal else "    "
        print(f"{marker_status} {wc_status} {um_status} {row.feature:35s} Π̂={row.selection_prob:.3f}")
    
    # Biomarker validation
    print("\n" + "-"*90)
    print("Biomarker Validation:")
    print("-"*90)
    
    found_strong = 0
    found_weak = 0
    
    for marker, desc in expected_markers.items():
        if marker in results['feature'].values:
            prob = results[results['feature'] == marker]['selection_prob'].values[0]
            rank = results[results['feature'] == marker].index[0] + 1
            
            if prob >= 0.5:
                print(f"✓✓ {marker:20s} Rank {rank:3d}, Π̂={prob:.3f} - {desc}")
                found_strong += 1
            elif prob >= 0.1:
                print(f"✓  {marker:20s} Rank {rank:3d}, Π̂={prob:.3f} - {desc}")
                found_weak += 1
            else:
                print(f"~  {marker:20s} Rank {rank:3d}, Π̂={prob:.3f} - {desc} (too weak)")
        else:
            print(f"✗  {marker:20s} NOT FOUND - {desc}")
    
    print(f"\nStrong (Π̂≥0.5): {found_strong}/{len(expected_markers)}")
    print(f"Weak (Π̂≥0.1): {found_weak}/{len(expected_markers)}")
    print(f"Total: {found_strong + found_weak}/{len(expected_markers)} ({100*(found_strong+found_weak)/len(expected_markers):.1f}%)")
    
    # Save results with thresholds
    results.to_csv(f"cpss_linear_{drug_name}_B{B}_results.csv", index=False)
    
    return found_strong, found_weak, len(expected_markers)


if __name__ == "__main__":
    print("\n" + "="*90)
    print("LINEAR CPSS TEST: Paper-conform ElasticNet implementation")
    print("="*90)
    print("\nChanges from old implementation:")
    print("  1. q is FIXED (not adaptive)")
    print("  2. λ is tuned to achieve |Ŝ| ≈ q on EACH subsample")
    print("  3. NO invalid unimodality tests")
    print("  4. Only worst-case bound (always valid)")
    
    # Test all 3 drugs
    results_summary = []
    
    for drug, markers in [
        ("Erlotinib", ERLOTINIB_MARKERS),
        ("Cisplatin", CISPLATIN_MARKERS),
        ("Dabrafenib", DABRAFENIB_MARKERS)
    ]:
        strong, weak, total = test_drug(drug, markers, B=10)
        results_summary.append((drug, strong, weak, total))
        print("\n")
    
    # Summary
    print("="*90)
    print("SUMMARY - LINEAR CPSS")
    print("="*90)
    for drug, strong, weak, total in results_summary:
        pct = 100 * (strong + weak) / total
        print(f"{drug:15s}: {strong+weak}/{total} biomarkers ({pct:.1f}%) - Strong: {strong}, Weak: {weak}")
    
    total_found = sum(s + w for _, s, w, _ in results_summary)
    total_markers = sum(t for _, _, _, t in results_summary)
    overall_pct = 100 * total_found / total_markers
    
    print(f"\nOVERALL: {total_found}/{total_markers} ({overall_pct:.1f}%)")
    
    if overall_pct >= 50:
        print("✓ SUCCESS: >50% biomarkers found")
    elif overall_pct >= 25:
        print("~ PARTIAL: 25-50% biomarkers found")
    else:
        print("✗ FAILURE: <25% biomarkers found")
    
    # Comparison with RF
    print("\n" + "="*90)
    print("COMPARISON: Linear vs RF (both B=10)")
    print("="*90)
    print("\nRF-CPSS Results (from previous run):")
    print("  Erlotinib : 0/7 (0.0%)")
    print("  Cisplatin : 3/8 (37.5%)")
    print("  Dabrafenib: 2/4 (50.0%)")
    print("  OVERALL   : 5/19 (26.3%)")
    print("\nLinear CPSS Results (this run):")
    for drug, strong, weak, total in results_summary:
        pct = 100 * (strong + weak) / total
        print(f"  {drug:10s}: {strong+weak}/{total} ({pct:.1f}%)")
    print(f"  OVERALL   : {total_found}/{total_markers} ({overall_pct:.1f}%)")
