"""
Systematischer Test: RF-CPSS auf 3 Medikamenten mit bekannten Biomarkern.

Ziel: Evaluieren ob RF-basierter nicht-linearer Pfad biologisch relevante Features findet.
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from cpss import cpss_feature_selection
import pandas as pd

# =======================================================================================
# MEDIKAMENT 1: Erlotinib (EGFR-Inhibitor)
# =======================================================================================
def test_erlotinib():
    print("="*90)
    print("MEDIKAMENT 1: Erlotinib (EGFR Tyrosine Kinase Inhibitor)")
    print("="*90)
    
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'Erlotinib' / 'v1-union-na20' / 'design_matrix'
    
    print("\nErwartete Biomarker:")
    print("  Primary: MUT__EGFR (aktivierende Mutationen → Sensitivität)")
    print("  Secondary: CNV__EGFR, TPM__EGFR (Amplifikation/Expression)")
    print("  Resistance: MUT__KRAS, MUT__BRAF (downstream activation)")
    print("  Bypass: TPM__MET, CNV__MET (alternative signaling)")
    print()
    
    cpss_scores, important_features = cpss_feature_selection(
        path=data_path,
        XName='X_general_sis_reduced.parquet',
        B=200,
        linear=False  # RF path
    )
    
    expected = {
        'MUT__EGFR': 'EGFR mutations (primary)',
        'CNV__EGFR': 'EGFR amplification',
        'TPM__EGFR': 'EGFR expression',
        'MUT__KRAS': 'KRAS resistance',
        'MUT__BRAF': 'BRAF resistance',
        'TPM__MET': 'MET bypass',
        'CNV__MET': 'MET amplification'
    }
    
    print("\n" + "-"*90)
    print("Top 15 Features:")
    print("-"*90)
    for idx, row in cpss_scores.head(15).iterrows():
        marker = "✓" if row['feature'] in expected else " "
        print(f"{marker} {idx+1:2d}. {row['feature']:35s} Score={row['cpss_score']:.3f}")
    
    print("\n" + "-"*90)
    print("Biomarker Validierung:")
    print("-"*90)
    found = 0
    for marker, desc in expected.items():
        if marker in cpss_scores['feature'].values:
            score = cpss_scores[cpss_scores['feature'] == marker]['cpss_score'].values[0]
            rank = cpss_scores[cpss_scores['feature'] == marker].index[0] + 1
            if score > 0.5:
                print(f"✓✓ {marker:20s} Rank {rank:3d}, Score {score:.3f} - {desc}")
                found += 1
            elif score > 0.1:
                print(f"✓  {marker:20s} Rank {rank:3d}, Score {score:.3f} - {desc} (schwach)")
            else:
                print(f"~  {marker:20s} Rank {rank:3d}, Score {score:.3f} - {desc} (zu schwach)")
        else:
            print(f"✗  {marker:20s} NOT FOUND - {desc}")
    
    print(f"\nStark gefunden: {found}/{len(expected)}")
    return cpss_scores, found, len(expected)


# =======================================================================================
# MEDIKAMENT 2: Cisplatin (DNA-Schädigendes Agens)
# =======================================================================================
def test_cisplatin():
    print("\n\n")
    print("="*90)
    print("MEDIKAMENT 2: Cisplatin (DNA Cross-Linking Agent)")
    print("="*90)
    
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'Cisplatin' / 'v1-union-na20' / 'design_matrix'
    
    print("\nErwartete Biomarker:")
    print("  DNA Repair: MUT__TP53, MUT__ERCC1, MUT__ERCC2 (NER pathway)")
    print("  MMR: MUT__MSH2, MUT__MLH1 (Mismatch repair)")
    print("  HR: MUT__BRCA1, MUT__BRCA2 (Homologous recombination)")
    print("  Resistance: MUT__KRAS")
    print()
    
    cpss_scores, important_features = cpss_feature_selection(
        path=data_path,
        XName='X_general_sis_reduced.parquet',
        B=200,
        linear=False  # RF path
    )
    
    expected = {
        'MUT__TP53': 'TP53 (apoptosis)',
        'MUT__ERCC1': 'ERCC1 (NER)',
        'MUT__ERCC2': 'ERCC2 (NER helicase)',
        'MUT__MSH2': 'MSH2 (MMR)',
        'MUT__MLH1': 'MLH1 (MMR)',
        'MUT__BRCA1': 'BRCA1 (HR)',
        'MUT__BRCA2': 'BRCA2 (HR)',
        'MUT__KRAS': 'KRAS (resistance)'
    }
    
    print("\n" + "-"*90)
    print("Top 15 Features:")
    print("-"*90)
    for idx, row in cpss_scores.head(15).iterrows():
        marker = "✓" if row['feature'] in expected else " "
        print(f"{marker} {idx+1:2d}. {row['feature']:35s} Score={row['cpss_score']:.3f}")
    
    print("\n" + "-"*90)
    print("Biomarker Validierung:")
    print("-"*90)
    found = 0
    for marker, desc in expected.items():
        if marker in cpss_scores['feature'].values:
            score = cpss_scores[cpss_scores['feature'] == marker]['cpss_score'].values[0]
            rank = cpss_scores[cpss_scores['feature'] == marker].index[0] + 1
            if score > 0.5:
                print(f"✓✓ {marker:20s} Rank {rank:3d}, Score {score:.3f} - {desc}")
                found += 1
            elif score > 0.1:
                print(f"✓  {marker:20s} Rank {rank:3d}, Score {score:.3f} - {desc} (schwach)")
            else:
                print(f"~  {marker:20s} Rank {rank:3d}, Score {score:.3f} - {desc} (zu schwach)")
        else:
            print(f"✗  {marker:20s} NOT FOUND - {desc}")
    
    print(f"\nStark gefunden: {found}/{len(expected)}")
    return cpss_scores, found, len(expected)


# =======================================================================================
# MEDIKAMENT 3: Dabrafenib (BRAF-Inhibitor)
# =======================================================================================
def test_dabrafenib():
    print("\n\n")
    print("="*90)
    print("MEDIKAMENT 3: Dabrafenib (BRAF V600E Inhibitor)")
    print("="*90)
    
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'Dabrafenib' / 'v1-union-na20' / 'design_matrix'
    
    print("\nErwartete Biomarker:")
    print("  Primary: MUT__BRAF (V600E Mutation → high sensitivity)")
    print("  Pathway: MUT__NRAS, MUT__KRAS (MAPK pathway)")
    print("  Expression: TPM__BRAF")
    print()
    
    cpss_scores, important_features = cpss_feature_selection(
        path=data_path,
        XName='X_general_sis_reduced.parquet',
        B=200,
        linear=False  # RF path
    )
    
    expected = {
        'MUT__BRAF': 'BRAF V600E (primary target)',
        'MUT__NRAS': 'NRAS (MAPK pathway)',
        'MUT__KRAS': 'KRAS (MAPK pathway)',
        'TPM__BRAF': 'BRAF expression'
    }
    
    print("\n" + "-"*90)
    print("Top 15 Features:")
    print("-"*90)
    for idx, row in cpss_scores.head(15).iterrows():
        marker = "✓" if row['feature'] in expected else " "
        print(f"{marker} {idx+1:2d}. {row['feature']:35s} Score={row['cpss_score']:.3f}")
    
    print("\n" + "-"*90)
    print("Biomarker Validierung:")
    print("-"*90)
    found = 0
    for marker, desc in expected.items():
        if marker in cpss_scores['feature'].values:
            score = cpss_scores[cpss_scores['feature'] == marker]['cpss_score'].values[0]
            rank = cpss_scores[cpss_scores['feature'] == marker].index[0] + 1
            if score > 0.5:
                print(f"✓✓ {marker:20s} Rank {rank:3d}, Score {score:.3f} - {desc}")
                found += 1
            elif score > 0.1:
                print(f"✓  {marker:20s} Rank {rank:3d}, Score {score:.3f} - {desc} (schwach)")
            else:
                print(f"~  {marker:20s} Rank {rank:3d}, Score {score:.3f} - {desc} (zu schwach)")
        else:
            print(f"✗  {marker:20s} NOT FOUND - {desc}")
    
    print(f"\nStark gefunden: {found}/{len(expected)}")
    return cpss_scores, found, len(expected)


# =======================================================================================
# MAIN
# =======================================================================================
if __name__ == "__main__":
    results = []
    
    # Test 1: Erlotinib
    scores1, found1, total1 = test_erlotinib()
    results.append(('Erlotinib', found1, total1))
    
    # Test 2: Cisplatin
    scores2, found2, total2 = test_cisplatin()
    results.append(('Cisplatin', found2, total2))
    
    # Test 3: Dabrafenib
    scores3, found3, total3 = test_dabrafenib()
    results.append(('Dabrafenib', found3, total3))
    
    # Summary
    print("\n\n")
    print("="*90)
    print("GESAMT-ZUSAMMENFASSUNG: RF-CPSS Performance")
    print("="*90)
    
    total_found = sum(r[1] for r in results)
    total_expected = sum(r[2] for r in results)
    
    for drug, found, total in results:
        pct = 100 * found / total if total > 0 else 0
        print(f"{drug:15s}: {found}/{total} Biomarker gefunden ({pct:.1f}%)")
    
    overall_pct = 100 * total_found / total_expected if total_expected > 0 else 0
    print(f"\n{'GESAMT':15s}: {total_found}/{total_expected} Biomarker gefunden ({overall_pct:.1f}%)")
    
    print("\n" + "="*90)
    if overall_pct > 50:
        print("✓ RF-CPSS funktioniert: Mehr als 50% der Biomarker gefunden")
    elif overall_pct > 25:
        print("~ RF-CPSS teilweise erfolgreich: 25-50% der Biomarker gefunden")
    else:
        print("✗ RF-CPSS unzureichend: Weniger als 25% der Biomarker gefunden")
    print("="*90)
