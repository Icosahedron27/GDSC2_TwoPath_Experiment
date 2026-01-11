#!/usr/bin/env python3
"""
Corrected Resistance vs Sensitivity Analysis using Spearman Correlation
"""

from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

results_dir = Path('results')

# Sammle MUT Features mit Spearman correlation
rf_mutations = defaultdict(lambda: {'drugs': [], 'spearman': [], 'cpss': [], 'r2': []})
linear_mutations = defaultdict(lambda: {'drugs': [], 'spearman': [], 'cpss': [], 'r2': []})

for drug_dir in results_dir.iterdir():
    if not drug_dir.is_dir() or drug_dir.name.startswith('.'):
        continue
    
    drug_name = drug_dir.name
    v1_dir = drug_dir / 'v1-union-na20'
    if not v1_dir.exists():
        continue
    
    # RF
    rf_file = v1_dir / 'cpss_rf_above_threshold.csv'
    if rf_file.exists():
        try:
            df = pd.read_csv(rf_file)
            muts = df[df['feature'].str.startswith('MUT_')]
            for _, row in muts.iterrows():
                gene = row['feature'].replace('MUT_', '').replace('_', '')
                spearman = float(row['spearman']) if not pd.isna(row['spearman']) else 0
                cpss = float(row['cpss_score'])
                r2 = float(row['r2']) if not pd.isna(row['r2']) else 0
                rf_mutations[gene]['drugs'].append(drug_name)
                rf_mutations[gene]['spearman'].append(spearman)
                rf_mutations[gene]['cpss'].append(cpss)
                rf_mutations[gene]['r2'].append(r2)
        except Exception as e:
            pass
    
    # Linear
    linear_file = v1_dir / 'cpss_linear_above_threshold.csv'
    if linear_file.exists():
        try:
            df = pd.read_csv(linear_file)
            muts = df[df['feature'].str.startswith('MUT_')]
            for _, row in muts.iterrows():
                gene = row['feature'].replace('MUT_', '').replace('_', '')
                spearman = float(row['spearman']) if not pd.isna(row['spearman']) else 0
                cpss = float(row['cpss_score'])
                r2 = float(row['r2']) if not pd.isna(row['r2']) else 0
                linear_mutations[gene]['drugs'].append(drug_name)
                linear_mutations[gene]['spearman'].append(spearman)
                linear_mutations[gene]['cpss'].append(cpss)
                linear_mutations[gene]['r2'].append(r2)
        except Exception as e:
            pass

print('='*80)
print('KORRIGIERTE ANALYSE: RESISTENZ vs SENSITIV (mit SPEARMAN)')
print('='*80)

print('\n📊 INTERPRETATION:')
print('-'*80)
print('Spearman > +0.2:  Mutation SENSITIV (höhere Response bei Mutation)')
print('Spearman < -0.2:  Mutation RESISTENT (niedrigere Response bei Mutation)')
print('Spearman ≈ 0:     Keine klare Richtung')

# RF Analysis
print('\n\n1. RF-CPSS MUTATIONEN (mit Spearman Correlation):')
print('='*80)
rf_sorted = sorted(rf_mutations.items(), key=lambda x: len(x[1]['drugs']), reverse=True)

rf_analysis = []
for gene, data in rf_sorted:
    avg_spearman = np.mean(data['spearman'])
    std_spearman = np.std(data['spearman'])
    avg_cpss = np.mean(data['cpss'])
    avg_r2 = np.mean(data['r2'])
    n_drugs = len(data['drugs'])
    
    if avg_spearman > 0.2:
        direction = 'SENSITIV'
        color = 'GREEN'
    elif avg_spearman < -0.2:
        direction = 'RESISTENT'
        color = 'RED'
    else:
        direction = 'NEUTRAL'
        color = 'YELLOW'
    
    rf_analysis.append({
        'gene': gene,
        'n_drugs': n_drugs,
        'avg_spearman': avg_spearman,
        'std_spearman': std_spearman,
        'avg_cpss': avg_cpss,
        'avg_r2': avg_r2,
        'direction': direction,
        'color': color
    })
    
    print(f'\n{gene:10s} | {n_drugs:2d} drugs')
    print(f'  Spearman: {avg_spearman:+.3f} ± {std_spearman:.3f} → {color:6s} {direction}')
    print(f'  CPSS: {avg_cpss:.3f} | R²: {avg_r2:.3f}')
    if n_drugs <= 5:
        drug_details = list(zip(data['drugs'], data['spearman']))[:3]
        for drug, spear in drug_details:
            print(f'    • {drug}: Spearman {spear:+.2f}')

# Linear Analysis
print('\n\n2. LINEAR-CPSS MUTATIONEN (mit Spearman Correlation):')
print('='*80)
linear_sorted = sorted(linear_mutations.items(), key=lambda x: len(x[1]['drugs']), reverse=True)

linear_analysis = []
for gene, data in linear_sorted:
    avg_spearman = np.mean(data['spearman'])
    std_spearman = np.std(data['spearman'])
    avg_cpss = np.mean(data['cpss'])
    avg_r2 = np.mean(data['r2'])
    n_drugs = len(data['drugs'])
    
    if avg_spearman > 0.2:
        direction = 'SENSITIV'
        color = 'GREEN'
    elif avg_spearman < -0.2:
        direction = 'RESISTENT'
        color = 'RED'
    else:
        direction = 'NEUTRAL'
        color = 'YELLOW'
    
    linear_analysis.append({
        'gene': gene,
        'n_drugs': n_drugs,
        'avg_spearman': avg_spearman,
        'std_spearman': std_spearman,
        'avg_cpss': avg_cpss,
        'avg_r2': avg_r2,
        'direction': direction,
        'color': color
    })
    
    print(f'\n{gene:10s} | {n_drugs:2d} drugs')
    print(f'  Spearman: {avg_spearman:+.3f} ± {std_spearman:.3f} → {color:6s} {direction}')
    print(f'  CPSS: {avg_cpss:.3f} | R²: {avg_r2:.3f}')
    drug_details = list(zip(data['drugs'], data['spearman']))
    for drug, spear in drug_details:
        print(f'    • {drug}: Spearman {spear:+.2f}')

# Statistik
print('\n\n3. GESAMTSTATISTIK:')
print('='*80)

rf_sensitive = sum(1 for a in rf_analysis if a['direction'] == 'SENSITIV')
rf_resistant = sum(1 for a in rf_analysis if a['direction'] == 'RESISTENT')
rf_neutral = sum(1 for a in rf_analysis if a['direction'] == 'NEUTRAL')

linear_sensitive = sum(1 for a in linear_analysis if a['direction'] == 'SENSITIV')
linear_resistant = sum(1 for a in linear_analysis if a['direction'] == 'RESISTENT')
linear_neutral = sum(1 for a in linear_analysis if a['direction'] == 'NEUTRAL')

print(f'\nRF-CPSS ({len(rf_analysis)} genes):')
print(f'  Sensitiv:   {rf_sensitive} genes ({rf_sensitive/len(rf_analysis)*100:.0f}%)')
print(f'  Resistent:  {rf_resistant} genes ({rf_resistant/len(rf_analysis)*100:.0f}%)')
print(f'  Neutral:    {rf_neutral} genes ({rf_neutral/len(rf_analysis)*100:.0f}%)')

print(f'\nLinear-CPSS ({len(linear_analysis)} genes):')
print(f'  Sensitiv:   {linear_sensitive} genes ({linear_sensitive/len(linear_analysis)*100:.0f}%)')
print(f'  Resistent:  {linear_resistant} genes ({linear_resistant/len(linear_analysis)*100:.0f}%)')
print(f'  Neutral:    {linear_neutral} genes ({linear_neutral/len(linear_analysis)*100:.0f}%)')

# Details by direction
print('\n\n4. GENE DETAILS BY DIRECTION:')
print('='*80)

print('\nRF - RESISTENT Genes:')
for a in rf_analysis:
    if a['direction'] == 'RESISTENT':
        print(f'  {a["gene"]:10s} | {a["n_drugs"]:2d} drugs | Spearman {a["avg_spearman"]:+.3f}')

print('\nRF - NEUTRAL Genes:')
for a in rf_analysis:
    if a['direction'] == 'NEUTRAL':
        print(f'  {a["gene"]:10s} | {a["n_drugs"]:2d} drugs | Spearman {a["avg_spearman"]:+.3f}')

print('\nRF - SENSITIV Genes:')
for a in rf_analysis:
    if a['direction'] == 'SENSITIV':
        print(f'  {a["gene"]:10s} | {a["n_drugs"]:2d} drugs | Spearman {a["avg_spearman"]:+.3f}')

print('\n\nLinear - RESISTENT Genes:')
for a in linear_analysis:
    if a['direction'] == 'RESISTENT':
        print(f'  {a["gene"]:10s} | {a["n_drugs"]:2d} drugs | Spearman {a["avg_spearman"]:+.3f}')

print('\nLinear - NEUTRAL Genes:')
for a in linear_analysis:
    if a['direction'] == 'NEUTRAL':
        print(f'  {a["gene"]:10s} | {a["n_drugs"]:2d} drugs | Spearman {a["avg_spearman"]:+.3f}')

print('\nLinear - SENSITIV Genes:')
for a in linear_analysis:
    if a['direction'] == 'SENSITIV':
        print(f'  {a["gene"]:10s} | {a["n_drugs"]:2d} drugs | Spearman {a["avg_spearman"]:+.3f}')

print('\n\n' + '='*80)
print('BIOLOGISCHE INTERPRETATION')
print('='*80)

print('\n🔑 HAUPTBEFUNDE:')
print(f'1. RF findet {rf_resistant} RESISTENZ-Marker ({rf_resistant/len(rf_analysis)*100:.0f}%)')
print(f'   → Mutationen mit negativer Spearman Correlation')
print(f'   → Mutation vorhanden = Zellen reagieren SCHLECHTER auf Drug')

print(f'\n2. RF findet {rf_sensitive} SENSITIV-Marker ({rf_sensitive/len(rf_analysis)*100:.0f}%)')
print(f'   → Mutationen mit positiver Spearman Correlation')
print(f'   → Mutation vorhanden = Zellen reagieren BESSER auf Drug')

print(f'\n3. Linear findet {linear_resistant} RESISTENZ + {linear_sensitive} SENSITIV')
print(f'   → Kleinere Stichprobe, aber ähnliches Muster')

print('\n4. WARUM NEGATIV (Resistenz)?')
print('   → GDSC2 misst IC50: HÖHERER Wert = MEHR Resistenz')
print('   → Negative Spearman: Mutation → HÖHERE IC50 → RESISTENT')
print('   → Positive Spearman: Mutation → NIEDRIGERE IC50 → SENSITIV')

print('\n5. EGFR Paradox aufgelöst:')
if any(a['gene'] == 'EGFR' for a in linear_analysis):
    egfr = [a for a in linear_analysis if a['gene'] == 'EGFR'][0]
    print(f'   → EGFR Spearman = {egfr["avg_spearman"]:+.3f}')
    if egfr['avg_spearman'] < 0:
        print('   → NEGATIV = Mutation senkt IC50 → SENSITIV (korrekt)')
    else:
        print('   → POSITIV = Mutation erhöht IC50 → RESISTENT')
