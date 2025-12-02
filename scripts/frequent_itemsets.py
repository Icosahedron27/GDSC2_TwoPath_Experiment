"""Frequent Itemset Mining across drugs to find common biomarkers."""
import pandas as pd
import argparse
from pathlib import Path
import json


def load_drug_features(results_dir: Path, drugs: list, method: str = "rf") -> dict:
    drug_features = {}
    for drug in drugs:
        feature_file = results_dir / drug / "v1-union-na20" / f"cpss_{method}_above_threshold.csv"
        if feature_file.exists():
            df = pd.read_csv(feature_file)
            drug_features[drug] = set(df['feature'])
    return drug_features


def frequent_itemsets(drug_features: dict, min_support: int = 2, max_size: int = 3):
    all_features = set()
    for features in drug_features.values():
        all_features.update(features)
    
    results = []
    
    for size in range(1, min(max_size + 1, len(all_features) + 1)):
        if size == 1:
            candidates = [(f,) for f in all_features]
        else:
            prev_items = [item for item in results if len(item['itemset']) == size - 1]
            candidate_set = set()
            for item in prev_items:
                for f in all_features:
                    candidate = tuple(sorted(list(item['itemset']) + [f]))
                    if len(set(candidate)) == size:
                        candidate_set.add(candidate)
            candidates = list(candidate_set)
        
        for itemset in candidates:
            support_count = sum(1 for drug_feats in drug_features.values() 
                               if set(itemset).issubset(drug_feats))
            
            if support_count >= min_support:
                supporting_drugs = [drug for drug, feats in drug_features.items() 
                                   if set(itemset).issubset(feats)]
                
                results.append({
                    'itemset': list(itemset),
                    'size': size,
                    'support': support_count,
                    'support_pct': support_count / len(drug_features) * 100,
                    'drugs': supporting_drugs
                })
    
    return sorted(results, key=lambda x: (x['support'], x['size']), reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=['linear', 'rf', 'both'], default='rf')
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--max-size", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    
    results_dir = Path("results")
    drugs = [d.name for d in results_dir.iterdir() 
             if d.is_dir() and (d / "v1-union-na20").exists()]
    
    if args.method in ['rf', 'both']:
        rf_features = load_drug_features(results_dir, drugs, "rf")
        rf_itemsets = frequent_itemsets(rf_features, args.min_support, args.max_size)
        
        if rf_itemsets:
            rf_df = pd.DataFrame(rf_itemsets[:args.top_k])
            rf_df['itemset'] = rf_df['itemset'].apply(lambda x: ', '.join(x))
            rf_df['drugs'] = rf_df['drugs'].apply(lambda x: ', '.join(x[:5]) + (f' (+{len(x)-5})' if len(x) > 5 else ''))
            rf_df.to_csv(results_dir / "frequent_itemsets_rf.csv", index=False)
    
    if args.method in ['linear', 'both']:
        linear_features = load_drug_features(results_dir, drugs, "linear")
        linear_itemsets = frequent_itemsets(linear_features, args.min_support, args.max_size)
        
        if linear_itemsets:
            linear_df = pd.DataFrame(linear_itemsets[:args.top_k])
            linear_df['itemset'] = linear_df['itemset'].apply(lambda x: ', '.join(x))
            linear_df['drugs'] = linear_df['drugs'].apply(lambda x: ', '.join(x[:5]) + (f' (+{len(x)-5})' if len(x) > 5 else ''))
            linear_df.to_csv(results_dir / "frequent_itemsets_linear.csv", index=False)


if __name__ == "__main__":
    main()
