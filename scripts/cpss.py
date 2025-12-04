from pathlib import Path
import pandas as pd
from pandas import Series
import numpy as np
import math
import random
import json
import yaml
from numpy.typing import NDArray
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
from diptest import diptest

# ----------------------------------------------------------------------------------------
# General methods
# ----------------------------------------------------------------------------------------
def getSubsamples(X, y, B: int):
    n = X.shape[0]
    n_half = math.floor(n / 2)
    
    subsamples = []
    
    for j in range(B):
        indices = list(range(n))
        random.shuffle(indices)

        A_2j_minus_1 = indices[:n_half]
        A_2j = indices[n_half:2*n_half]
        
        subsamples.append((A_2j_minus_1, A_2j))
    
    return subsamples


def compute_feature_metrics(X: pd.DataFrame, y: pd.Series) -> tuple[dict, dict, dict]:
    """
    Compute R², Spearman correlation, and Mutual Information for each feature.
    
    Args:
        X: Design matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        
    Returns:
        Tuple of (r2_scores, spearman_corr, mutual_info) dictionaries
    """
    r2_scores = {}
    spearman_corr = {}
    mutual_info = {}
    
    y_np = y.values.ravel() if isinstance(y, pd.Series) else y.ravel()
    
    for col in X.columns:
        X_col = X[col].values.ravel()
        
        # Remove NaN values for this feature
        mask = ~(np.isnan(X_col) | np.isnan(y_np))
        if mask.sum() < 2:  # Need at least 2 samples
            r2_scores[col] = np.nan
            spearman_corr[col] = np.nan
            mutual_info[col] = np.nan
            continue
        
        X_clean = X_col[mask].reshape(-1, 1)
        y_clean = y_np[mask]
        
        # 1) Linear R²
        lr = LinearRegression()
        lr.fit(X_clean, y_clean)
        y_pred = lr.predict(X_clean)
        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r2_scores[col] = r2
        
        # 2) Spearman correlation (with sign)
        rho, _ = spearmanr(X_clean.ravel(), y_clean)
        spearman_corr[col] = rho
        
        # 3) Mutual Information
        mi = mutual_info_regression(X_clean, y_clean, random_state=42)[0]
        mutual_info[col] = mi
    
    return r2_scores, spearman_corr, mutual_info


def tau_grid_TB(B: int):
    start = 0.5 + 1.0 / B
    end = 1.0
    n_points = B - 1
    return np.linspace(start, end, n_points)

def C_tau_B(tau: float, B: int, theta: float = 1/np.sqrt(3)) -> float:
    if theta > 1/np.sqrt(3):
        return np.inf
    
    lower_1 = min(0.5 + theta**2, 0.5 + 1.0/(2*B) + 0.75*theta**2)
    
    if (tau > lower_1) and (tau <= 0.75):
        denom = 2.0 * (2.0 * tau - 1.0 - 1.0/(2.0 * B))
        if denom <= 0:
            return np.inf
        return 1.0 / denom
    
    if (tau > 0.75) and (tau <= 1.0):
        num = 4.0 * (1.0 - tau + 1.0/(2.0 * B))
        denom = 1.0 + 1.0/B
        return num / denom
    
    return np.inf

def tau_star_unimodal(B: int, q_hat: float, d: int, l: float, theta: float) -> float | None:
    if theta > 1/np.sqrt(3):
        return None
    
    taus = tau_grid_TB(B)
    factor = (q_hat**2) / d
    
    feasible = []
    for tau in taus:
        c_val = C_tau_B(tau, B, theta=theta)
        if c_val * factor <= l:
            feasible.append(tau)
    
    if not feasible:
        return None
    return min(feasible)

def get_important_features(cpss_scores, d, q_hat, theta, error_control_level, B, name):
    worst_case_bound = min(1, 0.5*(1+ (q_hat**2)/(error_control_level*d)))
    
    unimodal_bound = None
    if theta <= 1/np.sqrt(3):
        unimodal_bound = tau_star_unimodal(B, q_hat, d, error_control_level, theta)
    
    # Determine threshold for each feature based on unimodality
    cpss_scores['threshold_type'] = cpss_scores['unimodal'].apply(
        lambda x: 'unimodal' if (x and unimodal_bound is not None) else 'worst_case'
    )
    cpss_scores['threshold_value'] = cpss_scores['threshold_type'].apply(
        lambda x: unimodal_bound if (x == 'unimodal' and unimodal_bound is not None) else worst_case_bound
    )
    cpss_scores['above_threshold'] = cpss_scores['cpss_score'] >= cpss_scores['threshold_value']
    
    important_features = pd.DataFrame(
    index=cpss_scores["feature"],
    data={
        "Selection count": cpss_scores["selection_count"].values,
        "Pair count": cpss_scores["pair_count"].values,
        "CPSS Score": cpss_scores["cpss_score"].values
    })
    important_features.index.name = "feature"
    
    important_features = important_features.sort_values('CPSS Score', ascending=False)
    important_features.to_csv("Significant_features"+name+".csv")
    
    bounds_info = {
        "worst_case_bound": float(worst_case_bound),
        "unimodal_bound": float(unimodal_bound) if unimodal_bound is not None else None,
        "theta": float(theta),
        "q_hat": float(q_hat),
        "d": int(d),
        "l": int(error_control_level),
        "B": int(B),
        "n_features_worst_case": int((cpss_scores["cpss_score"] >= worst_case_bound).sum()),
        "n_features_unimodal": int((cpss_scores["cpss_score"] >= unimodal_bound).sum()) if unimodal_bound is not None else None,
        "n_features_above_threshold": int(cpss_scores['above_threshold'].sum())
    }
    
    with open(f"cpss_bounds_{name}.json", "w") as f:
        json.dump(bounds_info, f, indent=2)
    
    # Select features based on their individual threshold
    features_above_threshold = cpss_scores[cpss_scores['above_threshold']].copy()
    features_above_threshold.to_csv(f"features_above_threshold_{name}.csv", index=False)
    
    # Keep legacy worst-case file for comparison
    features_above_worst_case = cpss_scores[cpss_scores["cpss_score"] >= worst_case_bound].copy()
    features_above_worst_case.to_csv(f"features_above_worst_case_{name}.csv", index=False)
    
    return important_features

# ----------------------------------------------------------------------------------------
# Main CPSS Feature Selection
# ----------------------------------------------------------------------------------------
def cpss_feature_selection(path: Path, XName, B: int, linear):
    config_path = Path(__file__).parent.parent / 'configs' / 'prep' / 'blocks.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    alpha_grid = config['cpss']['alpha_grid']
    
    X = pd.read_parquet(path / XName)
    y = pd.read_parquet(path / 'y.parquet')

    if 'cell_id' in X.columns:
        X = X.drop(columns=['cell_id'])
    if 'cell_id' in y.columns:
        y = y.drop(columns=['cell_id'])

    d = X.shape[1]
    error_control_level = int(0.05 * d)
    q = int(np.sqrt(0.8 * error_control_level * d))

    print(f"\nGenerating {B} complementary pairs...")

    subsamples = getSubsamples(X, y, B)

    print(f"\nRunning CPSS with B={B} pairs...")
    if linear:
        selection_counts, pair_counts, pair_distributions = cpss_linear_core(X, y, B, alpha_grid[0], subsamples, q)
        name = 'linear'
    else:
        rf_params = {'n_estimators': 100, 'max_depth': 12, 'random_state': 42}
        selection_counts, pair_counts, pair_distributions = cpss_rf_core(X, y, B, rf_params, subsamples, q)
        name = 'rf'

    q_hat = sum(selection_counts.values()) / (2*B)
    theta = q_hat / d
    
    # Load full feature matrix for computing validation metrics (only for selected features)
    X_full = pd.read_parquet(path / 'X.parquet')
    if 'cell_id' in X_full.columns:
        X_full = X_full.drop(columns=['cell_id'])
    
    selected_features = [feat for feat, count in selection_counts.items() if count > 0]
    print(f"\nComputing validation metrics (R², Spearman, MI) for {len(selected_features)} selected features from full matrix...")
    X_selected = X_full[selected_features]
    r2_dict, spearman_dict, mi_dict = compute_feature_metrics(X_selected, y.iloc[:, 0])
    
    # Test unimodality of pair occurrence distributions
    print(f"\nTesting unimodality of pair occurrence distributions...")
    unimodal_flags = {}
    diptest_pvalues = {}
    for feature in selection_counts.keys():
        if pair_counts[feature] > 0:  # Only test features that appear at least once
            distribution = np.array(pair_distributions[feature])
            try:
                dip_stat, p_value = diptest(distribution)
                diptest_pvalues[feature] = p_value
                # p > 0.05 suggests unimodality
                unimodal_flags[feature] = p_value > 0.05
            except:
                diptest_pvalues[feature] = np.nan
                unimodal_flags[feature] = False
        else:
            diptest_pvalues[feature] = np.nan
            unimodal_flags[feature] = False

    cpss_scores = pd.DataFrame({
        'feature': list(selection_counts.keys()),
        'selection_count': list(selection_counts.values()),
        'pair_count': list(pair_counts.values()),
        'cpss_score': [count / (2 * B) for count in selection_counts.values()],
        'r2': [round(r2_dict.get(feat, np.nan), 2) for feat in selection_counts.keys()],
        'spearman': [round(spearman_dict.get(feat, np.nan), 2) for feat in selection_counts.keys()],
        'mutual_info': [round(mi_dict.get(feat, np.nan), 2) for feat in selection_counts.keys()],
        'unimodal': [unimodal_flags.get(feat, False) for feat in selection_counts.keys()],
        'diptest_pvalue': [round(diptest_pvalues.get(feat, np.nan), 3) for feat in selection_counts.keys()]
    }).sort_values('cpss_score', ascending=False)

    cpss_scores.to_csv(f'cpss_scores_{name}.csv', index=False)
    
    important_features = get_important_features(cpss_scores, d, q_hat, theta, error_control_level, B, name)
    
    print(f"\nCPSS Results:")
    print(f"  Features (d): {d}")
    print(f"  Target q: {q}")
    print(f"  Observed q̂: {q_hat:.1f}")
    print(f"  θ (q̂/d): {theta:.4f}")
    print(f"  Worst-case τ*: {min(1, 0.5*(1 + q_hat**2/(error_control_level*d))):.3f}")
    print(f"  Features passing worst-case: {(cpss_scores['cpss_score'] >= min(1, 0.5*(1 + q_hat**2/(error_control_level*d)))).sum()}")
    
    return cpss_scores, important_features

# ----------------------------------------------------------------------------------------
# Linear Path specific
# ----------------------------------------------------------------------------------------
def compute_lambda_grid(X, y, n_lambda=100, lambda_min_ratio=0.001):
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    
    n = X.shape[0]
    correlations = np.abs(X.T @ y) / n
    lambda_max = np.max(correlations)
    lambda_min = lambda_min_ratio * lambda_max
    lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), n_lambda)
    return lambdas

def linear_base_selector(X, y, alpha_set, target_q):
    lambda_grid = compute_lambda_grid(X, y)
    best_features = set()
    best_diff = float('inf')

    model = ElasticNet(
        alpha=lambda_grid[0],
        l1_ratio=alpha_set,
        fit_intercept=True,
        max_iter=5000,
        tol=1e-3,
        warm_start=True)

    for lam in lambda_grid:
        model.alpha = lam
        model.fit(X, y)

        selected_mask = np.abs(model.coef_) > 1e-6
        n_selected = np.sum(selected_mask)

        diff = abs(n_selected - target_q)
        if diff < best_diff:
            best_diff = diff
            best_features = set(np.where(selected_mask)[0])

            if diff == 0:
                break
    return best_features

def cpss_linear_core(X, y, B, l1_ratios, subsamples, q):
    if subsamples is None:
        raise ValueError("subsamples cannot be None")
    
    feature_names = X.columns.tolist()
    selection_counts = {feature: 0 for feature in feature_names}
    pair_counts = {feature: 0 for feature in feature_names}
    pair_distributions = {feature: [] for feature in feature_names}  # Track per-pair occurrences

    for j, (A_odd, A_even) in enumerate(subsamples, start=1):
        X_sub_odd = X.iloc[A_odd].values
        y_sub_odd = y.iloc[A_odd].values.ravel()
        selected_odd = linear_base_selector(X_sub_odd, y_sub_odd, l1_ratios, q)
        for feat_idx in selected_odd:
            selection_counts[feature_names[feat_idx]] += 1

        X_sub_even = X.iloc[A_even].values
        y_sub_even = y.iloc[A_even].values.ravel()
        selected_even = linear_base_selector(X_sub_even, y_sub_even, l1_ratios, q)
        for feat_idx in selected_even:
            selection_counts[feature_names[feat_idx]] += 1
        
        selected_both = selected_odd & selected_even
        for feat_name in feature_names:
            feat_idx = feature_names.index(feat_name)
            if feat_idx in selected_both:
                pair_counts[feat_name] += 1
                pair_distributions[feat_name].append(1)
            else:
                pair_distributions[feat_name].append(0)
        
    return selection_counts, pair_counts, pair_distributions


# ----------------------------------------------------------------------------------------
# Non Linear Path specific
# ----------------------------------------------------------------------------------------
def rf_base_selector(X, y, target_q, feature_names=None, n_estimators=100, max_depth=12, 
                     max_features='sqrt', min_samples_split=10, random_state=42):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X, y)
    
    importances = model.feature_importances_
    
    if feature_names is not None:
        weights = np.ones(len(feature_names))
        for i, name in enumerate(feature_names):
            if name.startswith('MUT__'):
                weights[i] = 2.0
            elif name.startswith('CNV__'):
                weights[i] = 1.5
            elif name.startswith('TPM__'):
                weights[i] = 1.2
        importances = importances * weights
    
    ranked_indices = np.argsort(importances)[::-1]
    selected = set(ranked_indices[:target_q])
    
    return selected

def cpss_rf_core(X, y, B, rf_params, subsamples, q):
    if subsamples is None:
        raise ValueError("subsamples cannot be None")
    
    feature_names = X.columns.tolist()
    selection_counts = {feature: 0 for feature in feature_names}
    pair_counts = {feature: 0 for feature in feature_names}
    pair_distributions = {feature: [] for feature in feature_names}  # Track per-pair occurrences

    for j, (A_odd, A_even) in enumerate(subsamples, start=1):
        X_sub_odd = X.iloc[A_odd].values
        y_sub_odd = y.iloc[A_odd].values.ravel()
        selected_odd = rf_base_selector(X_sub_odd, y_sub_odd, q, feature_names=feature_names, **rf_params)
        for feat_idx in selected_odd:
            selection_counts[feature_names[feat_idx]] += 1

        X_sub_even = X.iloc[A_even].values
        y_sub_even = y.iloc[A_even].values.ravel()
        selected_even = rf_base_selector(X_sub_even, y_sub_even, q, feature_names=feature_names, **rf_params)
        for feat_idx in selected_even:
            selection_counts[feature_names[feat_idx]] += 1
        
        selected_both = selected_odd & selected_even
        for feat_name in feature_names:
            feat_idx = feature_names.index(feat_name)
            if feat_idx in selected_both:
                pair_counts[feat_name] += 1
                pair_distributions[feat_name].append(1)
            else:
                pair_distributions[feat_name].append(0)
        
    return selection_counts, pair_counts, pair_distributions
