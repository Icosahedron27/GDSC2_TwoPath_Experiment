from pathlib import Path
import pandas as pd
from pandas import Series
import numpy as np
import math
import random
from numpy.typing import NDArray
from diptest import diptest
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

# ----------------------------------------------------------------------------------------
# General methods
# ----------------------------------------------------------------------------------------
def test_unimodality(x):
    dip, pval = diptest(x)
    return pval >= 0.05


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

def get_important_features(cpss_scores, d, q_hat, theta, error_control_level, B, name):
    worst_case_bound = min(1, 0.5*(1+ (q_hat**2)/(error_control_level*d)))
    
    important_features = pd.DataFrame(
    index=cpss_scores["feature"],
    data={
        "Selection count": cpss_scores["selection_count"].values,
        "Pair count": cpss_scores["pair_count"].values,
        "Unimodal": cpss_scores["is_unimodal"].values,
        "Stability frequence 1 (WC)": 0.0,
        "Stability frequence 2 (UM)": 0.0,
        "Worst Case Bound": False,
        "Unimodal Bound": False,
        "Unimodal Threshold": None
    })
    important_features.index.name = "feature"

    for row in cpss_scores.itertuples(index=False):
        feat = row.feature
        pair_count = row.pair_count
        unimodal = row.is_unimodal
        cpss_score = row.cpss_score

        important_features.loc[feat, "Stability frequence 1 (WC)"] = cpss_score
        if cpss_score >= worst_case_bound:
            important_features.loc[feat, "Worst Case Bound"] = True

        if unimodal:
            stability_freq_unimodal = (1/B)*pair_count
            important_features.loc[feat, "Stability frequence 2 (UM)"] = stability_freq_unimodal
            tau_star = tau_star_unimodal(B, q_hat, s0_size=d, ell=error_control_level, theta=theta)
            if tau_star is not None and stability_freq_unimodal >= tau_star:
                important_features.loc[feat, "Unimodal Bound"] = True
                important_features.loc[feat, "Unimodal Threshold"] = tau_star
    
    important_features = important_features.sort_values('Stability frequence 1 (WC)', ascending=False)
    important_features.to_csv("Significant_features"+name+".csv")
    return important_features

# --------------------
# Unimodality Optimization
# --------------------
def tau_star_unimodal(B: int, q_hat: float, s0_size: int, ell: float, theta: float) -> float | None:
    taus = tau_grid_TB(B)
    factor = (q_hat**2) / s0_size

    feasible = []
    for tau in taus:
        c_val = C_tau_B(tau, B, theta=theta)
        if c_val * factor <= ell:
            feasible.append(tau)

    if not feasible:
        return None
    return min(feasible)

def tau_grid_TB(B: int) -> NDArray[np.float64]:
    start = 0.5 + 1.0 / B          # 1/2 + 1/B
    end = 1.0                      # 1
    n_points = B - 1               # ergibt Schrittweite 1/(2B)
    taus = np.linspace(start, end, n_points)
    return taus

def C_tau_B(tau: float, B: int, theta: float = 1/np.sqrt(3)) -> float:
    if theta > 1/np.sqrt(3):
        raise ValueError("Formula gilt only for theta ≤ 1/√3.")

    lower_1 = min(0.5 + theta**2,
                  0.5 + 1.0/(2*B) + 0.75*theta**2)

    if (tau > lower_1) and (tau <= 0.75):
        denom = 2.0 * (2.0 * tau - 1.0 - 1.0/(2.0 * B))
        if denom <= 0:
            raise ValueError(f"Denominator ≤ 0 for tau={tau}, B={B}")
        return 1.0 / denom

    if (tau > 0.75) and (tau <= 1.0):
        num = 4.0 * (1.0 - tau + 1.0/(2.0 * B))
        denom = 1.0 + 1.0/B
        return num / denom
    
    raise ValueError(
        f"tau={tau} is not in defined intervals acc. Theorem 2"
    )





def cpss_feature_selection(path: Path, XName, B: int, linear):
    X = pd.read_parquet(path / XName)
    y = pd.read_parquet(path / 'y.parquet')

    if 'cell_id' in X.columns:
        X = X.drop(columns=['cell_id'])
    if 'cell_id' in y.columns:
        y = y.drop(columns=['cell_id'])

    d = X.shape[1]
    error_control_level = int(0.05 * d)
    q = int(np.sqrt(0.8 * error_control_level * d))

    print("Testing unimodality for all features...")
    unimodal_flags = {feature: test_unimodality(X[feature].values) for feature in X.columns}

    print(f"\nGenerating {B} complementary pairs...")

    subsamples = getSubsamples(X, y, B)

    print(f"\nRunning CPSS with B={B} pairs...")
    if linear:
        selection_counts, pair_counts = cpss_linear_core(X, y, B, ALPHA_GRID[0], subsamples, q)
        name = 'linear'
    else:
        raise NotImplementedError("Nonlinear path not yet implemented")

    q_hat = sum(selection_counts.values()) / (2*B)
    theta = q_hat / d

    cpss_scores = pd.DataFrame({
        'feature': list(selection_counts.keys()),
        'selection_count': list(selection_counts.values()),
        'pair_count': list(pair_counts.values()),
        'cpss_score': [count / (2 * B) for count in selection_counts.values()],
        'is_unimodal': [unimodal_flags[feat] for feat in selection_counts.keys()]
    }).sort_values('cpss_score', ascending=False)

    cpss_scores.to_csv(f'cpss_scores_{name}.csv')
    
    important_features = get_important_features(cpss_scores, d, q_hat, theta, error_control_level, B, name)
    
    print(f"\nCPSS Results:")
    print(f"  Features (d): {d}")
    print(f"  Target q: {q}")
    print(f"  Observed q̂: {q_hat:.1f}")
    print(f"  θ (q̂/d): {theta:.4f}")
    print(f"  Worst-case τ*: {min(1, 0.5*(1 + q_hat**2/(error_control_level*d))):.3f}")
    print(f"  Features passing worst-case: {important_features['Worst Case Bound'].sum()}")
    print(f"  Features passing unimodal: {important_features['Unimodal Bound'].sum()}")
    
    return cpss_scores, important_features

# ----------------------------------------------------------------------------------------
# Linear Path specific
# ----------------------------------------------------------------------------------------
ALPHA_GRID = [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]
ALPHA = 0.05

def compute_lambda_grid(X, y, n_lambda=100, lambda_min_ratio=0.001):
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
        for feat_idx in selected_both:
            pair_counts[feature_names[feat_idx]] += 1
        
    return selection_counts, pair_counts


# ----------------------------------------------------------------------------------------
# Non Linear Path specific
# ----------------------------------------------------------------------------------------
def rf_base_selector(X, y, target_q, n_estimators=100, max_depth=10, 
                     max_features='sqrt', min_samples_split=5, random_state=42):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X, y)
    
    importances = model.feature_importances_
    ranked_indices = np.argsort(importances)[::-1]
    selected = set(ranked_indices[:target_q])
    
    return selected

def cpss_rf_core(X, y, B, rf_params, subsamples, q):
    if subsamples is None:
        raise ValueError("subsamples cannot be None")
    
    feature_names = X.columns.tolist()
    selection_counts = {feature: 0 for feature in feature_names}
    pair_counts = {feature: 0 for feature in feature_names}

    for j, (A_odd, A_even) in enumerate(subsamples, start=1):
        X_sub_odd = X.iloc[A_odd].values
        y_sub_odd = y.iloc[A_odd].values.ravel()
        selected_odd = rf_base_selector(X_sub_odd, y_sub_odd, q, **rf_params)
        for feat_idx in selected_odd:
            selection_counts[feature_names[feat_idx]] += 1

        X_sub_even = X.iloc[A_even].values
        y_sub_even = y.iloc[A_even].values.ravel()
        selected_even = rf_base_selector(X_sub_even, y_sub_even, q, **rf_params)
        for feat_idx in selected_even:
            selection_counts[feature_names[feat_idx]] += 1
        
        selected_both = selected_odd & selected_even
        for feat_idx in selected_both:
            pair_counts[feature_names[feat_idx]] += 1
        
    return selection_counts, pair_counts
