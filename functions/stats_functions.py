import numpy as np
import itertools
from scipy.stats import ttest_rel

def compute_significant_pairs(metric, names, alpha=0.05):
    """
    metric : ndarray, shape (n_subjects, n_regions)
    names  : list of length n_regions with region names
    alpha  : significance threshold after Bonferroni correction

    Returns:
    --------
    sig_pairs   : list of (region1, region2, p_corr) with p_corr < alpha
    all_results : list of (region1, region2, p_raw, p_corr) for every pair
    """
    n_regions = metric.shape[1]
    pairs = list(itertools.combinations(range(n_regions), 2))
    
    # 1) run raw tests
    results = []
    for i, j in pairs:
        x = metric[:, i]
        y = metric[:, j]
        # drop any subject where either is NaN
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_, y_ = x[mask], y[mask]
        
        # paired t-test
        _, p_raw = ttest_rel(x_, y_)
        results.append((i, j, p_raw))
    
    m = len(results)  # number of comparisons
    
    # 2) Bonferroni correction & collect
    sig_pairs   = []
    all_results = []
    for i, j, p_raw in results:
        p_corr = min(p_raw * m, 1.0)
        all_results.append((names[i], names[j], p_raw, p_corr))
        #if p_corr < alpha:
        sig_pairs.append((names[i], names[j], p_corr))
    
    return sig_pairs, all_results
