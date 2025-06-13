
"""
Simplified standalone scalability coefficient estimation.

This module provides a minimal implementation for computing item-level 
scalability coefficients (Hi and Zi) without confidence intervals or 
multilevel modeling. Designed for portability and simplicity.

Author: Michael Hardy
"""
import numpy as np
import pandas as pd
from typing import Union, Dict

def scalability_coefs(X: Union[np.ndarray, pd.DataFrame]) -> Dict:
    """
    Compute item-level scalability coefficients (Hi and Zi) using simplified approach.
    (Loevinger, 1948; Mokken, 1971; Molenaar and Sijtsma, 2000; Sijtsma and Molenaar, 2002)

    This function computes:
    - Hi: Item-level H coefficients (scalability of each item with rest of scale)
    - Zi: Item-level Z-scores (standardized Hi coefficients)
    
    Parameters
    ----------
    X : array-like of shape (n_subjects, n_items)
        Data matrix containing item responses. Should be integer-valued.
        Missing values are handled by listwise deletion.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'Hi': Item-level H coefficients (array of length n_items)
        - 'Zi': Item-level Z-scores (array of length n_items)
        - 'H': Overall scale H coefficient (scalar)
        - 'Z': Overall scale Z-score (scalar)
    
    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randint(0, 4, (100, 5))
    >>> result = scalability_simple(X)
    >>> print(f"Item coefficients: {result['Hi']}")
    >>> print(f"Overall coefficient: {result['H']:.3f}")
    """
    # Convert input to numpy array
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.asarray(X, dtype=float)
    
    # Handle missing data with listwise deletion
    if np.any(np.isnan(X)):
        complete_cases = ~np.any(np.isnan(X), axis=1)
        X = X[complete_cases]
        if X.shape[0] < 5:
            raise ValueError("Insufficient complete cases after removing missing data")
    
    # Convert to integers
    X = X.astype(int)
    
    # Validate input
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[1] < 2:
        raise ValueError("X must have at least 2 items")
    if X.shape[0] < 5:
        raise ValueError("X must have at least 5 subjects")
    
    n_subjects, n_items = X.shape
    
    # Check for zero variance, handle with listwise deletion
    if np.any(np.var(X, axis=0) == 0):
        complete_cases = ~np.any(np.var(X, axis=0) == 0, axis=1)
        X = X[complete_cases]
        if X.shape[0] < 5:
            raise ValueError("Insufficient complete cases after removing zero variance items")

    # Compute H scaling (Loevinger, 1948; Mokken, 1971) using simple method 
    # Compute covariance matrices
    S = np.cov(X, rowvar=False)  # Item covariance matrix
    X_sorted = np.sort(X, axis=0)  # Sort each item independently
    Smax = np.cov(X_sorted, rowvar=False)  # Maximum possible covariance
    
    # Compute Hij matrix (item-pair coefficients)
    Hij = S / Smax
    np.fill_diagonal(Hij, 0)  # Zero out diagonal
    
    # Compute Hi coefficients (item-level)
    S_offdiag = S.copy()
    Smax_offdiag = Smax.copy()
    np.fill_diagonal(S_offdiag, 0)
    np.fill_diagonal(Smax_offdiag, 0)

    ## for future reference:
    Hij = np.divide(S_offdiag, Smax_offdiag, 
                   out=np.zeros_like(S_offdiag), where=Smax_offdiag != 0)
    Hi = np.sum(Hij, axis=1)
    
    
    # Hi = np.sum(S_offdiag, axis=1) / np.sum(Smax_offdiag, axis=1)
    
    # Compute overall H coefficient
    H = np.sum(S_offdiag) / np.sum(Smax_offdiag)
    
    # Compute Z-standardized scaling using simple method 
    # (Mokken, 1971; Molenaar and Sijtsma, 2000; Sijtsma and Molenaar, 2002)
    # Only appropriate for testing lowerbound = 0.
    var_vec = np.var(X, axis=0, ddof=1)  # Item variances, unweighted and unbiased
    Sij = np.outer(var_vec, var_vec)  # Outer product of variances
    
    # Item-pair Z-standardized scaling coefficients
    Zij = np.divide(S * np.sqrt(n_subjects - 1), np.sqrt(Sij), 
                   out=np.zeros_like(S_offdiag), where=Sij != 0)
    np.fill_diagonal(Zij, 0)  # Zero diagonal
    
    # Item-level Z-standardized scaling
    # S_for_z = S.copy()
    Sij_for_z = Sij.copy()
    # np.fill_diagonal(S_for_z, 0)
    np.fill_diagonal(Sij_for_z, 0)
    
    Zi = np.divide(np.sum(S_offdiag, axis=1) * np.sqrt(n_subjects - 1), 
                  np.sqrt(np.sum(Sij_for_z, axis=1)),
                  out=np.zeros(n_items), where=np.sum(Sij_for_z, axis=1) != 0)
    
    # Overall Z-standardized scaling (divided by 2 because the matrix is symmetric, I think)
    sum_S = np.sum(S_offdiag) / 2.0
    sum_Sij = np.sum(Sij_for_z) / 2.0
    Z = (sum_S * np.sqrt(n_subjects - 1)) / np.sqrt(sum_Sij) if sum_Sij != 0 else 0.0
    
    return {
        'Hi': Hi,
        'Zi': Zi,
        'H': H,
        'Z': Z,
        'Hij': Hij,  
        'Zij': Zij   
    }
