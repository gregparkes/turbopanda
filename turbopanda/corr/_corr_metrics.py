#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Additional correlation metrics. Taken from pingouin package."""
# imports
import numpy as np
from scipy.stats import chi2, pearsonr, spearmanr, t

from turbopanda._dependency import is_sklearn_installed

__all__ = ("skipped", "bsmahal", "shepherd", "percbend")


def skipped(x, y, method="spearman"):
    """
    Skipped correlation (Rousselet and Pernet 2012).
    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be independent.
    method : str
        Method used to compute the correlation after outlier removal. Can be
        either 'spearman' (default) or 'pearson'.
    Returns
    -------
    r : float
        Skipped correlation coefficient.
    pval : float
        Two-tailed p-value.
    outliers : array of bool
        Indicate if value is an outlier or not
    Notes
    -----
    The skipped correlation involves multivariate outlier detection using a
    projection technique (Wilcox, 2004, 2005). First, a robust estimator of
    multivariate location and scatter, for instance the minimum covariance
    determinant estimator (MCD; Rousseeuw, 1984; Rousseeuw and van Driessen,
    1999; Hubert et al., 2008) is computed. Second, data points are
    orthogonally projected on lines joining each of the data point to the
    location estimator. Third, outliers are detected using a robust technique.
    Finally, Spearman correlations are computed on the remaining data points
    and calculations are adjusted by taking into account the dependency among
    the remaining data points.
    Code inspired by Matlab code from Cyril Pernet and Guillaume
    Rousselet [1]_.
    Requires scikit-learn.
    References
    ----------
    .. [1] Pernet CR, Wilcox R, Rousselet GA. Robust Correlation Analyses:
       False Positive and Power Validation Using a New Open Source Matlab
       Toolbox. Frontiers in Psychology. 2012;3:606.
       doi:10.3389/fpsyg.2012.00606.
    """
    # Check that sklearn is installed
    is_sklearn_installed(raise_error=True)
    from sklearn.covariance import MinCovDet

    X = np.column_stack((x, y))
    nrows, ncols = X.shape
    gval = np.sqrt(chi2.ppf(0.975, 2))

    # Compute center and distance to center
    center = MinCovDet(random_state=42).fit(X).location_
    B = X - center
    B2 = B ** 2
    bot = B2.sum(axis=1)

    # Loop over rows
    dis = np.zeros(shape=(nrows, nrows))
    for i in np.arange(nrows):
        if bot[i] != 0:
            dis[i, :] = np.linalg.norm(B * B2[i, :] / bot[i], axis=1)

    # Detect outliers
    def idealf(x):
        """Compute the ideal fourths IQR (Wilcox 2012)."""
        n = len(x)
        j = int(np.floor(n / 4 + 5 / 12))
        y = np.sort(x)
        g = (n / 4) - j + (5 / 12)
        low = (1 - g) * y[j - 1] + g * y[j]
        k = n - j + 1
        up = (1 - g) * y[k - 1] + g * y[k - 2]
        return up - low

    # One can either use the MAD or the IQR (see Wilcox 2012)
    # MAD = mad(dis, axis=1)
    iqr = np.apply_along_axis(idealf, 1, dis)
    thresh = np.median(dis, axis=1) + gval * iqr
    outliers = np.apply_along_axis(np.greater, 0, dis, thresh).any(axis=0)

    # Compute correlation on remaining data
    if method == "spearman":
        r, pval = spearmanr(X[~outliers, 0], X[~outliers, 1])
    else:
        r, pval = pearsonr(X[~outliers, 0], X[~outliers, 1])
    return r, pval, outliers


def bsmahal(a, b, n_boot=200):
    """
    Bootstraps Mahalanobis distances for Shepherd's pi correlation.
    Parameters
    ----------
    a : ndarray (shape=(n, 2))
        Data
    b : ndarray (shape=(n, 2))
        Data
    n_boot : int
        Number of bootstrap samples to calculate.
    Returns
    -------
    m : ndarray (shape=(n,))
        Mahalanobis distance for each row in a, averaged across all the
        bootstrap resamples.
    """
    n, m = b.shape
    MD = np.zeros((n, n_boot))
    nr = np.arange(n)
    xB = np.random.choice(nr, size=(n_boot, n), replace=True)

    # Bootstrap the MD
    for i in np.arange(n_boot):
        s1 = b[xB[i, :], 0]
        s2 = b[xB[i, :], 1]
        X = np.column_stack((s1, s2))
        mu = X.mean(0)
        _, R = np.linalg.qr(X - mu)
        sol = np.linalg.solve(R.T, (a - mu).T)
        MD[:, i] = np.sum(sol ** 2, 0) * (n - 1)

    # Average across all bootstraps
    return MD.mean(1)


def shepherd(x, y, n_boot=200):
    """
    Shepherd's Pi correlation, equivalent to Spearman's rho after outliers
    removal.
    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be independent.
    n_boot : int
        Number of bootstrap samples to calculate.
    Returns
    -------
    r : float
        Pi correlation coefficient
    pval : float
        Two-tailed adjusted p-value.
    outliers : array of bool
        Indicate if value is an outlier or not
    Notes
    -----
    It first bootstraps the Mahalanobis distances, removes all observations
    with m >= 6 and finally calculates the correlation of the remaining data.
    Pi is Spearman's Rho after outlier removal.
    """
    X = np.column_stack((x, y))

    # Bootstrapping on Mahalanobis distance
    m = bsmahal(X, X, n_boot)

    # Determine outliers
    outliers = m >= 6

    # Compute correlation
    r, pval = spearmanr(x[~outliers], y[~outliers])

    # (optional) double the p-value to achieve a nominal false alarm rate
    # pval *= 2
    # pval = 1 if pval > 1 else pval
    return r, pval, outliers


def percbend(x, y, beta=0.2):
    """
    Percentage bend correlation (Wilcox 1994).
    Parameters
    ----------
    x, y : array_like
        First and second set of observations. x and y must be independent.
    beta : float
        Bending constant for omega (0 <= beta <= 0.5).
    Returns
    -------
    r : float
        Percentage bend correlation coefficient.
    pval : float
        Two-tailed p-value.
    Notes
    -----
    Code inspired by Matlab code from Cyril Pernet and Guillaume Rousselet.
    References
    ----------
    .. [1] Wilcox, R.R., 1994. The percentage bend correlation coefficient.
       Psychometrika 59, 601–616. https://doi.org/10.1007/BF02294395
    .. [2] Pernet CR, Wilcox R, Rousselet GA. Robust Correlation Analyses:
       False Positive and Power Validation Using a New Open Source Matlab
       Toolbox. Frontiers in Psychology. 2012;3:606.
       doi:10.3389/fpsyg.2012.00606.
    """
    X = np.column_stack((x, y))
    nx = X.shape[0]
    M = np.tile(np.median(X, axis=0), nx).reshape(X.shape)
    W = np.sort(np.abs(X - M), axis=0)
    m = int((1 - beta) * nx)
    omega = W[m - 1, :]
    P = (X - M) / omega
    P[np.isinf(P)] = 0
    P[np.isnan(P)] = 0

    # Loop over columns
    a = np.zeros((2, nx))
    for c in [0, 1]:
        psi = P[:, c]
        i1 = np.where(psi < -1)[0].size
        i2 = np.where(psi > 1)[0].size
        s = X[:, c].copy()
        s[np.where(psi < -1)[0]] = 0
        s[np.where(psi > 1)[0]] = 0
        pbos = (np.sum(s) + omega[c] * (i2 - i1)) / (s.size - i1 - i2)
        a[c] = (X[:, c] - pbos) / omega[c]

    # Bend
    a[a <= -1] = -1
    a[a >= 1] = 1

    # Get r, tval and pval
    a, b = a
    r = (a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum())
    tval = r * np.sqrt((nx - 2) / (1 - r ** 2))
    pval = 2 * t.sf(abs(tval), nx - 2)
    return r, pval
