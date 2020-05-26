#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles dimensionality reduction methods.

These methods include:

- Missing Value Filter
- Low Variance Filter
- High Correlation Filter
- Random Forest
- Reverse Feature Elimination (RFE)
- Forward Feature Selection (ANOVA)
- Factor Analysis
- PCA
- ICA
- Manifold Isomap
- t-SNE
- UMAP
"""

import numpy as np

__all__ = ('missing_value_filter', 'low_variance_filter', 'high_correlation_filter')


def missing_value_filter(df, threshold=0.5):
    """Drops columns that have more than threshold proportion of missing values."""
    return df.drop(df.columns[1 - (df.count() / df.shape[0] > threshold)], axis=1)


def low_variance_filter(df, threshold=1e-6):
    """Drops columns that have more than threshold proportion of missing values."""
    float_cols = df.select_dtypes(float)
    # maintains non-float columns, only drops from float features
    droppable_cols = float_cols.columns[float_cols.var() < threshold]
    return df.drop(float_cols.columns[float_cols.var() < threshold], axis=1)


def high_correlation_filter(df, threshold=0.6):
    """Drops columns that have a very high correlation r2 magnitude to too many features."""
    float_cols = df.select_dtypes(float)
    sel = np.mean(np.square(float_cols.corr(method="spearman")) - np.eye(float_cols.shape[1])) > (threshold**2)
    return df.drop(df.columns[sel], axis=1)
