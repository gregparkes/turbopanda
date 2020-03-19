#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides a basic linear model class for statistical purposes."""

import numpy as np
from scipy import stats


class LinearModel(object):
    """Provides a primitive `LinearModel` template.

    Also provides a large host of properties to access once fitted which
    provide huge statistical use.

    References
    ----------
    Drawn from https://xavierbourretsicotte.github.io/stats_inference_2.html
    """

    def __init__(self, X, y):
        """Initialise the dataset and compute basic values.

        Parameters
        ----------
        X : np.ndarray, pandas.DataFrame
            The inputs to the array
        y : np.ndarray, pandas.Series
            The continuous response vector
        """
        self.is_fit = False
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.p = X.shape[1] - 1
        # degrees of freedom
        self.df = self.n - self.p - 1
        # means
        self.X_bar = np.mean(X, 0)
        self.y_bar = np.mean(y)
        # define attributes here
        self.b, self.H, self.M, self.leverage = None, None, None, None
        self.e = None
        self.y_hat, self.R2, self.R2_a = None, None, None
        self.SS_tot, self.SS_res, self.SS_exp = None, None, None
        self.S2_n_p_1, self.S2_b, self.S_b = None, None, None
        self.b_t_values, self.b_t_p_values, self.F_value, self.F_p_value = None, None, None, None
        self.aic, self.bic, self.aicc = None, None, None
        self.cook = None

    def fit(self):
        """Fit the model using X and y."""
        ninv = np.linalg.pinv(self.X.T @ self.X)

        self.b = ninv @ self.X.T @ self.y
        self.H = self.X @ ninv @ self.X.T
        self.M = np.identity(self.n) - self.H
        self.leverage = np.diag(self.H)

        # predicted values
        self.y_hat = self.X @ self.b
        self.e = self.y - self.y_hat

        # sum of square values
        self.SS_tot = np.sum((self.y - self.y_bar)**2)
        self.SS_res = np.sum(np.square(self.e))
        self.SS_exp = np.sum((self.y_hat - self.y_bar)**2)

        # R2 and the adjusted R2
        self.R2 = self.SS_exp / self.SS_tot
        self.R2_a = (self.R2 * (self.n - 1) - self.p) / self.df

        # variances and SE of coefficients
        self.S2_n_p_1 = self.SS_res / self.df
        self.S2_b = np.diag(self.S2_n_p_1 * ninv)
        self.S_b = np.sqrt(self.S2_b)

        # determine aic, bic
        self.aic = 2*self.p - (2*np.log(self.SS_res))
        self.aicc = self.aic + ((2*self.p)**2 + 2*self.p) / self.df
        self.bic = (np.log(self.n)*self.p) - (2*np.log(self.SS_res))
        # cooks distance
        self.cook = np.square(self.e) / (self.p * (self.SS_res / self.n)) * (self.leverage / np.square(1 - self.leverage))

        # determine probabilities
        self.b_t_values = self.b / self.S_b
        self.b_t_p_values = (1 - stats.t.cdf(np.abs(self.b_t_values), self.df)) * 2
        self.F_value = (self.SS_exp / self.p) / (self.SS_res / self.df)
        self.F_p_value = (1 - stats.f.cdf(self.F_value, self.p, self.df))
        # is fit
        self.is_fit = True
