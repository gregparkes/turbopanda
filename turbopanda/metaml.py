#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:01:53 2019

@author: gparkes

Creates Meta Scikit-learn models.
"""
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_predict

from .pipes import ml_pipe
from .metrics import correlate

class MetaML(object):
    """
    An object that handles complex operations on a MetaPanda object to run
    scikit-learn models.
    """

    def __init__(self, mp, X_select, Y_select):
        """
        Receives a MetaPanda object with all of the features ready to go(ish).

        Parameters
        --------
        mp : MetaPanda
            A pandas.DataFrame object containing preprocessed input/output
        X_select : selector
            A selector of x-inputs
        Y_select : selector
            A selector of y-inputs
        """
        # make mp ML-ready
        self.mdf_ = mp.compute_extern(ml_pipe(mp, X_select, Y_select))
        # create X and y
        self.X, self.y = self.mdf_[X_select], self.mdf_[Y_select]

        # defaults
        self.cv = 10
        if self.mdf_.view(Y_select).shape[0] > 1:
            self.lm = MultiOutputRegressor(LinearRegression())
        else:
            self.lm = LinearRegression()

        self.fit = False


    def single_run(self):
        """
        Performs a single run and returns predictions and scores based on defaults.
        """
        # perform cross-validated-predictions
        self.yp = cross_val_predict(self.lm, self.X, self.y, cv=self.cv)
        if self.yp.ndim > 1:
            self.yp = pd.DataFrame(self.yp, columns=self.y.columns)
        else:
            self.yp = pd.Series(self.yp, index=self.X.index)
        # calculate scores
        self.score = correlate(self.y, self.yp, method="r2")
        self.score_mean = self.score["r2"].mean()
        self.fit = True
        return self


    def __repr__(self):
        if self.fit:
            return "MetaML(X: {}, Y: {}, cv={}, score={:0.3f})".format(self.X.shape, self.y.shape, self.cv, self.score_mean)
        else:
            return "MetaML(X: {}, Y: {}, cv={})".format(self.X.shape, self.y.shape, self.cv)
