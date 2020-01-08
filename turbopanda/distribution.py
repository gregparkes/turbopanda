#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:55:47 2019

@author: gparkes

Taken from blog: http://www.insightsbot.com/blog/WEjdW/fitting-probability-distributions-with-python-part-1
"""
# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# imports
from scipy import stats
import matplotlib.pyplot as plt


class Distribution(object):
    """
    Distributions make use of the scipy.stats module to assess the continuous distribution of a given
    Series of data, and fits the best distribution to it where possible.

    Distributions currently checked include:
        {'norm', 'lognorm', 'expon', 'pareto', 'logistic', 't', 'laplace', 'uniform', 'halfnorm'}

    Includes generating random numbers and plotting methods.
    """
    def __init__(self, dist_names=()):
        """
        Initialises a Distribution object, specifying the distribution space to be searched.
        """
        self.dist_names = dist_names if len(dist_names) > 0 else (
            "norm", "lognorm", "expon", "pareto",
            "logistic", "t", "laplace", "uniform",
            "halfnorm"
        )
        self.dist_name = ""
        self.PValue = 0
        self.Param = None
        self.isFitted = False

    def fit(self, y):
        self.dist_results = []
        self.params = {}

        for dist_name in self.dist_names:
            my_dist = getattr(stats, dist_name)
            param = my_dist.fit(y)

            self.params[dist_name] = param
            # apply KS-test
            d, p = stats.kstest(y, dist_name, args=param)
            self.dist_results.append((dist_name, p))

        # select the best-fitted distribution
        sel_dist, p = (max(self.dist_results, key=lambda item: item[1]))
        # store best name
        self.dist_name = sel_dist
        self.PValue = p

        self.isFitted = True
        return self.dist_name, self.PValue

    def random(self, n=1):
        if self.isFitted:
            dist = getattr(stats, self.dist_name)
            param = self.params[self.dist_name]
            return dist.rvs(*param[:-2], loc=param[-2], scale=param[-1], size=n)
        else:
            raise ValueError("Distribution is not fitted.")

    def Plot(self, y):
        x = self.random(n=len(y))
        plt.hist(x, alpha=.5, label="Fitted")
        plt.hist(y, alpha=.5, label="Actual")
        plt.legend(loc="upper right")
