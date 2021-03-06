#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides a selection of useful plotting functions."""

from ._gridplot import gridplot
from ._histogram import histogram
from ._scatter import scatter, scatter_slim
from ._heatmap import heatmap, hinton
from ._boxplot import box1d, bibox1d, widebox
from ._bar import bar1d, errorbar1d, widebar

# from ._kdeplot import kde2d

from ._annotate import annotate
from ._widgets import legend_line, legend_scatter
from ._palette import *
from ._save_fig import save

# import statements here
from ._visualise import *
