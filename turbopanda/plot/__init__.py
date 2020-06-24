#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides a selection of useful plotting functions."""

from ._gridplot import gridplot
from ._histogram import histogram
from ._scatter import scatter
from ._heatmap import heatmap
from ._boxplot import box1d, bibox1d, widebox
from ._bar import bar1d, errorbar1d, widebar
# from ._kdeplot import kde2d

from ._annotate import annotate
from ._widgets import legend_line, legend_scatter
from ._palette import color_qualitative
from ._save_fig import save
# import statements here
from ._visualise import *
