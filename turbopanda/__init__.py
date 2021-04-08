"""Module level file. """

# folder extensions.
from ._fileio import read
from ._merge import merge
from ._vectorize import vectorize, Param
from ._melt import melt

# import Objects and critical global functions
from ._metapanda import MetaPanda

# vectorization methods
from .dev import cache, cached
from ._pipe import Pipe
from .corr import correlate

from . import corr, dev, ml, plot, stats, str, utils, pipe, sample


__version__ = "0.2.8"
__name__ = "turbopanda"
__doc__ = """turbopanda: Turbo-charging the Pandas library in an integrative, meta-orientated style.

The aim of this library is extend the functionality of the `pandas` library package,
which is extensively used for data munging,
manipulation and visualization of large datasets. There are a number of areas that the `pandas` library is
lacklustre from a user standpoint - we'll cover a few of these in more detail and then
explain `turbopanda` response to these particular issues.
"""
