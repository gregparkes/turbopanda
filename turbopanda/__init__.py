# check imports

""" This block handles the import needs of the package """
hard_dependencies = ("numpy", "scipy", "pandas", "matplotlib", "sklearn", "joblib")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append("{0}: {1}".format(dependency, str(e)))

if missing_dependencies:
    raise ImportError("Unable to import required dependencies:\n" + "\n".join(missing_dependencies))
del hard_dependencies, dependency, missing_dependencies
""" Block ends """

# folder extensions.
from ._fileio import read
from ._merge import merge
from ._vectorize import vectorize, Param
# import Objects and critical global functions
from ._metapanda import MetaPanda
# vectorization methods
from .dev import cache, cached
from ._pipe import Pipe
from .corr import correlate

from . import corr, dev, ml, plot, stats, str, utils, pipe

__version__ = "0.2.6"
__name__ = "turbopanda"
__doc__ = """turbopanda: Turbo-charging the Pandas library in an integrative, meta-orientated style.

The aim of this library is extend the functionality of the `pandas` library package,
which is extensively used for data munging,
manipulation and visualization of large datasets. There are a number of areas that the `pandas` library is
lacklustre from a user standpoint - we'll cover a few of these in more detail and then
explain `turbopanda` response to these particular issues.
"""
