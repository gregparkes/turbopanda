# import

""" This block handles the import needs of the package """
hard_dependencies = ("numpy", "scipy", "pandas", "sklearn")
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

# import local objects

from .metapanda import MetaPanda
from .fileio import *
from .visualise import *

__version__ = "0.1.0"
__name__ = "turbopanda"
