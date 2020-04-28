.. include:: _commands.rst

Version 0.2.6
=============
**23rd Apr 2020**

Changelog
---------
- Added `dimension_reduction` notebook

`turbopanda.corr`
.................

`turbopanda.dev`
................
- |Enhancement| `cached_chunk` now has optional parallelization using joblib when chunking

`turbopanda.merge`
..................

`turbopanda.metapanda`
......................

`turbopanda.ml`
...............
- |Enhancement| `.fit.basic` and `.fit.grid` have changed parameter ordering,
and accept pandas.DataFrame objects
- |Fix| Many bugs in `.plot.pca_overview' which wasn't plotting eigenvectors properly

`turbopanda.pipe`
.................

`turbopanda.plot`
.................
- |Fix| `histogram` now handles `freeform` argument for kde properly when adding the x-label

`turbopanda.str`
................
- |Feature| Added `patcolumnmatch` to match column names in a dataframe using regex to clean code

`turbopanda.utils`
..................
- |Feature| Added `upcast` method to convert lists to numpy, numpy to pandas, etc.
- |Feature| Added `arrays_dimension` method for checking dimensions on numpy, pandas objects

`turbopanda.validate`
.....................
