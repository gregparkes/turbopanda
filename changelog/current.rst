.. include:: _commands.rst

Version 0.2.6
=============
**23rd Apr 2020**

Changelog
---------
- Added `dimension_reduction` notebook
- `vectorize` method moved into the main folder from `.dev`
- `unimplemented` method for users to see when functions haven't been completed
- General cleaning of code to remove duplicates
- Slowly replacing use of `Pipe` class with `.pipe` library.

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
- |Enhancement| `.plot` methods now use boxplots from the `.plot` library
- |Fix| Many bugs in `.plot.pca_overview' which wasn't plotting eigenvectors properly

`turbopanda.pipe`
.................
- |Feature| Added `absolute` and `filter_rows_by_column` methods to chain together common pandas
operations

`turbopanda.plot`
.................
- |MajorFeature| Added `box1d`, `bibox1d` and `widebox` to enable easy boxplot drawing.
- |Feature| Added a number of shading methods to do with *luminance*, such as `autoshade`,
`contrast` and `noncontrast` within the palette
- |Enhancement| `gridplot` now annotates plots with A..K for papers
- |Fix| `histogram` now handles `freeform` argument for kde properly when adding the x-label

`turbopanda.str`
................
- |Feature| Added `patcolumnmatch` to match column names in a dataframe using regex to clean code
- |Fix| `shorten` now properly uses the parameters passed to it

`turbopanda.utils`
..................
- |Feature| Added `upcast` method to convert lists to numpy, numpy to pandas, etc.
- |Feature| Added `arrays_dimension` method for checking dimensions on numpy, pandas objects

`turbopanda.validate`
.....................
