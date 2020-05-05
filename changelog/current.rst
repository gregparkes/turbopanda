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
- |Enhancement| `correlate` now has a `parallel` parameter to calculate bicorrelates in parallel
using joblib.

`turbopanda.dev`
................
- |Enhancement| `cached_chunk` now has optional parallelization using joblib when chunking

`turbopanda.merge`
..................

`turbopanda.metapanda`
......................
- meta calculations make use of `.str.common_substrings` rather than the pairwise implementation
as before.

`turbopanda.ml`
...............
- |Enhancement| `.fit.basic` and `.fit.grid` have changed parameter ordering,
and accept pandas.DataFrame objects
- |Enhancement| `pca` now is vectorizable, with the addition of `stratified_pca`
to streamline this
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
- |Fix| `common_substrings` now properly filters substrings as needed
- `score_pairwise_common_substring` is now deprecated

`turbopanda.utils`
..................
- |Feature| Added `upcast` method to convert lists to numpy, numpy to pandas, etc.
- |Feature| Added `arrays_dimension` method for checking dimensions on numpy, pandas objects
- |Feature| Added `bounds_check` function to check numeric bounding of functions
