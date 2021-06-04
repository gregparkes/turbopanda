.. include:: _commands.rst

Version 0.2.9
=============
**21st April 2021**

Changelog
---------
- |Enhancement| `joblib` and `tqdm` are now 'optional' modules in most cases.
- |Fix| `requires` can now accept multiple arguments

`turbopanda.MetaPanda`
......................

`turbopanda.corr`
.................
- |Enhancement| `bicorr` now handles category-category and continuous-category combinations.
- |Fix| `entropy`, `continuous_mutual_info` now add small eps to prevent NaN results for log2(0) cases.

`turbopanda.stats`
..................
- |Feature| Added `is_mvn` to collect normally distributed columns from a pandas dataframe.

`turbopanda.plot`
.................
- |Fix| `scatter` now accepts X and Y with missing values.
- |Fix| `annotate` now takes a copy of X and Y, preventing a bug where the data underlying was changing.
- |Fix| `palette` uses `map` now instead of `umap`

`turbopanda.str`
................
- |Fix| `pattern` now properly handles cases where a DataFrame is passed with ~ not operator, and single non-split cases.

`turbopanda.utils`
..................
- |Enhancement| `tqdm` now works properly with parallelized loops with `joblib`
- |Enhancement| `cache` now checks the filepath and can create folders before writing
- |Feature| Added `check_file_path` method to ensure that a path to a file is real even if the file itself exists
- |Feature| Added `umap_validate` which reads in 'partial' chunk run files for analysis.
- |Fix| `umap` and other functions now incorporate tqdm, filepaths fixed properly

`turbopanda.sample`
...................
- |MajorFeature| `matrix_homo` and `matrix_hetero` allow for creation of toy data matrices X with homoscedastic and heteroscedastic noise.
- |Fix| `covariance_matrix` now properly handle numba dependency
