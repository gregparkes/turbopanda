.. include:: _commands.rst

Version 0.2.9
=============
**21st April 2021**

Changelog
---------
- |Enhancement| `joblib` and `tqdm` are now 'optional' modules in most cases.

`turbopanda.MetaPanda`
......................

`turbopanda.corr`
.................
- |Enhancement| `bicorr` now handles category-category and continuous-category combinations.
- |Fix| `entropy`, `continuous_mutual_info` now add small eps to prevent NaN results for log2(0) cases.

`turbopanda.dev`
................

`turbopanda.ml`
...............

`turbopanda.pipe`
.................

`turbopanda.plot`
.................
- |Fix| `scatter` now accepts X and Y with missing values.

`turbopanda.str`
................
- |Fix| `pattern` now properly handles cases where a DataFrame is passed with ~ not operator, and single non-split cases.

`turbopanda.utils`
..................
- |Enhancement| `cache` now checks the filepath and can create folders before writing
- |Feature| Added `check_file_path` method to ensure that a path to a file is real even if the file itself exists

`turbopanda.sample`
...................
- |Fix| `covariance_matrix` now properly handle numba dependency