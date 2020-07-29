.. include:: _commands.rst

Version 0.2.8
=============
**17th July 2020**

Changelog
---------
- |Enhancement| `turb.read` now accepts `.pkl` files using joblib

`turbopanda.MetaPanda`
......................
- |Feature| Added `shape` property to access df shape under the hood
- |Enhancement| `source_` setter property now writes a file if not present
- |Enhancement| `MetaPanda.write` now accepts `.pkl` files, and dumps using joblib
- |Fix| using the `__setitem__` property now works, so column assignment is functional

`turbopanda.dev`
................
- |API| `cached` and `cached_chunk` are now deprecated, to be removed in 0.3

`turbopanda.ml`
...............
- |API| `ml_ready` and `get_best_model` is now deprecated, to be removed in 0.3

`turbopanda.pipe`
.................
- |Feature| Added `impute_missing` to impute missing values to continous, integer columns.

`turbopanda.plot`
.................
- |Enhancement| `widebar` now has keyword argument allowance.
- |Fix| `cat_array_to_color` now handles numpy 'O' type object inputs correctly.
- |API| Functions in _palette now accessible through the `turbopanda.plot` API

`turbopanda.str`
................
- |Fix| `pattern` now drops NA values when receives a pandas.Series object before regex search
- |API| `pattern` docs now has some examples to learn from

`turbopanda.utils`
..................
- |Feature| Added `zipe` function, with extended zip functionality. Comes with examples
- |Enhancement| `cache` now exists as a util option, gradually making `dev.cache` redundant.
- |Efficiency| `intersect` and `union` now use `reduce` instead of for loops
- |Fix| `arrays_dimension` now properly handles DataFrames
- |Fix| `umap` now checks that all elements passed to f(x) are iterable
- |API| `lparallel` is now deprecated, to be removed in 0.3
