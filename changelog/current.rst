.. include:: _commands.rst

Version 0.2.8
=============
**17th July 2020**

Changelog
---------
- |Enhancement| `turb.read` now accepts `.pkl` files using joblib

`turbopanda.MetaPanda`
......................
- |Enhancement| `source_` setter property now writes a file if not present
- |Enhancement| `MetaPanda.write` now accepts `.pkl` files, and dumps using joblib

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

`turbopanda.utils`
..................
- |Enhancement| `cache` now exists as a util option, gradually making `dev.cache` redundant.
- |Fix| `arrays_dimension` now properly handles DataFrames
- |Fix| `umap` now checks that all elements passed to f(x) are iterable
- |API| `lparallel` is now deprecated, to be removed in 0.3
