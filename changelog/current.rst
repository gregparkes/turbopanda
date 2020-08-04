.. include:: _commands.rst

Version 0.2.8
=============
**17th July 2020**

Changelog
---------
- |Enhancement| `turb.read` now accepts `.pkl` files using joblib
- |API| Added tqdm library as requirement

`turbopanda.MetaPanda`
......................
- |Feature| Added `shape` property to access df shape under the hood
- |Enhancement| `source_` setter property now writes a file if not present
- |Enhancement| `MetaPanda.write` now accepts `.pkl` files, and dumps using joblib
- |Fix| using the `__setitem__` property now works, so column assignment is functional
- |API| `add_prefix` and `add_suffix` to be removed next version

`turbopanda.dev`
................
- |API| `cached` and `cached_chunk` are now deprecated, to be removed in 0.3

`turbopanda.ml`
...............
- |API| `ml_ready` and `get_best_model` is now deprecated, to be removed in 0.3

`turbopanda.pipe`
.................
- |Feature| Added `impute_missing` to impute missing values to continous, integer columns.
- |Feature| Added `add_prefix`, `add_suffix` to pipeable methods
- Added `index_name` and `krange` functions

`turbopanda.plot`
.................
- |Enhancement| `widebar` now has keyword argument allowance.
- |Enhancement| `bar1d` has a scale parameter to do log transform plots
- |Enhancement| `legend` now allows for ncol and title selection
- |Fix| `cat_array_to_color` now handles numpy 'O' type object inputs correctly.
- |API| `scatter` now has x_label and y_label arguments as optional
- |API| Functions in _palette now accessible through the `turbopanda.plot` API

`turbopanda.str`
................
- |Fix| `pattern` now drops NA values when receives a pandas.Series object before regex search
- |API| `pattern` docs now has some examples to learn from

`turbopanda.utils`
..................
- |MajorFeature| Added `zipe` function, with extended zip functionality. Comes with examples
- |Enhancement| `cache` now exists as a util option, gradually making `dev.cache` redundant
- |Enhancement| `umap` and `umapc` now have in-built progress bar using tqdm library
- |Efficiency| `intersect` and `union` now use `reduce` instead of for loops
- |Fix| `arrays_dimension` now properly handles DataFrames
- |Fix| `umap` now checks that all elements passed to f(x) are iterable
- |API| `lparallel` is now deprecated, to be removed in 0.3
