.. include:: _commands.rst

Version 0.2.7
=============
**6th May 2020**

Changelog
---------
- |Enhancement| `vectorize` now supports caching for each step
keyword argument `cache`, default set to False

`turbopanda.dev`
................
- |Enhancement| `cache` for non-pandas objects uses `joblib` to serialize/pickle the object

`turbopanda.plot`
.................
- |Feature| Added `errorbar1d` function to draw barplots with errorbars.
- |Feature| Added `bar1d` function to efficiently draw barplots, with automatic sorting and colouring
- |Enhancement| `widebox` function now has a `sort` parameter which sorts variables by the mean.
- |Fix| An issue in `scatter` where colors were not corresponding to legend

`turbopanda.merge`
- |Enhancement| Includes proper verbose outputs to see which columns are merging together

`turbopanda.utils`
- |Feature| Added `umap` and variants to add with maps, caching and parallelism in loops
