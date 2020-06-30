.. include:: _commands.rst

Version 0.2.7
=============
**6th May 2020**

Changelog
---------
- |MajorFeature| `melt` now supports complex `pandas.melt` operations, including
regex (flex) support and more. Includes an example.
- |Enhancement| `vectorize` now supports caching for each step
- |Fix| keyword argument `cache`, default set to False

`turbopanda.dev`
................
- |Efficiency| `cache` for non-pandas objects uses `joblib` to serialize/pickle the object

`turbopanda.plot`
.................
- |Feature| Added `errorbar1d` function to draw barplots with errorbars.
- |Feature| Added `bar1d` function to efficiently draw barplots, with automatic sorting and colouring
- |Feature| Added `widebar` to provide a hued barplot functionality instinctively to pandas.DataFrames.
- |Enhancement| `widebox` function now has a `sort` parameter which sorts variables by the mean.
- |Fix| An issue in `scatter` where colors were not corresponding to legend

`turbopanda.str`
................
- |Enhancement| Added `pattern`, collapsing down `strpattern` and `patcolumnmatch` functions
into a catch-all operation
- |Fix| `common_substrings` now properly handles cases where list a only has one element, with no b. This simply
returns the string in question

`turbopanda.merge`
- |Enhancement| Includes proper verbose outputs to see which columns are merging together

`turbopanda.utils`
- |Feature| Added `umap` and variants to add with maps, caching and parallelism in loops
- |Feature| Added `absdifference` to the .sets methods, to handle absolute set difference (we had
issues with symmetric difference only in the past)
