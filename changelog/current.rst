.. include:: _commands.rst

Version 0.2.5
=============
**7th Apr 2020**

Changelog
---------
- Added scatterpoints example notebook

`turbopanda.corr`
.................
- Removed `pcm`

`turbopanda.merge`
..................

`turbopanda.metapanda`
......................

`turbopanda.ml`
...............
- |Efficiency| Folder reorganized into *.fit* and *.plot* submodules

`turbopanda.pipe`
.................
- |Feature| Added `yeo_johnson`, `zscore` operations

`turbopanda.plot`
.................
- |Feature| Added `scatter` for normal and density-based scatter plots
- |Enhancement| `hist_grid`, `scatter_grid` now also accept DataFrame as arguments
- Added a bunch of functions to the palette selection
- Split `legend` into `legend_line` and `legend_scatter`

`turbopanda.stats`
..................

`turbopanda.str`
................
- |Feature| Added `common_substrings` which encompasses single and cartesian product-like input
- `common_substring_match` and its pairwise form are now deprecated

`turbopanda.utils`
..................
- |Feature| `array_equal_sizes` check for k arrays of equal dimensions
- |Feature| `nonnegative` check for integers, `disallow_instance_pair` to prevent weird pair combinations for input
- |Enhancement| `instance_check` now performs checks on groups of objects using the same type check
- `zfilter`, `standardize` is now deprecated
