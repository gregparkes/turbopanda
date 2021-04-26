.. include:: _commands.rst

Version 0.2.9
=============
**21st April 2021**

Changelog
---------

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

`turbopanda.str`
................
- |Fix| `pattern` now properly handles cases where a DataFrame is passed with ~ not operator.

`turbopanda.utils`
..................

`turbopanda.sample`
...................
