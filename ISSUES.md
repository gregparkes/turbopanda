## Known Issues with `pandas 0.2.5`:

Here we will cover some personal and more established gripes with `pandas`:

#### The verbose selection of columns

In Pandas, the main accessor-based methods
have very clunky notation of accessing subsets of column-data using the `.loc` and `.iloc`
accessors. `.iloc` is almost useless as it is dependent on the ordering of the columns, which
is usually unknown, and `.loc` works fairly well, except you have to write out the 
whole column name, or spend some time constructing a list of column names selected, for example,
in the following fashion:

```python
# where df is our pandas.DataFrame
import pandas as pd
df = pd.read_csv("some_data.csv")
df_subset = df.loc[:, df.columns[df.columns.str.contains('something_interesting_pattern')]]
```

Why on earth do I have to write so much just to select all columns with an 
interesting regex pattern?
In addition, what about if I have two selection string criteria, or three? Very quickly this
becomes a mountain of code just to get a specific subset of columns I'm interested in.

How we fix this in `turbopanda` is as follows:

```python
# mdf is our turbopanda.MetaPanda object
import turbopanda as turb
mdf = turb.read("some_data.csv")
subset = mdf['some_interesting_pattern']
```

where this returns a `pandas.DataFrame` with the selected columns. We expand this functionality
to include a wide range of accepted arguments for **flexible column selection**.

#### Dishonest data typing

Pandas aims to keep itself
low-level as possible (and as close to NumPy) in order to maximise the performance aspect of
it's package, however the handling of *missing data* is only implemented in the **float64**
and **object** data types. This design decision leads to weird behaviour if users want to 
infer the data type in performing downstream analyses. There are certain operations that can
only be perform on floats or ints, and type checking manually requires a deep knowledge
of the package. In `pandas` version 0.2.5, the introduction of `select_dtypes` method 
provides some much needed support in selection via data type, but doesn't go far enough.

```python
# x is dtype = np.float64.... what?
import pandas as pd
x = pd.Series([0, 1, 0, 1, np.nan, 0, 0])
```

In `turbopanda`, metadata is calculated for the columns, including the *true data type*
that can be determined, ignoring missing data.

#### The ineffectiveness of multi-indexing

Multi-indexing in `pandas` is a huge pain. The attractiveness of providing intuitive data-grouping
is offsetted completely by the effective inability to access subsets of this data easily,
the disabling of many of the most powerful reshaping algorithms available in `pandas`, 
not to mention being riddled with bugs.

In `turbopanda` we distrust the system completely. Multi-indexes are always collapsed down
into single-indices as soon as possible 
but with longer string names (divided by `__`) whereby we 
use the *selection system*
and meta attributes to separate them accordingly.

#### The workflow of using pandas is difficult to track

One of the large benefits to `pandas` is the extensive of use of function chaining
that allows one to pipeline large numbers of changes to a `DataFrame`, facilitated
by the custom `apply`, `transform` and `aggregate` methods. However keeping up with 
these changes in a long workflow can be cumbersome. If there is an error part of the way
through (with expensive operations), this can significantly delay code development progress.

`turbopanda` in later versions will provide *automatic caching* to MetaPanda function chains
through the `compute` method in conjunction with `turb.Pipe` objects, such that if code breaks
several steps in, work can be resumed from the latest cached file, and at the end all intermediate
files are deleted.

---

The downstream effect of these issues is the poor handling *heterogenous*
 datasets that contain both numeric and categorical types. Further to this,
 selecting column subgroups is less than optimal and
 highly verbose, whilst giving you considerable control. 
 
 #### Disclaimer
 
 Finally it is worth mentioning that these issues are not necessarily the fault of the development
 team involved in the `pandas` project. The work is phenomenal and very good, and many of these
 issues can be argued to be outside the remit of `pandas` and its' objectives,
  hence why `turbopanda` is here.
  
**NEW**: From `pandas` version `1.0.1`, the introduction of *proper* typing should solve some of these problems,
but column selection is still disgusting.