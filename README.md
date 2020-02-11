# turbopanda: Turbo-charging the Pandas library in an integrative, meta-orientated style

**Current version: 0.2.2**

The aim of this library is extend the functionality of the `pandas` library package,
 which is extensively used for data munging,
 manipulation and visualization of large datasets.

## Motivation

![Image not found](extras/readme.svg "Describing the interactions between turbopanda and pandas.")

There are a number of areas that the Pandas library is
  lacklustre from a user standpoint - we'll cover a few of these in more detail and then
  explain TurboPandas' response to these particular issues.
      
### Known Issues

Here we will cover some personal and more established gripes with `pandas`:

#### The verbose **selection of column-data**

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
subset = mdf['something_interesting_pattern']
```

where this returns a `pandas.DataFrame` with the selected columns. We expand this functionality
to include a wide range of accepted arguments for **flexible column selection**.

#### Dishonest **data typing**

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

#### The ineffectiveness of **multi-indexing**

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
 
 Finally it is worth mentioning that these issues are not necessarily the fault of the development
 team involved in the `pandas` project. The work is phenomenal and very good, and many of these
 issues can be argued to be outside the remit of `pandas` and its' objectives,
  hence why `turbopanda` is here.

## How to use: The Basics

You will need to import the package as:

```python
import turbopanda as turb
```

We recommend using `turb` as a shorthand to reduce the amount of writing
 out you have to do. All of the heavy lifting comes in the `turb.MetaPanda` 
 object which acts as a hood over the top of a `pandas.DataFrame`:

```python
# where df is a pandas.DataFrame object.
import turbopanda as turb
g = turb.MetaPanda(df)
```

Alternatively a `MetaPanda` object can be created using the in-built 
`read` function found in `turbopanda`:

```python
import turbopanda as turb
g = turb.read("translation.csv")
```

here you see the `__repr__` of the object presents the dataset in terms
 of dimensions and memory usage which is incredibly useful to know at a
  cursory glance.

The raw pandas object can be accessed through the `df_` attribute:

```python
g.head()
```

| - | **Protein_IDs** | **Majority_protein_IDs** | **Protein_names** | **...** |
| --- | --------------------- | -------------------------- | ------------------- | ---------------- |
| 0 | Q96IC2;Q96IC2-2;H3BM72;H3BV93;H3BSC5 | Q96IC2;Q96IC2-2;H3BM72;H3BV93;H3BSC5 | Putative RNA exonuclease NEF-sp | ... |
| 1 | H0YGH4;P01023;H0YGH6;F8W7L3 | H0YGH4;P01023 | Alpha-2-macroglobulin | ... |
| 2 | A8K2U0;F5H2W3;H0YGG5;F5H2Z2;F5GXP1 | A8K2U0;F5H2W3;H0YGG5;F5H2Z2 | Alpha-2-macroglobulin-like protein 1 | ... |
| 3 | Q9NRG9;Q9NRG9-2;F8VZ44;H3BU82;F8VUB6 | Q9NRG9;Q9NRG9-2;F8VZ44;H3BU82 | Aladin | ... |
| 4 | Q86V21;Q86V21-2;E7EW25;F5H790;F8W8B5;Q86V21-3;... | Q86V21;Q86V21-2;E7EW25;F5H790 | Acetoacetyl-CoA synthetase | ... |

Whereas **metadata** can be accessed through the `meta_` which is automatically created upon instantiation:

```python
g.meta_.head()
```

| - | **mytypes** | **is_unique** | **potential_id** | **potential_stacker** |
| --- | -------- | -------- |---------- | --------- |
| Protein_IDs | object | True | True | True |
| Majority_protein_IDs | object | True | True | True |
| Protein_names | object | False | True | True |
| Gene_names | object | False | True | True |
| Intensity_G1_1 | float64 | False | False | False |

### Accessing column subsets

Unlike traditional pandas which is clunky to access column subsets of
 a DataFrame with ease, we allow flexible forms of input to override
  the `__getitem__` attribute, including:

* *regex*: string regular expressions
* type-casting: using a specific type
* Direct-column: the column name/`pandas.Index`
* meta-info: Using selections from the `meta_` attributes

Inputs examples could include:

```python
g[object].head()
```

Returns all of the columns of type `object`. Or we could return all 
the columns whose name obeys some regular expression:

```python
g["Intensity_[MG12S]*_1"].head()
```

Or we could return all of the columns that are unique identifiers,
 as determined by the `meta_` column, `is_unique`:

```python
g["is_unique"].head()
```

Sometimes the columns returned may not be as expected for the user,
 so we provide a `view` and `view_not` functions which merely returns 
 the `pd.Index` or list-like representation of the column names identified:

```python
g.view(object)
```

### Complex access by multi-views

`turbopanda` helps to facilitate more complex-like selections of 
columns, by default, by keeping the **union** of search terms, for example:

```python
g.view(float, "Gene")
```

Returns all of the columns of type `float` and where the string
 name contains the word 'Gene'. 

### Transformations to columns

Often in `pandas`, operations are applied across the entire 
dataframe, which can be annoying if you just want to transform 
a **selection** of columns and return the changes inplace, or
 create a new column. `turbopanda` solves this with the `transform` function:

```python
g.transform(lambda x: x**2, float)
```

This takes every column of type `float` and applies a square-function
 to it. `lambda` in this case accepts a `pandas.Series` object
  representing a given column, and expects the return type to be
   the same size as before.

Further details can be found by exploring the 
[examples](https://github.com/gregparkes/turbopanda/blob/master/examples/) 
section of this repository.

## Installation

`turbopanda` requires the following [dependencies](environment.yml):

* `python`>=3.6
* `numpy`>=1.11.0
* `scipy`>=1.3
* `pandas`>=0.25
* `matplotlib`>=3.1.1
* `scikit-learn`>=0.21

The following packages are needed to read or write `.xls` or `.hdf`/`.h5` files:

* `xlrd`
* `pytables`

The following packages are not required but significantly improve 
the usage of this package. If you are unfamiliar with the Jupyter 
project see [here](https://jupyter.org/):

* `jupyter` (1.0.0)

### From Cloning the GitHub Repository

Alternatively if you are cloning this [GitHub repository](https://github.com/gregparkes/turbopanda), use:

```bash
git clone https://github.com/gregparkes/turbopanda.git
conda env create -f environment.yml
# or source activate turbopanda...
conda activate turbopanda
```

Now within the `turbopanda` environment run your Jupyter notebook:

```bash
jupyter notebook
```

***

Ensure that any use of this material is appropriately referenced 
and in compliance with the [license](LICENSE.txt).
