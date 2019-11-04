# turbopanda: Turbo-charging the Pandas library in an integrative, meta-orientated style

The aim of this library is to re-haul the `pandas` library package, which is extensively used for data munging, manipulation and visualization of large datasets. There are a number of areas that the Pandas library is lacklustre from a user standpoint - in particular it does a poor job of handling *heterogenous* datasets that contain both numeric and categorical types. Further to this, selecting column subgroups is less than optimal and highly verbose, whilst giving you considerable control. 

This library aims to let you handle multiple DataFrames with ease, perform automatic checking, typing, cleaning and intuitive selection by adding a *meta-layer* over a `pandas.DataFrame` with useful information to the user.

**Current version: 0.0.3**

## Installation

`turbopanda` requires the following [dependencies](environment.yml):

* python (>=3.6)
* numpy (>=1.11.0)
* scipy (>=1.3)
* pandas (>=0.25.1)
* matplotlib (>=3.1.1)
* scikit-learn (>=0.21)

The following packages are not required but significantly improve the usage of this package. If you are unfamiliar with the Jupyter project see [here](https://jupyter.org/):

* jupyter (1.0.0)

### From Cloning the GitHub Repository

Alternatively if you are cloning this [GitHub repository](https://github.com/gregparkes/turbopanda), use:

```bash
git clone https://github.com/gregparkes/turbopanda
conda env create -f environment.yml
conda activate turbopanda
```

Now within the `turbopanda` environment run your Jupyter notebook:

```bash
jupyter notebook
```

## How to use: The Basics

You will need to import the package as:

```python
>>> import turbopanda as trb
```

We recommend using `trb` as a shorthand to reduce the amount of writing out you have to do. All of the heavy lifting comes in the `trb.MetaPanda` object which acts as a hood over the top of a `pandas.DataFrame` giving you two key attributes.

```python

>>> turb = trb.MetaPanda(df)
>>> turb
MetaPanda(DataSet(n=3000, p=9, mem=0.214MB))
```

here you see the `__repr__` of the object presents the dataset in terms of dimensions and memory usage which is incredibly useful to know at a cursory glance.

The raw pandas object can be accessed through the `df_` attribute:

```python
>>> turb.df_
[Dataframe output]
```

### Some additional modifications...

`MetaPanda` does not accept `pd.MultiIndex` for the columns due to the massive increase in complexity that this entails, so multi-columns are concatenated together using `__` string join, further to this, column names will be cleaned to remove spaces, tabs etc.

Further to this, your dataframe will be properly categorized using the most efficient datatype to save memory space.

Meta information is pre-calculated so you don't need to do the heavy-lifting, this can be incredibly useful for users to know which columns are likely to be used as IDs (and are unique), data columns, whether they are normally distributed, and so on. All of this is encapsulated within the `meta_` attribute:

```python
>>> turb.meta_
[Dataframe output]
```

### Accessing column subsets using a vast array of methods

Unlike traditional pandas which is clunky to access column subsets of a DataFrame with ease, we allow multiple forms of input to override the `__getitem__` attribute, including:

* `regex`: regular expressions
* type-casting: using a specific type
* Direct-column: the column name
* meta-info: Using selections from the `meta_` attributes

Inputs could include:

```python
>>> turb[object].head()
```

Returns all of the columns of type `object`. Or we could return all the columns whose name obeys some regular expression:

```python
g["Intensity_[MG12S]*_1"].head()
```

Or we could return all of the columns that follow a strict normal distribution, as determined by `scipy.stats.kstest`:

```python
g["is_norm"].head()
```

Future plans include providing a way to combine these methods to provide a slick way of quickly subsetting your columns, and of course coupling this with `pandas` native `.loc`, `.iloc` and `query` methods for further subdivision.

Further to this, coupling these metadataframes with scikit-learn to provide parallel processing to machine learning tasks, while retaining the appropriate labels.

***

Ensure that any use of this material is appropriately referenced and in compliance with the [license](LICENSE.txt).