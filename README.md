# turbopanda: Turbo-charging the Pandas library in an integrative, meta-orientated style

The aim of this library is to re-haul the `pandas` library package, which is extensively used for data munging, manipulation and visualization of large datasets. There are a number of areas that the Pandas library is lacklustre from a user standpoint - in particular it does a poor job of handling *heterogenous* datasets that contain both numeric and categorical types. Further to this, selecting column subgroups is less than optimal and highly verbose, whilst giving you considerable control. 

![Image not found](extras/readme.svg "Describing the interactions between turbopanda and pandas.")

This library creates two `pandas.DataFrames`, the raw data and a *metadata* frame.

**Current version: 0.1.0**

## Installation

`turbopanda` requires the following [dependencies](environment.yml):

* python (>=3.6)
* numpy (>=1.11.0)
* scipy (>=1.3)
* pandas (>=0.25.1)

The following packages are not required but significantly improve the usage of this package. If you are unfamiliar with the Jupyter project see [here](https://jupyter.org/):

* jupyter (1.0.0)

### From Cloning the GitHub Repository

Alternatively if you are cloning this [GitHub repository](https://github.com/gregparkes/turbopanda), use:

```bash
git clone https://github.com/gregparkes/turbopanda.git
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
>>> import turbopanda as turb
```

We recommend using `turb` as a shorthand to reduce the amount of writing out you have to do. All of the heavy lifting comes in the `turb.MetaPanda` object which acts as a hood over the top of a `pandas.DataFrame`:

```python
# where df is a pandas.DataFrame object.
>>> g = turb.MetaPanda(df)
>>> g
MetaPanda(DataSet(n=3000, p=9, mem=0.214MB))
```

Alternatively a `MetaPanda` object can be created using the in-built `read` function found in `turbopanda`:

```python
>>> g = turb.read("translation.csv")
>>> g
MetaPanda(DataSet(n=3000, p=9, mem=0.214MB))
```

here you see the `__repr__` of the object presents the dataset in terms of dimensions and memory usage which is incredibly useful to know at a cursory glance.

The raw pandas object can be accessed through the `df_` attribute:

```python
>>> g.df_.head()
```

| - | **Protein_IDs** | **Majority_protein_IDs** | **Protein_names** | **...** |
| 0 | Q96IC2;Q96IC2-2;H3BM72;H3BV93;H3BSC5 | Q96IC2;Q96IC2-2;H3BM72;H3BV93;H3BSC5 | Putative RNA exonuclease NEF-sp | ... |
| 1 | H0YGH4;P01023;H0YGH6;F8W7L3 | H0YGH4;P01023 | Alpha-2-macroglobulin | ... |
| 2 | A8K2U0;F5H2W3;H0YGG5;F5H2Z2;F5GXP1 | A8K2U0;F5H2W3;H0YGG5;F5H2Z2 | Alpha-2-macroglobulin-like protein 1 | ... |
| 3 | Q9NRG9;Q9NRG9-2;F8VZ44;H3BU82;F8VUB6 | Q9NRG9;Q9NRG9-2;F8VZ44;H3BU82 | Aladin | ... |
| 4 |  	Q86V21;Q86V21-2;E7EW25;F5H790;F8W8B5;Q86V21-3;... | Q86V21;Q86V21-2;E7EW25;F5H790 | Acetoacetyl-CoA synthetase | ... |

Whereas **metadata** can be accessed through the `meta_` which is automatically created upon instantiation:

```python
>>> g.meta_.head()
```

| - | **mytypes** | **is_unique** | **potential_id** | **potential_stacker** |
| Protein_IDs | object | True | True | True |
| Majority_protein_IDs | object | True | True | True |
| Protein_names | object | False | True | True |
| Gene_names | object | False | True | True |
| Intensity_G1_1 | float64 | False | False | False |

### Accessing column subsets

Unlike traditional pandas which is clunky to access column subsets of a DataFrame with ease, we allow flexible forms of input to override the `__getitem__` attribute, including:

* *regex*: string regular expressions
* type-casting: using a specific type
* Direct-column: the column name/`pandas.Index`
* meta-info: Using selections from the `meta_` attributes

Inputs examples could include:

```python
>>> g[object].head()
```

Returns all of the columns of type `object`. Or we could return all the columns whose name obeys some regular expression:

```python
>>> g["Intensity_[MG12S]*_1"].head()
```

Or we could return all of the columns that are unique identifiers, as determined by the `meta_` column, `is_unique`:

```python
>>> g["is_unique"].head()
```

Sometimes the columns returned may not be as expected for the user, so we provide a `view` and `view_not` functions which merely returns the `pd.Index` or list-like representation of the column names identified:

```python
>>> g.view(object)
["Protein_IDs", "Majority_protein_IDs", "Protein_names", "Gene_names"]
```

### Complex access by multi-views

`turbopanda` helps to facilitate more complex-like selections of columns, by default, by keeping the **union** of search terms:

$$
S=\bigcup_i t_i
$$

For example:

```python
>>> g.view(float, "Gene")
```

Returns all of the columns of type `float` and where the string name contains the word 'Gene'. 

### Transformations to columns

Often in `pandas`, operations are applied across the entire dataframe, which can be annoying if you just want to transform a **selection** of columns and return the changes inplace, or create a new column. `turbopanda` solves this with the `transform` function:

```python
>>> g.transform(float, lambda x: x**2)
```

This takes every column of type `float` and applies a square-function to it. `lambda` in this case accepts a `pandas.Series` object representing a given column, and expects the return type to be the same size as before.

Further details can be found by exploring the [examples](https://github.com/gregparkes/turbopanda/blob/master/examples/) section of this repository.

***

Ensure that any use of this material is appropriately referenced and in compliance with the [license](LICENSE.txt).