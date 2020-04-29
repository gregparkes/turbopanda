# turbopanda: The All-encompassing Data Science package
========================================================================================
Turbo-charging the Pandas library in an integrative, meta-orientated style.

![pypi](https://img.shields.io/pypi/v/turbopanda)
![last commit](https://img.shields.io/github/last-commit/gregparkes/turbopanda)
![repo size](https://img.shields.io/github/repo-size/gregparkes/turbopanda)
![commit activity](https://img.shields.io/github/commit-activity/m/gregparkes/turbopanda)
![License](https://img.shields.io/badge/LICENSE-GPLv3-blue)

**Current version: 0.2.6**

The aim of this library is extend the functionality of a number of Python packages,
including the `pandas` library, to integrate cohesively together a unified approach to data
modelling, including machine learning.

## The basic idea

![Image not found](extras/readme.svg "Describing the interactions between turbopanda and pandas.")

The main purpose is to build a layer on top of pandas which regulates the main data and also associates
some meta information to the columns which *remembers* interactions the user has with it,
specifically to do with grouping data columns by the name or some other defining feature.

## Motivation

There are a number of areas that the Pandas library is
  lacklustre from a user standpoint - we'll cover a few of these in more detail and then
  explain TurboPandas' response to these particular issues.
  
For details, read the [ISSUES](ISSUES.md) markdown file found in the repository.

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
g.head(2)
```

| - | **Protein_IDs** | **Majority_protein_IDs** | **Protein_names** | **...** |
| --- | --------------------- | -------------------------- | ------------------- | ---------------- |
| 0 | Q96IC2;Q96IC2-2;H3BM72;H3BV93;H3BSC5 | Q96IC2;Q96IC2-2;H3BM72;H3BV93;H3BSC5 | Putative RNA exonuclease NEF-sp | ... |
| 1 | H0YGH4;P01023;H0YGH6;F8W7L3 | H0YGH4;P01023 | Alpha-2-macroglobulin | ... |

Whereas **metadata** can be accessed through the `meta_` which is automatically created upon instantiation:

```python
g.meta_.head(2)
```

| - | **true_type** | **is_unique** | **potential_id**
| --- | -------- | -------- |---------- |
| Protein_IDs | object | True | True |
| Majority_protein_IDs | object | True | True |

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

## Installation

`turbopanda` requires a number of dependencies in order to function well, you can find these
 in the [dependencies](environment.yml) file. The majority of the requirements can be met
 by using the [Anaconda][6] distribution.

We recommend you use [Jupyter][7] to work with `turbopanda` given the benefits of
quick development of code, with fast visualisation.

### From Cloning the GitHub Repository

Alternatively if you are cloning this [GitHub repository][5], use:

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

## Changelog

Details as to specific and on-going changes can be found either in the [Changelog](CHANGELOG.rst) file or
in the [GitHub repository][5].

## Acknowledgments

We would like to acknowledge the following sources for inspiration for much of this work:

- `pandas` [dev team](https://github.com/pandas-dev/pandas): Forming a solid backbone package to build upon
- `pingouin` [python library][3]: For inspiration and code regarding correlation analysis
- `pyitlib` [library](https://github.com/pafoster/pyitlib): For inspiration on mutual information and entropy
- `matplotlib` [library][4]
- `statsmodels` and `patsy` libraries for inspiration on how to formulate design matrices.
- [PythonCentral](https://www.pythoncentral.io/validate-python-function-parameters-and-return-types-with-decorators/) tutorials for code validation
- Wikipedia for many topics

## References

[1]: "Pernet CR, Wilcox R, Rousselet GA. Robust Correlation Analyses:
       False Positive and Power Validation Using a New Open Source Matlab
       Toolbox. Frontiers in Psychology. 2012;3:606.
       doi:10.3389/fpsyg.2012.00606."
       
[2]: "Wilcox, R.R., 1994. The percentage bend correlation coefficient.
       Psychometrika 59, 601–616. https://doi.org/10.1007/BF02294395"
       
[3]: <https://github.com/raphaelvallat/pingouin/blob/master/pingouin/correlation.py> "Pingouin's correlation"
[4]: <https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html> "Matplotlib heatmap"
[5]: <https://github.com/gregparkes/turbopanda> "Github repo"
[6]: <https://www.anaconda.com> "Anaconda distribution"
[7]: <https://jupyter.org/> "Jupyter"
***

Ensure that any use of this material is appropriately referenced 
and in compliance with the [license](LICENSE.txt).
