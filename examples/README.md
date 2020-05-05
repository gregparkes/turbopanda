# Examples with `turbopanda`

This folder contains a large array of examples to use and look at in order to understand
the various aspects of this project.

## Boxplots

This example explores how to draw powerful boxplots depending on the format of your dataset,
whether a numpy.array or pandas.DataFrame.

## Correlation

Performing default correlations using `pandas` is simple, but niche edge cases such as 
automatic handling depending on data type of the column, changing the type of correlation, and
handling different data formats become challenging. Look at how to do correlation in `turbopanda`
instead.

## Dimension Reduction

`turbopanda` provides a clean way to perform PCA analysis with instant feedback in the form
of data and plots.

## Gridplot

`turbopanda` provides a useful utility function that decides how to generate gridded matplotlib
plots. Never think about how to orient your plots again.

## KDEs and Histograms

Wouldn't it be nice if there was a function to draw a histogram AND automatically fit the best
known statistical distribution to it? Well we provide that and more. Our flexible `histogram` function
automatically bins and fits a KDE, or none.

## Merge

Joining a number of weird combinations of datasets can be a nightmare, but with our `merge` function
not only does it handle combinations of `numpy`, `pandas` and `MetaPanda` but it automatically finds
the best two columns to join for each dataset pairing, meaning you don't need to find this
yourself.

## MetaPanda Basics and Selection

Learning how to use a MetaPanda may significantly improve your workflow with tabular data.

## Model Fitting

We integrate many of the features developed into automatic machine learning models. Of these,
fitting a model has never been made simpler.

## Scatterplots

Drawing points to represent information is a fundamental to data science, but can be quite cumbersome
using matplotlib. Our `scatter` function helps to handle a number of use-cases, particularly when you are
drawing many points (>100k).

## String analysis

At the heart of turbopanda is a desire to type less for users, and as such some crafty text analysis
comes into play in terms of handling and referencing your data. Learn how this work in our string-analysis
example set.

## Vectorizer

Coding should be easy, and wouldn't it be nice if that function I want to call would accept a vector of
possible options, and it returns a list of the results? Well that's possible now with our `vectorize`
decorator which you can attach to any function, transforming it's input and output potential.
