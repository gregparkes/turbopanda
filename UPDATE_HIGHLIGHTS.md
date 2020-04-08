# Update Highlights

Here we cover some of the more exotic and latest 
changes to the `turbopanda` framework.

### v0.2.1: Merging support

With turbopanda merging has never been easier. Now if there are any columns between two datasets $X$ and $Y$ 
that share values, you can use `turb.merge` to join together $k$ datasets into one along their columns:

```python
import turbopanda as turb
df1 = pd.DataFrame([...]) # size (n, p)
df2 = pd.DataFrame([...]) # size (m, q)
result = turb.merge([df1, df2], how='inner', name="new_df")
```

Unlike the pandas version, turbopanda does not require the user to specify which column(s) the dataframes
decide to join on, but this information can be conveyed to the user using the `verbose` keyword if they
wish to know which two columns were used, as well as the number of columns/rows potentially dropped in the
process.

### v0.2.3: Statistical Support

`turbopanda` also supports a limited number of useful statistical functions, including 
automatic fitting of continuous distributions to sets of data, *correlation* between 
heterogeneous data columns, *mutual information* and entropy calculations, as well as
model checking operations such as `cook_distance` and variance inflationary factors.

### v0.2.4: Distribution fitting

There are very few packages in the Python ecosystem that consider both *continuous* and *discrete* distribution
fitting in a seamless fashion with other well known packages such as `pandas`. We attempt to rectify this
to give users powerful methods to statistically analyse the data they present to it.

### v0.2.5: Intelligent plotting

We include improved versions of scatterplots, boxplots and KDE plots (2D) in the latest patch
with automatic intelligent plotting. For example, a common problem with scatter plots with large `n`
is to plot too many points over each other, slowing memory. `turb.plot.scatter` remedies this
by density checking, and altering the size of points if clusters begin to emerge, saving run time and
space.

Further to this, DataFrame-to-DataFrame pipes are now beginning to be integrated.

### AOB

Further details can be found by exploring the 
[examples](https://github.com/gregparkes/turbopanda/blob/master/examples/) 
section of this repository.