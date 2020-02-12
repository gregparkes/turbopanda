#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to caching meta information in Metapanda."""

import warnings
import pandas as pd
from ._types import SelectorType
from turbopanda._pipe import PipeMetaPandaType, Pipe
from turbopanda.utils import t_numpy, dictmap

__all__ = ('cache', 'cache_k', 'cache_pipe')


def cache(self, name: str, *selector: SelectorType) -> "MetaPanda":
    """Add a cache element to `selectors_`.

    Saves a 'selector' to use at a later date. This can be useful if you
    wish to keep track of changes, or if you want to quickly reference a selector
    using a name rather than a group of selections.

    Parameters
    -------
    name : str
        A name to reference the selector with.
    selector : str or tuple args
        Contains either types, meta column names, column names or regex-compliant strings

    Warnings
    --------
    UserWarning
        Raised if `name` already exists in `selectors_`, overrides by default.

    Returns
    -------
    self

    See Also
    --------
    cache_k : Adds k cache elements to `selectors_.
    cache_pipe : Adds a pipe element to `pipe_`.
    """
    if name in self._select and self._with_warnings:
        warnings.warn("cache '{}' already exists in .cache, overriding".format(name), UserWarning)
    # convert selector over to list to make it mutable
    selector = list(selector)
    # encode to string
    enc_map = {
        **{object: "object", pd.CategoricalDtype: "category"},
        **dictmap(t_numpy(), lambda n: n.__name__)
    }
    # update to encode the selector as a string ALWAYS.
    selector = [enc_map[s] if s in enc_map else s for s in selector]
    # store to select
    self._select[name] = selector
    return self


def cache_k(self, **caches: SelectorType) -> "MetaPanda":
    """Add k cache elements to `selectors_`.

    Saves a group of 'selectors' to use at a later date. This can be useful
    if you wish to keep track of changes, or if you want to quickly reference a selector
    using a name rather than a group of selections.

    Parameters
    --------
    caches : dict (k, w)
        keyword: unique reference of the selector
        value: selector: str, tuple args
             Contains either types, meta column names, column names or regex-compliant

    Warnings
    --------
    UserWarning
        Raised if one of keys already exists in `selectors_`, overrides by default.

    Returns
    -------
    self

    See Also
    --------
    cache : Adds a cache element to `selectors_`.
    cache_pipe : Adds a pipe element to `pipe_`.
    """
    for name, selector in caches.items():
        if isinstance(selector, (tuple, list)) and len(selector) > 1:
            self.cache(name, *selector)
        else:
            self.cache(name, selector)
    return self


def cache_pipe(self, name: str, pipeline: PipeMetaPandaType) -> "MetaPanda":
    """Add a pipe element to `pipe_`.

    Saves a pipeline to use at a later date. Calls to `compute` can reference the name
    of the pipeline.

    Parameters
    ----------
    name : str
        A name to reference the pipeline with.
    pipeline : Pipe, list, tuple
        list of 3-tuple, (function name, *args, **kwargs), multiple pipes, optional
        A set of instructions expecting function names in MetaPanda and parameters.
        If None, computes the stored `pipe_` attribute.

    Warnings
    --------
    UserWarning
        Raised if `name` already exists in `pipe_`, overrides by default.

    Returns
    -------
    self
    """
    if name in self.pipe_.keys() and self._with_warnings:
        warnings.warn("pipe '{}' already exists in .pipe, overriding".format(name), UserWarning)
    if isinstance(pipeline, Pipe):
        # attempt to create a pipe from raw.
        self.pipe_[name] = pipeline.p
    else:
        self.pipe_[name] = pipeline
    return self
