#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to the computing pipelines in MetaPanda."""

import warnings
from typing import List, Optional

from turbopanda._pipe import PipeMetaPandaType, is_pipe_structure
from turbopanda._deprecator import deprecated_param

__all__ = ('compute', 'compute_k' '_apply_pipe')


def _apply_pipe(self, pipe):
    # extract stored string if it is present.
    if isinstance(pipe, (list, tuple)):
        if len(pipe) == 0 and self._with_warnings:
            warnings.warn("pipe element empty, nothing to compute.", UserWarning)
            return
        # basic check of pipe
        if is_pipe_structure(pipe):
            for fn, args, kwargs in pipe:
                # check that MetaPanda has the function attribute
                if hasattr(self, fn):
                    # execute function with args and kwargs
                    getattr(self, fn)(*args, **kwargs)
    else:
        raise TypeError("pipe element must be of type [list, tuple], not {}".format(type(pipe)))


@deprecated_param("0.2.7", "pipe", reason="deprecation of Pipe object")
def compute(self,
            pipe: PipeMetaPandaType,
            inplace: bool = False,
            update_meta: bool = False) -> "MetaPanda":
    """Execute a pipeline on `df_`.

    Computes a pipeline to the MetaPanda object. If there are no parameters, it computes
    what is stored in the pipe_ attribute, if any.

    .. note:: the `meta_` attribute is **refreshed** after a call to `compute`, if `update_meta`

    Parameters
    -------
    pipe : str, Pipe, list of 3-tuple, (function name, *args, **kwargs)
        A set of instructions expecting function names in MetaPanda and parameters.
        If None, computes the stored pipe_.current pipeline.
        If str, computes the stored pipe_.<name> pipeline.
        If Pipe object, computes the elements in that class.
        See `turb.Pipe` for details on acceptable input for Pipes.
    inplace : bool, optional
        If True, applies the pipe inplace, else returns a copy. Default has now changed
        to return a copy. Only True if `source='df'`.
    update_meta : bool, optional
        If True, resets the meta after the pipeline completes.

    Returns
    -------
    self/copy

    See Also
    --------
    compute_k : Executes `k` pipelines on `df_`, in order.
    """
    if inplace:
        # computes inplace
        self._apply_pipe(pipe)
        # reset meta here
        if update_meta:
            self.update_meta()
        return self
    else:
        # full deepcopy, including dicts, lists, hidden etc.
        cpy = self.copy()
        # compute on cop
        cpy.compute(pipe, inplace=True, update_meta=update_meta)
        return cpy


@deprecated_param("0.2.7", "pipes", reason="deprecation of Pipe object")
def compute_k(self,
              pipes: List[PipeMetaPandaType],
              inplace: bool = False) -> "MetaPanda":
    """Execute `k` pipelines on `df_`, in order.

    Computes multiple pipelines to the MetaPanda object, including cached types such as `.current`

    .. note:: the `meta_` attribute is **refreshed** after a call to `compute_k`.

    Parameters
    --------
    pipes : list of (str, list of (list of 3-tuple, (function name, *args, **kwargs))
        A set of instructions expecting function names in MetaPanda and parameters.
        If empty, computes nothing.
    inplace : bool, optional
        If True, applies the pipes inplace, else returns a copy.

    Returns
    -------
    self/copy

    See Also
    --------
    compute : Executes a pipeline on `df_`.
    """
    # join and execute
    return self.compute(join(pipes), inplace=inplace)
