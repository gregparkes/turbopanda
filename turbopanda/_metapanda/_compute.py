#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Organization of the compute() aspects of MetaPanda."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional, Tuple
import sys
import warnings
sys.path.append("../")
from .pipe import PipeMetaPandaType, is_pipe_structure, Pipe
from .utils import join
from ._overloaded import copy
from ._metadata import _reset_meta


def _apply_pipe(self, pipe):
    # checks
    if isinstance(pipe, str):
        # see if name is cached away.
        if pipe in self.pipe_.keys():
            pipe = self.pipe_[pipe]
        else:
            raise ValueError("pipe name '{}' not found in .pipe attribute".format(pipe))
    elif isinstance(pipe, Pipe):
        pipe = pipe.p
    if isinstance(pipe, (list, tuple)):
        if len(pipe) == 0 and self._with_warnings:
            warnings.warn("pipe_ element empty, nothing to compute.", UserWarning)
            return
        # basic check of pipe
        if is_pipe_structure(pipe):
            for fn, args, kwargs in pipe:
                # check that MetaPanda has the function attribute
                if hasattr(self, fn):
                    # execute function with args and kwargs
                    getattr(self, fn)(*args, **kwargs)


def compute(self,
            pipe: Optional[PipeMetaPandaType] = None,
            inplace: bool = False,
            update_meta: bool = False) -> "MetaPanda":
    """Execute a pipeline on `df_`.

    Computes a pipeline to the MetaPanda object. If there are no parameters, it computes
    what is stored in the pipe_ attribute, if any.

    .. note:: the `meta_` attribute is **refreshed** after a call to `compute`, if `update_meta`

    .. warning:: Not affected by `mode_` attribute.

    Parameters
    -------
    pipe : str, Pipe, list of 3-tuple, (function name, *args, **kwargs), optional
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
    if pipe is None:
        # use self.pipe_
        pipe = self.pipe_["current"]
        self.pipe_["current"] = []
    if inplace:
        # computes inplace
        self.mode_ = "instant"
        self._apply_pipe(pipe)
        # reset meta here
        if update_meta:
            self._reset_meta()
        return self
    else:
        # full deepcopy, including dicts, lists, hidden etc.
        cpy = self.copy()
        # compute on cop
        cpy.compute(pipe, inplace=True, update_meta=update_meta)
        return cpy


def compute_k(self,
              pipes: Tuple[PipeMetaPandaType, ...],
              inplace: bool = False) -> "MetaPanda":
    """Execute `k` pipelines on `df_`, in order.

    Computes multiple pipelines to the MetaPanda object, including cached custypes.py such as `.current`

    .. note:: the `meta_` attribute is **refreshed** after a call to `compute_k`.

    .. warning:: Not affected by `mode_` attribute.

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