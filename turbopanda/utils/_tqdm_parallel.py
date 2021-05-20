#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an extended class to use tqdm with joblib's Parallel object."""

from turbopanda._dependency import requires


@requires("joblib", "tqdm")
def TqdmParallel(use_tqdm=True, total=None):
    """A wrapper function that contains the ProgressParallel class for using joblib with tqdm progressbars.

    Inspired by StackOverflow:
        https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib
    """

    from joblib import Parallel
    from tqdm.auto import tqdm

    class ProgressParallel(Parallel):
        """Overrides the joblib parallel object to allow for compatibility with tqdm."""

        def __init__(self, *args, **kwargs):
            self._use_tqdm = use_tqdm
            self._total = total
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
                return Parallel.__call__(self, *args, **kwargs)

        def print_progress(self):
            """Overrided print progress string for tqdm"""
            if self._total is None:
                self._pbar.total = self.n_dispatched_tasks
            self._pbar.n = self.n_completed_tasks
            self._pbar.refresh()

    return ProgressParallel
