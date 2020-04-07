#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles custom parameter grid development."""
import numpy as np
from typing import List, Dict, Union, Optional

from turbopanda.ml._default import model_types, param_types
from turbopanda.ml._package import find_sklearn_model
from turbopanda.utils import broadsort, listify


def _get_default_param_name(model: str):
    """Given model m, get default parameter name."""
    _mt = model_types()
    if model in _mt.index:
        return _mt.loc[model, "Primary Parameter"]
    else:
        raise ValueError("model '{}' not found in selection".format(model))


def _get_default_params(model: str, param: Optional[str] = None, return_list: bool = True):
    """Given model m, gets the list of primary parameter values."""
    _pt = param_types()
    _param = _get_default_param_name(model) if param is None else param
    # is parameter available?
    if _param not in _pt.index:
        raise ValueError("parameter '{}' not found as valid option: {}".format(_param, _pt.index.tolist()))

    if _pt.loc[_param, "Scale"] == 'log':
        x = np.logspace(np.log10(_pt.loc[_param, "Range Min"]),
                        np.log10(_pt.loc[_param, "Range Max"]),
                        int(_pt.loc[_param, "Suggested N"]))
    elif _pt.loc[_param, 'Scale'] == "normal":
        x = np.linspace(_pt.loc[_param, 'Range Min'],
                        _pt.loc[_param, 'Range Max'],
                        int(_pt.loc[_param, 'Suggested N']))
    else:
        return _pt.loc[_param, 'Options'].split(", ")

    if _pt.loc[_param, "DataType"] == "int":
        x = x.astype(np.int)

    if return_list:
        x = x.tolist()
    # don't return model, just get the data.
    return x


def make_parameter_grid(models, header: str = "model") -> Union[Dict, List[Dict]]:
    """Generates a sklearn-compatible parameter grid to feed into GridSearchCV.

    Parameters
    ----------
    models : list/tuple/dict
        models can be one of:
            tuple: list of model names, uses default parameters
            dict: key (model name), value tuple/list (parameter names) /
                dict: key (parameter name), value (list of values)
        Accepts shorthand versions of model names
    header : str, default='model'
        The name of the start pipeline

    Returns
    -------
    grid : list of dict
        The parameter grid.
    """
    if isinstance(models, (list, tuple)):
        _p = [{header: [find_sklearn_model(model)[0]],
               header + "__" + _get_default_param_name(model): broadsort(_get_default_params(model))}
              for model in models]
        return _p
    elif isinstance(models, dict):
        def _handle_single_model(name, _val):
            if isinstance(_val, (list, tuple)):
                # if the values are list/tuple, they are parameter names, use defaults
                _p = {header + "__" + _v: broadsort(_get_default_params(name, _v)) for _v in _val}
                _p[header] = listify(find_sklearn_model(name)[0])
                return _p
            elif isinstance(_val, dict):
                _p = {header + "__" + k: broadsort(list(v)) for k, v in _val.items()}
                _p[header] = listify(find_sklearn_model(name)[0])
                return _p

        arg = [_handle_single_model(model_name, val) for model_name, val in models.items()]
        if len(arg) == 1:
            return arg[0]
        else:
            return arg
