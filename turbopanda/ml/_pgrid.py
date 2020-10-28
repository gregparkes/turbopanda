#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles custom parameter grid development."""
import numpy as np
from typing import List, Dict, Union, Optional

from turbopanda.ml._default import model_types, param_types
from turbopanda.ml._package import find_sklearn_model
from turbopanda.utils import broadsort, listify, join


def _get_default_param_name(model: str):
    """Given model m, get default parameter name."""
    _mt = model_types()
    if model in _mt.index:
        return _mt.loc[model, "Primary Parameter"]
    else:
        raise ValueError("model '{}' not found in selection".format(model))


def _get_default_param_value(param_name):
    _pt = param_types()
    return _pt.loc[param_name, "Default"]


def get_slim_continuous_param(param_name: str):
    """Given param name, gives the basic range of the parameter, must be continuous"""
    _pt = param_types()
    if param_name not in _pt.index:
        raise ValueError(
            "param name '{}' not found in available parameters".format(param_name)
        )
    if _pt.loc[param_name, "DataType"] == "float":
        return (
            _pt.loc[param_name, "Range Min"],
            _pt.loc[param_name, "Default"],
            _pt.loc[param_name, "Range Max"],
        )
    else:
        raise ValueError("parameter '{}' is not of type float")


def get_bounded_continuous_param(param_name: str):
    """Given param name, gives the basic range of the parameter, with bounds, must be continuous"""
    _pt = param_types()
    if param_name not in _pt.index:
        raise ValueError(
            "param name '{}' not found in available parameters".format(param_name)
        )
    if _pt.loc[param_name, "DataType"] == "float":
        return (
            _pt.loc[param_name, "Bound Min"],
            _pt.loc[param_name, "Range Min"],
            _pt.loc[param_name, "Default"],
            _pt.loc[param_name, "Range Max"],
            _pt.loc[param_name, "Bound Max"],
        )
    else:
        raise ValueError("parameter '{}' is not of type float")


def _get_default_params(
    model: str, param: Optional[str] = None, return_list: bool = True
):
    """Given model m, gets the list of primary parameter values."""
    _pt = param_types()
    _param = _get_default_param_name(model) if param is None else param
    # is parameter available?
    if _param not in _pt.index:
        raise ValueError(
            "parameter '{}' not found as valid option: {}".format(
                _param, _pt.index.tolist()
            )
        )

    if _pt.loc[_param, "Scale"] == "log":
        x = np.logspace(
            np.log10(_pt.loc[_param, "Range Min"]),
            np.log10(_pt.loc[_param, "Range Max"]),
            int(_pt.loc[_param, "Suggested N"]),
        )
    elif _pt.loc[_param, "Scale"] == "normal":
        x = np.linspace(
            _pt.loc[_param, "Range Min"],
            _pt.loc[_param, "Range Max"],
            int(_pt.loc[_param, "Suggested N"]),
        )
    else:
        return _pt.loc[_param, "Options"].split(", ")

    if _pt.loc[_param, "DataType"] == "int":
        x = x.astype(np.int)

    if return_list:
        x = x.tolist()
    # don't return model, just get the data.
    return x


def optimize_grid_for_model(params, k=3):
    """For a list of parameters, get the initial guess and boundaries for each parameter"""
    inits = []
    bounds = []
    init_k = []
    for p in params:
        values = get_bounded_continuous_param(p)
        inits.append(values[2])
        bounds.append((values[0], values[-1]))
        init_k.append(np.random.uniform(values[1], values[-2], size=k))

    # join together init options
    initials = join([inits], tuple(zip(*init_k)))

    return initials, bounds


def make_optimize_grid(models):
    """Returns a set of tuple parameters for each model type."""
    if isinstance(models, (list, tuple)):
        return tuple(map(lambda x: [_get_default_param_name(x)], models))
    elif isinstance(models, dict):
        return tuple(models.values())
    else:
        raise TypeError(
            "model input type '{}' not recognized, must be [list, tuple, dict]".format(
                type(models)
            )
        )


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
        _p = [
            {
                header: [find_sklearn_model(model)[0]],
                header
                + "__"
                + _get_default_param_name(model): broadsort(_get_default_params(model)),
            }
            for model in models
        ]
        return _p
    elif isinstance(models, dict):

        def _handle_single_model(name, _val):
            if isinstance(_val, (list, tuple)):
                # if the values are list/tuple, they are parameter names, use defaults
                _p = {
                    header + "__" + _v: broadsort(_get_default_params(name, _v))
                    for _v in _val
                }
                _p[header] = listify(find_sklearn_model(name)[0])
                return _p
            elif isinstance(_val, dict):
                _p = {header + "__" + k: broadsort(list(v)) for k, v in _val.items()}
                _p[header] = listify(find_sklearn_model(name)[0])
                return _p

        arg = [
            _handle_single_model(model_name, val) for model_name, val in models.items()
        ]
        if len(arg) == 1:
            return arg[0]
        else:
            return arg
    else:
        raise TypeError(
            "input type for 'models' {} is not recognized; choose from [list, tuple, dict]"
        ).format(type(models))
