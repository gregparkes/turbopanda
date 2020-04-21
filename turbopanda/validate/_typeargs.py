#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides decorators to check the argument and return types of functions."""

import functools

from turbopanda.utils import ordinal
from ._typeerrors import InvalidArgumentNumberError, InvalidReturnType, ArgumentValidationError


def accepts(*accepted_arg_types):
    """A decorator to validate the parameter types of a given function.

    It is passed a tuple of types. eg. (<type 'tuple'>, <type 'int'>)

    Note: It doesn't do a deep check, for example checking through a
          tuple of types. The argument passed must only be types.

    References
    ----------
    Taken from https://www.pythoncentral.io/validate-python-function-parameters-and-return-types-with-decorators/
    """
    def _accept_decorator(validate_function):
        # Check if the number of arguments to the validator
        # function is the same as the arguments provided
        # to the actual function to validate. We don't need
        # to check if the function to validate has the right
        # amount of arguments, as Python will do this
        # automatically (also with a TypeError).
        @functools.wraps(validate_function)
        def _decorator_wrapper(*function_args, **function_args_dict):
            if len(accepted_arg_types) is not len(accepted_arg_types):
                raise InvalidArgumentNumberError(validate_function.__name__)

            # We're using enumerate to get the index, so we can pass the
            # argument number with the incorrect type to ArgumentValidationError.
            for arg_num, (actual_arg, accepted_arg_type) in enumerate(zip(function_args, accepted_arg_types)):
                if not type(actual_arg) is accepted_arg_type:
                    ord_num = ordinal(arg_num + 1)
                    raise ArgumentValidationError(ord_num,
                                                  validate_function.__name__,
                                                  accepted_arg_type)

            return validate_function(*function_args)

        return _decorator_wrapper
    return _accept_decorator


def returns(*accepted_return_type_tuple):
    """Validates the return type.

    Since there's only ever one
    return type, this makes life simpler. Along with the
    accepts() decorator, this also only does a check for
    the top argument. For example you couldn't check
    (<type 'tuple'>, <type 'int'>, <type 'str'>).
    In that case you could only check if it was a tuple.

    References
    ----------
    Taken from https://www.pythoncentral.io/validate-python-function-parameters-and-return-types-with-decorators/
    """
    def _return_decorator(validate_function):
        # No return type has been specified.
        if len(accepted_return_type_tuple) == 0:
            raise TypeError('You must specify a return type.')

        @functools.wraps(validate_function)
        def _decorator_wrapper(*function_args):
            # More than one return type has been specified.
            if len(accepted_return_type_tuple) > 1:
                raise TypeError('You must specify one return type.')

            # Since the decorator receives a tuple of arguments
            # and the is only ever one object returned, we'll just
            # grab the first parameter.
            accepted_return_type = accepted_return_type_tuple[0]

            # We'll execute the function, and
            # take a look at the return type.
            return_value = validate_function(*function_args)
            return_value_type = type(return_value)

            if return_value_type is not accepted_return_type:
                raise InvalidReturnType(return_value_type,
                                        validate_function.__name__)

            return return_value

        return _decorator_wrapper
    return _return_decorator
