#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Basic main file for debugging purposes."""
import numpy as np
import pandas as pd
import turbopanda as turb
import simpleaudio as sa
import matplotlib.pyplot as plt
import itertools as it


@turb.dev.bleep(note='E')
def f(y):
    """

    Parameters
    ----------
    x

    Returns
    -------

    """
    return y**2


if __name__ == '__main__':
    #turb.dev._sounds.test_play_audio()
    x = f()
