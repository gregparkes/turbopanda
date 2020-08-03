#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Produces decorator sounds for success and fail operations."""

# future imports
from __future__ import absolute_import, division, print_function

import itertools as it
from typing import Callable, List
from functools import wraps

import numpy as np

from turbopanda.utils import belongs, union

__all__ = ['bleep', "Bleep"]


def _get_note_progression(n, A_4=440):
    """Given base note A4 frequency in Hz, calculate note $n$ steps away.

    Determined as A_4 * (2^1/12)^n.
    """
    return A_4 * np.power(np.power(2, 1. / 12.), n)


def _get_notes_flat():
    """Returns the musical notes as strings, from C, using only flats."""
    return 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'


def _get_notes_sharp():
    """Returns the musical notes as strings, from C, using only sharps."""
    return 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'


def _get_notes_all():
    return union(_get_notes_flat(), _get_notes_sharp())


def _major_arpeggios():
    """Given a note, give the next 2 major arpeggio progression notes.

    e.g 'C' gives: 'E', 'G' or 3, 5, 'D' gives: 'F#', 'A'.

    (including upper note).
    """
    return {
        'C': ['E', 'G'], 'D': ['F#', 'A'], 'E': ['G#', 'B'],
        'F': ['A', 'C'], 'G': ['B', 'D'], 'A': ['C#', 'E'],
        'B': ['D#', 'F#'], 'Db': ['F', 'Ab'], 'Eb': ['G', 'Bb'],
        'F#': ['A#', 'Db'], 'Ab': ['C', 'Eb'], 'Bb': ['D', 'F'],
        # also map equivalents e.g F#=Gb
        'C#': ['F', 'Ab'], 'D#': ['G', 'Bb'], 'Gb': ['A#', 'Db'],
        'G#': ['C', 'Eb'], 'A#': ['D', 'F']
    }


def _minor_arpeggios():
    """Given a note, give the next 2 minor arpeggio pregression notes.

    e.g 'C' gives: 'Eb', 'G' or 3, 5, 'D' gives: 'F', 'A'.

    (including upper note).
    """
    return {
        'C': ['Eb', 'G'], 'D': ['F', 'A'], 'E': ['G', 'B'],
        'F': ['Ab', 'C'], 'G': ['Bb', 'D'], 'A': ['C', 'E'],
        'B': ['D', 'F#'], 'C#': ['E', 'G#'], 'Eb': ['Gb', 'Bb'],
        'F#': ['A', 'C#'], 'G#': ['B', 'D#'], 'Bb': ['Db', 'F'],
        # also map equivalents e.g F#=Gb
        'Db': ['E', 'G#'], 'D#': ['Gb', 'Bb'], 'Gb': ['A', 'C#'],
        'Ab': ['B', 'D#'], 'A#': ['Db', 'F'],
    }


def _get_arpeggio(note='C', key='major', level=4):
    """Given a note and key, return the 4 notes that form an arpeggio."""
    if key == "major":
        arp = _major_arpeggios()
    elif key == 'minor':
        arp = _minor_arpeggios()
    else:
        raise ValueError("key '{}' not recognised".format(key))

    three_five = arp[note]
    l_i = "_%s" % str(level)
    l_j = "_%s" % str(level + 1)
    return [note + l_i, three_five[0] + l_i, three_five[1] + l_i, note + l_j]


def _get_notepack():
    hz = _get_note_progression(n=np.arange(-57, 51, 1))
    note_range_flat = list(
        it.chain.from_iterable([list(map(lambda x: x + '_%s' % str(i), _get_notes_flat())) for i in range(9)]))
    note_range_sharp = list(
        it.chain.from_iterable([list(map(lambda x: x + '_%s' % str(i), _get_notes_sharp())) for i in range(9)]))
    return {**dict(zip(note_range_flat, hz)), **dict(zip(note_range_sharp, hz))}


def _produce_audio(notes: List[str], seconds=2, fs=44100):
    """Given a list of notes, produce a musical audio of evenly spaced notes.

    Inputs could e.g ['C_4', 'E_4', 'G_4', 'C_5'] for a C major chord.
    """
    notepack = _get_notepack()
    # get frequencies
    fz = [notepack[a] for a in notes]
    # create timeloop
    sec_scaled = seconds / len(notes)
    t = np.linspace(0, sec_scaled, int(sec_scaled * fs), False)
    # create grouped sine waves
    note = np.hstack([np.sin(n * t * 2 * np.pi) for n in fz])
    # join together and move note into 16-bit range
    audio = note * (np.power(2, 15) - 1) / np.max(np.abs(note))
    # cast note as 16-bit and return
    return audio.astype(np.int16)


def _play_audio(audio, fs=44100):
    import simpleaudio as sa
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    # wait for playback to finish
    play_obj.wait_done()


def _play_arpeggio(note='C', key="major"):
    # plays the arpeggio given a note and key
    arp = _get_arpeggio(note=note, key=key)
    # get audio
    aud = _produce_audio(arp)
    # play
    _play_audio(aud)


def test_play_audio():
    aud = _produce_audio(['C_4', 'E_4', 'G_4', 'C_5'])
    _play_audio(aud)


def bleep(_func=None, *, note='C') -> Callable:
    """Provides automatic sound release when a function has completed.

    .. note:: this requires the `simpleaudio` package to run.

    Note chord progression is played at the *end* of the function, and not the start.

    Parameters
    ----------
    note : str
        Must be {'A':'G'}

    Examples
    --------
    >>> from turbopanda.dev import bleep
    >>> @bleep
    >>> def f(x):
    ...     # compute some long function here
    ...     pass
    """
    belongs(note, list(_get_notes_all()))

    # define decorator
    def _decorator_wrap(func):
        @wraps(func)
        def _bleep_function(*args, **kwargs):
            # enter try-catch and if success, positive noise, or failure, negative noise.
            try:
                result = func(*args, **kwargs)
                # make positive noise
                _play_arpeggio(note.upper(), key="major")
                # return
                return result
            except Exception as e:
                # make negative noise
                _play_arpeggio(note.upper(), key="minor")
                print(e.args)
        return _bleep_function

    if _func is None:
        return _decorator_wrap
    else:
        return _decorator_wrap(_func)


""" A Bleep class to decorate a block of code with. """


class Bleep:
    """Handles noise sound with a `with` statement."""

    def __enter__(self):
        pass

    def __exit__(self, type_, value, traceback):
        _play_arpeggio("C", key="major")
