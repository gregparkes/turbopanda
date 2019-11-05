#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:31:39 2019

@author: gparkes
"""

from setuptools import setup, find_packages

with open("README.md","r") as fh:
    long_description = fh.read()

setup(
    name="turbopanda",
    version="0.0.5",
    description="Turbo-charging the Pandas library in an integrative, meta-orientated style",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gregparkes/turbopanda",
    author="Gregory Parkes",
    author_email="g.m.parkes@soton.ac.uk",
    license="GPL-3.0",
    packages=find_packages(),
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.11.0",
        "scipy>=1.3",
        "pandas>=0.25.1",
    ],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Framework :: IPython",
        "Framework :: Jupyter",
    ],
)