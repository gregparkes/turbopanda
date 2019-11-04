"""
Basic set up file
"""

from setuptools import setup, find_packages

setup(
	name="skmodels",
	version="0.0.1",
	description="A python package for modelling complex ScitkitLearn pipelines and models",
	url="https://github.com/gregparkes/skmodels",
	author="Gregory Parkes",
	author_email="g.m.parkes@soton.ac.uk",
	license="MIT",
	packages=find_packages(),
	zip_safe=False,
	install_requires=[
		"numpy","pandas","matplotlib", "itertools"
	],
	classifiers=[
		"Natural Language :: English",
		"Programming Language :: Python :: 3.6",
		"License :: OSI Approved :: MIT License",
		"Intended Audience :: Science/Research",
		"Framework :: IPython", "Framework :: Jupyter",
		"Development Status :: 1 - Planning"
	],
)

