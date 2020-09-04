export SHELL := /bin/bash

test:
    pytest --doctest-modules turbopanda

unittests:
    pytest turbopanda

coverage:
    pytest --doctest-modules --cov=turbopanda --cov-config=.coveragerc turbopanda

lint:
	flake8 --ignore E121,E123,E126,E226,E24,E704,E741,W503,W504 --exclude turbopanda/__init__.py,turbopanda/_extras/__init__.py