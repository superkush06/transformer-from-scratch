# One command per task. Everything under examples/ and docs/ is a script
# rather than part of the installed package, so each recipe supplies the
# PYTHONPATH the README would otherwise ask you to type.
#
#   make test        the suite CI runs
#   make lint        the linter CI runs
#   make gradcheck   all 1,312 gradients against central differences (~1 s)
#   make validate    every number in docs/validation.md (~20 s)
#   make handoff     the regime -> next-label distribution example (~14 s)
#   make figures     redraw the three README figures (~15 s, needs matplotlib)
#   make check       lint + test, i.e. what a push has to survive

PYTHON ?= python3
export PYTHONPATH := .

.PHONY: help install test lint check gradcheck validate handoff figures clean

help:
	@sed -n 's/^#   //p' $(MAKEFILE_LIST)

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

check: lint test

gradcheck:
	$(PYTHON) examples/gradcheck.py

validate:
	$(PYTHON) examples/validate.py

handoff:
	$(PYTHON) examples/regime_handoff.py

figures:
	$(PYTHON) docs/figures.py

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
