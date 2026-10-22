.PHONY: install test lint run list
install:
	python -m pip install -e '.[dev,notebook]'
test:
	python -m pytest -q
lint:
	python -m ruff check src tests
run:
	python -m curse_of_dim --experiment all --out outputs
list:
	python -m curse_of_dim --list
