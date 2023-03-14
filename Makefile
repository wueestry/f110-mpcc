.PHONY: check format lint test
black_args = --line-length=99 --exclude='/(venv.*|__pycache__|env|\..*)/'
isort_args = --line-length=99 --multi-line=3 --profile=black --skip-glob='*/env*' --skip-glob='*/.*' --skip='*/__pycache__'
flake_args = --ignore='W503,E203,E501' --exclude='*/env*/,env*/,*/.*,.*,*/__pycache__,__pycache__'
dest = .

check: lint test

format:
	python3 -m black $(black_args) $(dest)
	python3 -m isort --atomic $(isort_args) $(dest)

lint:
	python3 -m black --check $(black_args) $(dest)
	python3 -m isort --check $(isort_args) $(dest)
	python3 -m flake8 $(flake_args) $(dest)
	python3 -m mypy $(dest)

test:
	python3 -m unittest discover . -p '*_test.py'
	pytest -s
