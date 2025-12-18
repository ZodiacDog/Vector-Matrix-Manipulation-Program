.PHONY: help install test clean examples run-server lint format

help:
	@echo "VecMatrix - Make Commands"
	@echo "========================="
	@echo "install     : Install package and dependencies"
	@echo "test        : Run test suite"
	@echo "examples    : Run example scripts"
	@echo "run-server  : Start API server"
	@echo "clean       : Remove build artifacts"
	@echo "lint        : Run code quality checks"
	@echo "format      : Format code with black"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	python test_vecmatrix.py

examples:
	python examples.py

run-server:
	python api_server.py --host localhost --port 8000

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

lint:
	flake8 core.py cli.py api_server.py test_vecmatrix.py examples.py
	mypy core.py cli.py api_server.py

format:
	black core.py cli.py api_server.py test_vecmatrix.py examples.py
