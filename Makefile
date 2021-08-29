
help:
	@echo "test - run tests quickly with the default Python"
	@echo "clean-pyc - remove Python file artifacts"

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

test: clean-pyc
	python setup.py test

coverage: clean-pyc
	coverage run -m --source luxai pytest tests
	coverage html --omit="tests/*,*/__init__.py"
	xdg-open  htmlcov/index.html

clean: clean-pyc
	rm -r htmlcov .coverage

env-export:
	conda env export > environment.yml