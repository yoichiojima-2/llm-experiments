install:
	uv sync
	uv pip install -e .

clean:
	-rm .DS_Store
	find . -name "*.ipynb" -type f -exec uv run jupyter nbconvert --clear-output {} +
	find . -name .venv -type d -exec rm -rf {} +
	find . -name __pycache__ -type d -exec rm -rf {} +
	find . -name .pytest_cache -type d -exec rm -rf {} +
	find . -name .ruff_cache -type d -exec rm -rf {} +
	find . -name .mypy_cache -type d -exec rm -rf {} +
	find . -name .ipynb_checkpoints -type d -exec rm -rf {} +
	find . -name .cache -type f -exec rm {} +
	find . -name uv.lock -type f -exec rm {} +
	find . -name "*.egg-info" -type d -exec rm -rf {} +
	-rm -rf db
	deactivate

lint:
	uv run isort .
	uv run ruff check --fix .
	uv run ruff format .

pre-commit: lint clean

test: install
	uv run pytest -vvv -s
