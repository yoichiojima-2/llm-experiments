UV = uv run

clean:
	find . -name "*.ipynb" -type f -exec ${UV} jupyter nbconvert --clear-output {} +
	find . -name .venv -type d -exec rm -rf {} +
	find . -name __pycache__ -type d -exec rm -rf {} +
	find . -name .pytest_cache -type d -exec rm -rf {} +
	find . -name .ruff_cache -type d -exec rm -rf {} +
	find . -name .mypy_cache -type d -exec rm -rf {} +
	find . -name .ipynb_checkpoints -type d -exec rm -rf {} +
	find . -name .cache -type f -exec rm {} +
	find . -name uv.lock -type f -exec rm {} +
	find . -name "*.egg-info" -type d -exec rm -rf {} +

lint:
	${UV} isort .
	${UV} ruff check --fix .
	${UV} ruff format .

pre-commit: lint clean

test:
	${UV} pytest -vvv
