UV = uv run

clean:
	find . -name .venv -type d -exec rm -rf {} +
	find . -name __pycache__ -type d -exec rm -rf {} +
	find . -name .pytest_cache -type d -exec rm -rf {} +
	find . -name .ruff_cache -type d -exec rm -rf {} +
	find . -name .mypy_cache -type d -exec rm -rf {} +

lint:
	${UV} ruff check --fix .
	${UV} ruff format .

pre-commit: lint clean