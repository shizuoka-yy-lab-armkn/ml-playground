.PHONY:	install
install:
	poetry install
	rm -f .venv/bin/black
	ln -s pyink .venv/bin/black


.PHONY:	fmt
fmt:
	poetry run pyink .
	poetry run isort .


.PHONY:	lint
lint:
	poetry run ruff check .


.PHONY: typecheck
typecheck:
	poetry run pyright .
