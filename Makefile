OS := $(shell uname -s)

.PHONY:	setup
setup:
	ln -sf poetry.$(OS).lock poetry.lock

.PHONY:	install
install:
	poetry lock --no-update
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
