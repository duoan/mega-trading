.PHONY: test demo

test:
	uv run python -m unittest discover -s tests

demo:
	uv run marketfm --help
