.PHONY: test demo platform-demo

test:
	uv run python -m unittest discover -s tests

demo:
	uv run mega-trading --help

platform-demo:
	uv run mega-trading --help
	uv run mega-trading replay --help
	uv run mega-trading materialize-labels --help
	uv run mega-trading backtest --help
	uv run mega-trading online-update --help
	uv run mega-trading serve-smoke --help
