.PHONY: test demo platform-demo

test:
	uv run python -m unittest discover -s tests

demo:
	uv run mega-trading ingest --config configs/ingest-demo.toml
	uv run mega-trading train data.data_dir=.mega-trading/demo data.mixture=demo run.run_id=demo training.max_steps=1 training.batch_size=2 training.eval_interval=1 training.device=cpu model.hidden_dim=8

platform-demo:
	uv run mega-trading --help
	uv run mega-trading replay --help
	uv run mega-trading materialize-labels --help
	uv run mega-trading backtest --help
	uv run mega-trading online-update --help
	uv run mega-trading serve-smoke --help
