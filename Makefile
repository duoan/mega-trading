.PHONY: test demo platform-demo

test:
	uv run python -m unittest discover -s tests

demo:
	uv run mega-trading ingest --config configs/ingest-demo.toml
	uv run mega-trading build data.data_dir=.mega-trading/demo data.source=fixture build.block_size=8 build.stride=4 build.min_events_per_ticker=2
	uv run mega-trading train data.data_dir=.mega-trading/demo run.run_id=demo training.max_steps=1 training.batch_size=2 training.eval_interval=1 training.device=cpu model.hidden_dim=8 model.layers=1 model.attention_heads=1 training.wandb_enabled=false

platform-demo:
	uv run mega-trading --help
	uv run mega-trading build --help
	uv run mega-trading train --help
	uv run mega-trading eval --help
