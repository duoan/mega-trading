.PHONY: test demo local remote platform-demo sync-wandb-secret prep-binance-modal upload-binance-modal train-modal-binance

test:
	uv run python -m unittest discover -s tests

demo:
	uv run python scripts/prepare_numpy_dataset.py --ingest-config configs/ingest-demo.toml --config-name default build.block_size=4 build.stride=2 build.min_events_per_ticker=2
	uv run mega-trading train data.data_dir=.mega-trading/demo run.run_id=demo training.max_steps=1 training.batch_size=2 training.eval_interval=1 training.device=cpu model.hidden_dim=8 model.layers=1 model.attention_heads=1 training.wandb_enabled=false

local:
	uv run python scripts/run_pipeline.py local

remote:
	uv run python scripts/run_pipeline.py remote

platform-demo:
	uv run mega-trading --help
	uv run mega-trading prepare --help
	uv run mega-trading train --help
	uv run mega-trading eval --help

sync-wandb-secret:
	uv run python scripts/sync_wandb_modal_secret.py

prep-binance-modal:
	uv run python scripts/prepare_numpy_dataset.py --ingest-config configs/ingest-binance-modal-prep.toml --config-name binance-modal-prep

upload-binance-modal:
	uv run modal volume put mega-trading-artifacts .mega-trading/binance-modal /binance-trades

train-modal-binance:
	uv run modal run modal_train.py --mode cluster --run-id modal-binance --data-dir /data/binance-trades --strategy fsdp --max-steps 50000
