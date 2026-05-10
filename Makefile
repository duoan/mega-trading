.PHONY: test demo local remote server platform-demo sync-wandb-secret install-flash-attn download-binance-local download-binance-modal prep-binance-modal prep-server upload-binance-modal train-modal-binance train-server-rtx6000

test:
	uv run python -m unittest discover -s tests

demo:
	uv run python scripts/prepare_numpy_dataset.py --ingest-config configs/ingest-demo.toml --config-name default build.block_size=4 build.stride=2 build.min_events_per_ticker=2
	uv run mega-trading train data.data_dir=.mega-trading/demo run.run_id=demo training.max_steps=1 training.batch_size=2 training.eval_interval=1 training.device=cpu model.hidden_dim=8 model.layers=1 model.attention_heads=1 training.wandb_enabled=false

local:
	uv run python scripts/run_pipeline.py local

remote:
	uv run python scripts/run_pipeline.py remote

server: prep-server train-server-rtx6000

platform-demo:
	uv run mega-trading --help
	uv run mega-trading prepare --help
	uv run mega-trading train --help
	uv run mega-trading eval --help

sync-wandb-secret:
	uv run python scripts/sync_wandb_modal_secret.py

install-flash-attn:
	uv run python scripts/install_flash_attn.py --require-cuda

download-binance-local:
	uv run python scripts/download_binance_archives.py --ingest-config configs/ingest-binance-local.toml

download-binance-modal:
	uv run python scripts/download_binance_archives.py --ingest-config configs/ingest-binance-modal-prep.toml

prep-binance-modal: download-binance-modal
	uv run python scripts/prepare_numpy_dataset.py --ingest-config configs/ingest-binance-modal-prep.toml --config-name binance-modal-prep

prep-server: prep-binance-modal

upload-binance-modal:
	uv run modal volume put mega-trading-artifacts .mega-trading/binance-modal/stage=05_shards /binance-trades/stage=05_shards

train-modal-binance:
	uv run modal run modal_train.py --mode cluster --run-id modal-binance --data-dir /data/binance-trades --strategy fsdp --max-steps 50000

train-server-rtx6000: install-flash-attn
	uv run mega-trading train --config-name server-rtx6000
