.PHONY: test demo mac rtx modal platform-demo install-flash-attn download-mac download-rtx download-modal prep-mac prep-rtx prep-modal upload-modal train-mac train-rtx train-modal pull-modal-artifacts backtest-mac backtest-rtx backtest-modal report-mac report-rtx report-modal

test:
	uv run python -m unittest discover -s tests

demo:
	uv run python scripts/prepare_numpy_dataset.py --ingest-config configs/ingest-demo.toml --config-name default build.block_size=4 build.stride=2 build.min_events_per_ticker=2
	uv run mega-trading train data.data_dir=.mega-trading/demo run.run_id=demo training.max_steps=1 training.batch_size=2 training.eval_interval=1 training.device=cpu model.hidden_dim=8 model.layers=1 model.attention_heads=1 training.mlflow_enabled=false

mac:
	uv run python scripts/run_pipeline.py mac

rtx:
	uv run python scripts/run_pipeline.py rtx

modal:
	uv run python scripts/run_pipeline.py modal

platform-demo:
	uv run mega-trading --help
	uv run mega-trading prepare --help
	uv run mega-trading train --help
	uv run mega-trading eval --help
	uv run mega-trading backtest --help
	uv run mega-trading report --help

install-flash-attn:
	uv run python scripts/install_flash_attn.py --require-cuda

download-mac:
	uv run python scripts/download_binance_archives.py --ingest-config configs/ingest-mac.toml

download-rtx:
	uv run python scripts/download_binance_archives.py --ingest-config configs/ingest-rtx.toml

download-modal:
	uv run python scripts/download_binance_archives.py --ingest-config configs/ingest-modal.toml

prep-mac:
	uv run python scripts/prepare_numpy_dataset.py --ingest-config configs/ingest-mac.toml --config-name mac

prep-rtx: download-rtx
	uv run python scripts/prepare_numpy_dataset.py --ingest-config configs/ingest-rtx.toml --config-name rtx

prep-modal: download-modal
	uv run python scripts/prepare_numpy_dataset.py --ingest-config configs/ingest-modal.toml --config-name modal data.data_dir=.mega-trading/data

upload-modal:
	uv run modal volume put mega-trading-artifacts .mega-trading/data/datasets /shared/datasets

train-mac:
	uv run mega-trading train --config-name mac

train-rtx:
	uv run mega-trading train --config-name rtx

train-modal:
	uv run modal run modal_train.py --mode cluster --run-id modal --config-name modal --data-dir /data/shared --strategy fsdp --max-steps 50000

pull-modal-artifacts:
	mkdir -p .mega-trading/data/runs .mega-trading/data/manifests
	uv run modal volume get --force mega-trading-artifacts /shared/runs .mega-trading/data/runs
	uv run modal volume get --force mega-trading-artifacts /shared/manifests .mega-trading/data/manifests

backtest-mac:
	uv run mega-trading backtest --config-name mac --max-batches 32

backtest-rtx:
	uv run mega-trading backtest --config-name rtx --max-batches 128

backtest-modal:
	uv run mega-trading backtest --config-name modal --max-batches 128 data.data_dir=.mega-trading/data eval.device=auto

report-mac:
	uv run mega-trading report --config-name mac

report-rtx:
	uv run mega-trading report --config-name rtx

report-modal:
	uv run mega-trading report --config-name modal data.data_dir=.mega-trading/data
