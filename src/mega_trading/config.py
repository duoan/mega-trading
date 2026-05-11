"""Configuration contracts for Mega-Trading."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuildConfig:
    mixture_name: str = "public"
    source: str = "fixture"
    block_size: int = 128
    stride: int = 64
    min_events_per_ticker: int = 32
    max_tickers: int | None = None
    tokenizer_method: str = "quantile"
    tokenizer_clip_quantile: float = 0.01
    tokenizer_relative_price_bins: int = 16
    tokenizer_price_bins: int = 16
    tokenizer_size_bins: int = 16
    tokenizer_time_bins: int = 4
    numpy_partition_rows: int | None = None
    validation_fraction: float = 0.1
    backtest_fraction: float = 0.1
    streaming_prepare: bool = False
    streaming_tokenizer_sample_events: int = 250_000
    streaming_baseline_sample_rows: int = 200_000

    def __post_init__(self) -> None:
        if self.block_size < 4:
            raise ValueError("block_size must be at least 4")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.min_events_per_ticker < 2:
            raise ValueError("min_events_per_ticker must be at least 2")
        if self.max_tickers is not None and self.max_tickers <= 0:
            raise ValueError("max_tickers must be positive when set")
        if self.tokenizer_method not in {"quantile", "histogram"}:
            raise ValueError("tokenizer_method must be one of: quantile, histogram")
        if not 0.0 <= self.tokenizer_clip_quantile < 0.5:
            raise ValueError("tokenizer_clip_quantile must be in [0.0, 0.5)")
        for name, value in {
            "tokenizer_relative_price_bins": self.tokenizer_relative_price_bins,
            "tokenizer_price_bins": self.tokenizer_price_bins,
            "tokenizer_size_bins": self.tokenizer_size_bins,
            "tokenizer_time_bins": self.tokenizer_time_bins,
        }.items():
            if value < 2:
                raise ValueError(f"{name} must be at least 2")
        if self.numpy_partition_rows is not None and self.numpy_partition_rows <= 0:
            raise ValueError("numpy_partition_rows must be positive when set")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0.0, 1.0)")
        if not 0.0 <= self.backtest_fraction < 1.0:
            raise ValueError("backtest_fraction must be in [0.0, 1.0)")
        if self.validation_fraction + self.backtest_fraction >= 1.0:
            raise ValueError("validation_fraction + backtest_fraction must be less than 1.0")
        if self.streaming_tokenizer_sample_events <= 0:
            raise ValueError("streaming_tokenizer_sample_events must be positive")
        if self.streaming_baseline_sample_rows <= 0:
            raise ValueError("streaming_baseline_sample_rows must be positive")


@dataclass(frozen=True)
class TrainConfig:
    run_id: str
    mixture_name: str = "public"
    max_steps: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    lr_schedule: str = "constant"
    lr_warmup_steps: int = 0
    min_learning_rate: float = 0.0
    validation_fraction: float = 0.1
    eval_interval: int = 20
    max_eval_batches: int | None = None
    metric_interval: int = 1
    preload_numpy_arrays: bool = False
    dataloader_num_workers: int = 0
    dataloader_prefetch_factor: int = 2
    dataloader_pin_memory: bool = False
    dataloader_persistent_workers: bool = False
    hidden_dim: int = 128
    layers: int = 4
    attention_heads: int = 4
    kv_heads: int | None = None
    intermediate_dim: int | None = None
    dropout: float = 0.1
    rope_theta: float = 500_000.0
    norm_eps: float = 1e-5
    seed: int = 7
    device: str = "auto"
    precision: str = "auto"
    distributed_strategy: str = "ddp"
    gradient_accumulation_steps: int = 1
    compile: bool = False
    compile_mode: str = "default"
    attention_backend: str = "auto"
    checkpoint_interval: int | None = None
    resume_from_checkpoint: str | None = None
    mlflow_enabled: bool = True
    mlflow_experiment: str = "mega-trading"
    mlflow_tracking_uri: str | None = None
    progress_bar: bool = True
    profiler_enabled: bool = False
    nvtx_enabled: bool = False
    profiler_trace_dir: str | None = None
    profiler_wait_steps: int = 1
    profiler_warmup_steps: int = 1
    profiler_active_steps: int = 3
    profiler_repeat: int = 1
    profiler_record_shapes: bool = False
    profiler_profile_memory: bool = False
    profiler_with_stack: bool = False

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.optimizer not in {"adamw", "muon"}:
            raise ValueError("optimizer must be one of: adamw, muon")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= self.adam_beta1 < 1.0:
            raise ValueError("adam_beta1 must be in [0.0, 1.0)")
        if not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError("adam_beta2 must be in [0.0, 1.0)")
        if self.adam_eps <= 0:
            raise ValueError("adam_eps must be positive")
        if not 0.0 <= self.muon_momentum < 1.0:
            raise ValueError("muon_momentum must be in [0.0, 1.0)")
        if self.muon_ns_steps <= 0:
            raise ValueError("muon_ns_steps must be positive")
        if self.lr_schedule not in {"constant", "cosine"}:
            raise ValueError("lr_schedule must be one of: constant, cosine")
        if self.lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps must be non-negative")
        if not 0.0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError("min_learning_rate must be in [0.0, learning_rate]")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0.0, 1.0)")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.max_eval_batches is not None and self.max_eval_batches <= 0:
            raise ValueError("max_eval_batches must be positive when set")
        if self.metric_interval <= 0:
            raise ValueError("metric_interval must be positive")
        if self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must be non-negative")
        if self.dataloader_prefetch_factor <= 0:
            raise ValueError("dataloader_prefetch_factor must be positive")
        if self.dataloader_persistent_workers and self.dataloader_num_workers == 0:
            raise ValueError("dataloader_persistent_workers requires dataloader_num_workers > 0")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.layers <= 0:
            raise ValueError("layers must be positive")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.hidden_dim // self.attention_heads % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.kv_heads is not None:
            if self.kv_heads <= 0:
                raise ValueError("kv_heads must be positive when set")
            if self.attention_heads % self.kv_heads != 0:
                raise ValueError("attention_heads must be divisible by kv_heads")
        if self.intermediate_dim is not None and self.intermediate_dim <= 0:
            raise ValueError("intermediate_dim must be positive when set")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        if self.precision not in {"auto", "fp32", "mixed"}:
            raise ValueError("precision must be one of: auto, fp32, mixed")
        if self.distributed_strategy not in {"ddp", "fsdp"}:
            raise ValueError("distributed_strategy must be one of: ddp, fsdp")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError("compile_mode must be one of: default, reduce-overhead, max-autotune")
        if self.attention_backend not in {"auto", "flash", "efficient", "math", "triton"}:
            raise ValueError("attention_backend must be one of: auto, flash, efficient, math, triton")
        if self.checkpoint_interval is not None and self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive when set")
        if self.resume_from_checkpoint is not None and not self.resume_from_checkpoint:
            raise ValueError("resume_from_checkpoint must be a non-empty path when set")
        if self.mlflow_tracking_uri is not None and not self.mlflow_tracking_uri:
            raise ValueError("mlflow_tracking_uri must be a non-empty path or URI when set")
        if self.profiler_trace_dir is not None and not self.profiler_trace_dir:
            raise ValueError("profiler_trace_dir must be a non-empty path when set")
        if self.profiler_wait_steps < 0:
            raise ValueError("profiler_wait_steps must be non-negative")
        if self.profiler_warmup_steps < 0:
            raise ValueError("profiler_warmup_steps must be non-negative")
        if self.profiler_active_steps <= 0:
            raise ValueError("profiler_active_steps must be positive")
        if self.profiler_repeat <= 0:
            raise ValueError("profiler_repeat must be positive")
