"""Benchmark the local Triton fused linear cross-entropy kernel.

Defaults match configs/rtx.yaml plus the rtx tokenizer vocab:
batch=8, sequence=512, hidden_dim=1024, vocab_size=14583, dtype=bf16.

Reports both latency (forward and forward+backward) and peak CUDA memory so the
HBM savings from skipping the [B, T, V] logits tensor are visible.
"""

from __future__ import annotations

import argparse
import json
from typing import Callable

import torch
import torch.nn.functional as F

from mega_trading.kernels.triton_ops import triton_fused_linear_cross_entropy


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Triton fused linear cross-entropy for Mega-Trading shapes.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    # rtx tokenizer: 3 specials + 2*2*9*9*9*5 = 14583 joint event tokens.
    parser.add_argument("--vocab-size", type=int, default=14_583)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--mode", choices=("forward", "forward-backward"), default="forward-backward")
    parser.add_argument("--ignore-index", type=int, default=-100)
    parser.add_argument("--ignored-fraction", type=float, default=0.0, help="fraction of rows to mark as ignored")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--no-validate", action="store_true", help="skip the parity check against PyTorch")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark the fused linear cross-entropy kernel")

    dtype = _dtype(args.dtype)
    rows = args.batch_size * args.sequence_length
    torch.manual_seed(53)

    requires_grad = args.mode == "forward-backward"

    def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.randn(rows, args.hidden_dim, device="cuda", dtype=dtype)
        # Initialize the projection like the trained model so logits stay numerically reasonable.
        weight = torch.randn(args.vocab_size, args.hidden_dim, device="cuda", dtype=dtype) * 0.02
        labels = torch.randint(low=0, high=args.vocab_size, size=(rows,), device="cuda", dtype=torch.long)
        if args.ignored_fraction > 0.0:
            mask = torch.rand(rows, device="cuda") < args.ignored_fraction
            labels[mask] = args.ignore_index
        if requires_grad:
            hidden.requires_grad_(True)
            weight.requires_grad_(True)
        return hidden, weight, labels

    hidden, weight, labels = _make_inputs()

    if not args.no_validate:
        _validate_parity(hidden, weight, labels, args.ignore_index, args.mode)

    triton_runner = lambda: _step_triton(hidden, weight, labels, args.ignore_index, args.mode)
    torch_runner = lambda: _step_torch(hidden, weight, labels, args.ignore_index, args.mode)

    triton_row = _benchmark_row("triton_fused_linear_ce", triton_runner, args)
    torch_row = _benchmark_row("torch_linear_plus_ce", torch_runner, args)

    payload = {
        "device": torch.cuda.get_device_name(),
        "shape": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "hidden_dim": args.hidden_dim,
            "vocab_size": args.vocab_size,
            "dtype": args.dtype,
            "mode": args.mode,
            "ignored_fraction": args.ignored_fraction,
        },
        "results": [triton_row, torch_row],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_report(payload)
    return 0


def _validate_parity(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    mode: str,
) -> None:
    detached_hidden = hidden.detach().clone().requires_grad_(hidden.requires_grad)
    detached_weight = weight.detach().clone().requires_grad_(weight.requires_grad)

    triton_loss = triton_fused_linear_cross_entropy(
        detached_hidden, detached_weight, labels, ignore_index=ignore_index
    )
    reference_hidden = hidden.detach().clone().float().requires_grad_(hidden.requires_grad)
    reference_weight = weight.detach().clone().float().requires_grad_(weight.requires_grad)
    reference_loss = F.cross_entropy(
        F.linear(reference_hidden, reference_weight),
        labels,
        ignore_index=ignore_index,
    )
    _assert_close("loss", triton_loss.float(), reference_loss, atol=5e-3, rtol=5e-3)
    if mode == "forward-backward":
        triton_loss.backward()
        reference_loss.backward()
        _assert_close("grad_hidden", detached_hidden.grad.float(), reference_hidden.grad, atol=5e-2, rtol=5e-2)
        _assert_close("grad_weight", detached_weight.grad.float(), reference_weight.grad, atol=5e-2, rtol=5e-2)


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        max_diff = (actual - expected).abs().max().item()
        raise RuntimeError(f"Triton fused linear CE diverged from PyTorch on {name}; max_diff={max_diff:.4f}")


def _step_triton(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    mode: str,
) -> torch.Tensor:
    loss = triton_fused_linear_cross_entropy(hidden, weight, labels, ignore_index=ignore_index)
    if mode == "forward":
        return loss
    grads = torch.autograd.grad(loss, (hidden, weight), retain_graph=False, create_graph=False)
    return grads[0]


def _step_torch(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    mode: str,
) -> torch.Tensor:
    logits = F.linear(hidden, weight)
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)
    if mode == "forward":
        return loss
    grads = torch.autograd.grad(loss, (hidden, weight), retain_graph=False, create_graph=False)
    return grads[0]


def _benchmark_row(name: str, fn: Callable[[], torch.Tensor], args: argparse.Namespace) -> dict[str, float | str]:
    latency_ms = _cuda_event_time_ms(fn, warmup=args.warmup, iterations=args.iterations)
    peak_mb = _peak_memory_mb(fn, warmup=args.warmup)
    return {"name": name, "latency_ms": latency_ms, "peak_memory_mb": peak_mb}


def _cuda_event_time_ms(fn: Callable[[], torch.Tensor], *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def _peak_memory_mb(fn: Callable[[], torch.Tensor], *, warmup: int) -> float:
    # Warm up so any one-time Triton/cuBLAS scratch buffers are already allocated.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _print_report(payload: dict[str, object]) -> None:
    shape = payload["shape"]
    print(f"device: {payload['device']}")
    print(
        "shape: "
        f"batch={shape['batch_size']} seq={shape['sequence_length']} "
        f"hidden_dim={shape['hidden_dim']} vocab={shape['vocab_size']} "
        f"dtype={shape['dtype']} mode={shape['mode']} ignored={shape['ignored_fraction']}"
    )
    print("")
    print(f"{'kernel':<28} {'latency_ms':>12} {'peak_mb':>10}")
    for row in payload["results"]:
        print(f"{row['name']:<28} {row['latency_ms']:>12.4f} {row['peak_memory_mb']:>10.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
