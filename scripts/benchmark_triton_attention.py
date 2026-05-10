"""Benchmark the local Triton causal GQA attention kernel.

Defaults match configs/server-rtx6000.yaml:
batch=8, sequence=512, query heads=16, kv heads=4, head dim=64, dtype=bf16.
"""

from __future__ import annotations

import argparse
import json
from typing import Callable

import torch
import torch.nn.functional as F

from mega_trading.kernels.triton_attention import triton_attention


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Triton causal GQA attention for Mega-Trading shapes.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--mode", choices=("forward", "forward-backward"), default="forward")
    parser.add_argument("--skip-native", action="store_true", help="only benchmark the local Triton kernel")
    parser.add_argument("--no-validate", action="store_true", help="skip output comparison against PyTorch SDPA")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark Triton attention")
    if args.query_heads % args.kv_heads != 0:
        raise ValueError("query-heads must be divisible by kv-heads")

    dtype = _dtype(args.dtype)
    repeats = args.query_heads // args.kv_heads
    torch.manual_seed(7)
    query = torch.randn(
        args.batch_size,
        args.query_heads,
        args.sequence_length,
        args.head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=args.mode == "forward-backward",
    )
    key = torch.randn(
        args.batch_size,
        args.kv_heads,
        args.sequence_length,
        args.head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=args.mode == "forward-backward",
    )
    value = torch.randn_like(key, requires_grad=args.mode == "forward-backward")
    grad_output = torch.randn_like(query)

    context = torch.enable_grad() if args.mode == "forward-backward" else torch.inference_mode()
    with context:
        triton_out = triton_attention(query, key, value, repeats=repeats)
        native_out = None
        if not args.skip_native:
            native_out = _native_attention(query, key, value, repeats)
        if native_out is not None and not args.no_validate:
            tolerance = 7e-2 if dtype in {torch.float16, torch.bfloat16} else 1e-4
            if not torch.allclose(triton_out, native_out, atol=tolerance, rtol=tolerance):
                max_diff = (triton_out - native_out).abs().max().item()
                raise RuntimeError(f"Triton output diverged from PyTorch SDPA; max_diff={max_diff:.6f}")
            if args.mode == "forward-backward":
                triton_grads = torch.autograd.grad(triton_out, (query, key, value), grad_output, retain_graph=True)
                native_grads = torch.autograd.grad(native_out, (query, key, value), grad_output, retain_graph=True)
                for name, triton_grad, native_grad in zip(("query", "key", "value"), triton_grads, native_grads, strict=True):
                    if not torch.allclose(triton_grad, native_grad, atol=tolerance, rtol=tolerance):
                        max_diff = (triton_grad - native_grad).abs().max().item()
                        raise RuntimeError(f"Triton {name} gradient diverged from PyTorch SDPA; max_diff={max_diff:.6f}")

        results = [
            _benchmark_row(
                "triton_descriptor",
                lambda: _attention_step(query, key, value, grad_output, repeats, args.mode, triton_attention),
                args,
            )
        ]
        if not args.skip_native:
            results.append(
                _benchmark_row(
                    "torch_sdpa",
                    lambda: _attention_step(query, key, value, grad_output, repeats, args.mode, _native_attention),
                    args,
                )
            )

    payload = {
        "device": torch.cuda.get_device_name(),
        "shape": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
            "mode": args.mode,
        },
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_report(payload)
    return 0


def _native_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, repeats: int) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=True,
        enable_gqa=repeats > 1,
    )


def _attention_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
    repeats: int,
    mode: str,
    attention_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    output = attention_fn(query, key, value, repeats=repeats)
    if mode == "forward":
        return output
    grads = torch.autograd.grad(output, (query, key, value), grad_output, retain_graph=False, create_graph=False)
    return grads[0]


def _benchmark_row(name: str, fn: Callable[[], torch.Tensor], args: argparse.Namespace) -> dict[str, float | str]:
    ms = _cuda_event_time_ms(fn, warmup=args.warmup, iterations=args.iterations)
    return {
        "name": name,
        "latency_ms": ms,
        "tflops": _causal_attention_tflops(
            ms,
            batch_size=args.batch_size,
            query_heads=args.query_heads,
            sequence_length=args.sequence_length,
            head_dim=args.head_dim,
        ),
    }


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


def _causal_attention_tflops(
    latency_ms: float,
    *,
    batch_size: int,
    query_heads: int,
    sequence_length: int,
    head_dim: int,
) -> float:
    causal_pairs = sequence_length * (sequence_length + 1) / 2
    # QK^T and PV each perform one multiply-add per causal pair and head dimension.
    flops = batch_size * query_heads * 4 * causal_pairs * head_dim
    return flops / (latency_ms / 1000.0) / 1e12


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
        f"q_heads={shape['query_heads']} kv_heads={shape['kv_heads']} "
        f"head_dim={shape['head_dim']} dtype={shape['dtype']}"
    )
    print("")
    print(f"{'kernel':<20} {'latency_ms':>12} {'tflops':>12}")
    for row in payload["results"]:
        print(f"{row['name']:<20} {row['latency_ms']:>12.4f} {row['tflops']:>12.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
