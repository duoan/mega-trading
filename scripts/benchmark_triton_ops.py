"""Benchmark local Triton transformer operator kernels.

Defaults match configs/rtx.yaml for the non-attention operators:
RMSNorm [8, 512, 1024], RoPE q=[8, 16, 512, 64] k=[8, 4, 512, 64],
and SwiGLU gate [8, 512, 2816], all in bf16.
"""

from __future__ import annotations

import argparse
import json
from typing import Callable

import torch
import torch.nn.functional as F

from mega_trading.kernels.triton_ops import triton_apply_rope, triton_rms_norm, triton_swiglu_gate


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Triton transformer operators for Mega-Trading shapes.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--intermediate-dim", type=int, default=2816)
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--mode", choices=("forward", "forward-backward"), default="forward")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--no-validate", action="store_true", help="skip comparisons against PyTorch operators")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark Triton operator kernels")

    dtype = _dtype(args.dtype)
    torch.manual_seed(31)
    requires_grad = args.mode == "forward-backward"
    rms_hidden = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.hidden_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad,
    )
    rms_weight = torch.randn(args.hidden_dim, device="cuda", dtype=dtype, requires_grad=requires_grad)
    query = torch.randn(
        args.batch_size,
        args.query_heads,
        args.sequence_length,
        args.head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad,
    )
    key = torch.randn(
        args.batch_size,
        args.kv_heads,
        args.sequence_length,
        args.head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad,
    )
    cos, sin = _rope_tables(args.sequence_length, args.head_dim, dtype)
    gate = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.intermediate_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad,
    )
    up = torch.randn_like(gate, requires_grad=requires_grad)
    rms_grad = torch.randn_like(rms_hidden)
    query_grad = torch.randn_like(query)
    key_grad = torch.randn_like(key)
    gate_grad = torch.randn_like(gate)

    context = torch.enable_grad() if requires_grad else torch.inference_mode()
    with context:
        if not args.no_validate:
            _validate_rms_norm(rms_hidden, rms_weight, args.mode)
            _validate_rope(query, key, cos, sin, args.mode)
            _validate_swiglu(gate, up, args.mode)

        results = [
            _benchmark_row(
                "triton_rms_norm",
                lambda: _step(
                    triton_rms_norm(rms_hidden, rms_weight, eps=1e-5),
                    (rms_hidden, rms_weight),
                    args.mode,
                    rms_grad,
                ),
                args,
            ),
            _benchmark_row(
                "torch_rms_norm",
                lambda: _step(
                    F.rms_norm(rms_hidden, (args.hidden_dim,), rms_weight, eps=1e-5),
                    (rms_hidden, rms_weight),
                    args.mode,
                    rms_grad,
                ),
                args,
            ),
            _benchmark_row(
                "triton_rope",
                lambda: _step(triton_apply_rope(query, key, cos, sin), (query, key), args.mode, (query_grad, key_grad)),
                args,
            ),
            _benchmark_row(
                "torch_rope",
                lambda: _step(_native_rope(query, key, cos, sin), (query, key), args.mode, (query_grad, key_grad)),
                args,
            ),
            _benchmark_row(
                "triton_swiglu_gate",
                lambda: _step(triton_swiglu_gate(gate, up), (gate, up), args.mode, gate_grad),
                args,
            ),
            _benchmark_row(
                "torch_swiglu_gate",
                lambda: _step(F.silu(gate) * up, (gate, up), args.mode, gate_grad),
                args,
            ),
        ]

    payload = {
        "device": torch.cuda.get_device_name(),
        "shape": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "hidden_dim": args.hidden_dim,
            "intermediate_dim": args.intermediate_dim,
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


def _validate_rms_norm(hidden: torch.Tensor, weight: torch.Tensor, mode: str) -> None:
    actual = triton_rms_norm(hidden, weight, eps=1e-5)
    expected = F.rms_norm(hidden, (hidden.shape[-1],), weight, eps=1e-5)
    _assert_close("rms_norm", actual, expected)
    if mode == "forward-backward":
        grad_output = torch.randn_like(hidden)
        _assert_grads_close("rms_norm", actual, expected, (hidden, weight), grad_output)


def _validate_rope(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mode: str) -> None:
    actual = triton_apply_rope(query, key, cos, sin)
    expected = _native_rope(query, key, cos, sin)
    _assert_close("rope query", actual[0], expected[0])
    _assert_close("rope key", actual[1], expected[1])
    if mode == "forward-backward":
        grad_outputs = (torch.randn_like(query), torch.randn_like(key))
        _assert_grads_close("rope", actual, expected, (query, key), grad_outputs)


def _validate_swiglu(gate: torch.Tensor, up: torch.Tensor, mode: str) -> None:
    actual = triton_swiglu_gate(gate, up)
    expected = F.silu(gate) * up
    _assert_close("swiglu_gate", actual, expected)
    if mode == "forward-backward":
        grad_output = torch.randn_like(gate)
        _assert_grads_close("swiglu_gate", actual, expected, (gate, up), grad_output)


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    tolerance = 7e-2 if actual.dtype in {torch.float16, torch.bfloat16} else 1e-4
    if not torch.allclose(actual, expected, atol=tolerance, rtol=tolerance):
        max_diff = (actual - expected).abs().max().item()
        raise RuntimeError(f"Triton {name} diverged from PyTorch; max_diff={max_diff:.6f}")


def _assert_grads_close(
    name: str,
    actual: torch.Tensor | tuple[torch.Tensor, ...],
    expected: torch.Tensor | tuple[torch.Tensor, ...],
    inputs: tuple[torch.Tensor, ...],
    grad_outputs: torch.Tensor | tuple[torch.Tensor, ...],
) -> None:
    actual_grads = torch.autograd.grad(actual, inputs, grad_outputs, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, inputs, grad_outputs, retain_graph=True)
    for index, (actual_grad, expected_grad) in enumerate(zip(actual_grads, expected_grads, strict=True)):
        _assert_close(f"{name} grad[{index}]", actual_grad, expected_grad)


def _step(
    output: torch.Tensor | tuple[torch.Tensor, ...],
    inputs: tuple[torch.Tensor, ...],
    mode: str,
    grad_outputs: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if isinstance(output, tuple):
        output_tensor = output[0]
    else:
        output_tensor = output
    if mode == "forward":
        return output_tensor
    grads = torch.autograd.grad(output, inputs, grad_outputs, retain_graph=False, create_graph=False)
    return grads[0]


def _benchmark_row(name: str, fn: Callable[[], torch.Tensor], args: argparse.Namespace) -> dict[str, float | str]:
    return {"name": name, "latency_ms": _cuda_event_time_ms(fn, warmup=args.warmup, iterations=args.iterations)}


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


def _native_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _rotate(query, cos, sin), _rotate(key, cos, sin)


def _rotate(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even = value[..., 0::2]
    odd = value[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(start_dim=-2)


def _rope_tables(sequence_length: int, head_dim: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(sequence_length, device="cuda", dtype=torch.float32)
    inv_freq = 1.0 / (500_000.0 ** (torch.arange(0, head_dim, 2, device="cuda", dtype=torch.float32) / head_dim))
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().to(dtype).view(1, 1, sequence_length, head_dim // 2)
    sin = freqs.sin().to(dtype).view(1, 1, sequence_length, head_dim // 2)
    return cos, sin


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
        f"hidden_dim={shape['hidden_dim']} intermediate_dim={shape['intermediate_dim']} "
        f"q_heads={shape['query_heads']} kv_heads={shape['kv_heads']} "
        f"head_dim={shape['head_dim']} dtype={shape['dtype']} mode={shape['mode']}"
    )
    print("")
    print(f"{'operator':<20} {'latency_ms':>12}")
    for row in payload["results"]:
        print(f"{row['name']:<20} {row['latency_ms']:>12.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
