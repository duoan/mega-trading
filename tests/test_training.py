import importlib.util
import inspect
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn.functional as F

from mega_trading.cli import load_config, main
from mega_trading.config import BuildConfig, TrainConfig
from mega_trading.backtest import run_backtest
from mega_trading.core.store import LocalObjectStore
from mega_trading.dataset import NumpyTickerTimeDataset, tokens_to_example
from mega_trading.eval import run_eval
from mega_trading.events import EventBuilder, events_by_ticker_from_rows
from mega_trading.kernels import triton_attention as triton_attention_module
from mega_trading.kernels import triton_ops as triton_ops_module
from mega_trading.model import LlamaAttention, RMSNorm, RotaryEmbedding, SwiGLU, TradingModel, _apply_rope, _triton_attention
from mega_trading.report import _training_chart, run_report
from mega_trading.tokenizer import MarketEventTokenizer
from mega_trading.trainer import (
    MuonAdamW,
    Trainer,
    _accelerator,
    _attention_kernel_context,
    _build_lr_scheduler,
    _build_optimizer,
    _current_learning_rate,
    _maybe_compile_model,
    _maybe_cudagraph_mark_step_begin,
)


class TrainingTests(unittest.TestCase):
    def test_named_training_configs_parse_for_local_and_modal_runs(self) -> None:
        binance_local = load_config(Path("configs"), "binance-local", [])
        binance_modal_prep = load_config(Path("configs"), "binance-modal-prep", [])
        server = load_config(Path("configs"), "server-rtx6000", [])
        binance_modal = load_config(Path("configs"), "modal-binance", [])

        self.assertEqual(str(binance_local.data.source), "binance_trades")
        self.assertEqual(int(binance_local.build.numpy_partition_rows), 8192)
        self.assertEqual(str(binance_modal_prep.data.data_dir), ".mega-trading/binance-modal")
        self.assertEqual(str(binance_modal_prep.data.mixture), "binance_public")
        self.assertEqual(int(binance_modal_prep.build.numpy_partition_rows), 16384)
        self.assertEqual(float(binance_modal_prep.build.validation_fraction), 0.02)
        self.assertEqual(float(binance_modal_prep.build.backtest_fraction), 0.10)
        self.assertTrue(bool(binance_modal_prep.build.streaming_prepare))
        self.assertEqual(str(server.data.data_dir), ".mega-trading/binance-modal")
        self.assertEqual(str(server.training.device), "cuda")
        self.assertEqual(str(server.training.distributed_strategy), "ddp")
        self.assertEqual(int(server.model.hidden_dim), 1024)
        self.assertEqual(int(server.model.layers), 20)
        self.assertEqual(int(server.training.batch_size), 8)
        self.assertEqual(int(server.training.gradient_accumulation_steps), 8)
        self.assertEqual(int(server.training.max_eval_batches), 64)
        self.assertEqual(str(server.training.compile_mode), "default")
        self.assertEqual(str(server.training.optimizer), "muon")
        self.assertEqual(str(server.training.lr_schedule), "cosine")
        self.assertEqual(int(server.training.lr_warmup_steps), 1000)
        self.assertEqual(float(server.training.min_learning_rate), 0.00002)
        self.assertEqual(str(binance_modal.data.data_dir), "/data/binance-trades")
        self.assertEqual(str(binance_modal.data.mixture), "binance_public")
        self.assertEqual(str(binance_modal.training.compile_mode), "reduce-overhead")
        self.assertEqual(str(binance_modal.training.distributed_strategy), "fsdp")
        self.assertEqual(int(binance_modal.build.numpy_partition_rows), 16384)
        self.assertTrue(bool(server.build.streaming_prepare))

    def test_train_config_validates_distributed_runtime_options(self) -> None:
        self.assertEqual(TrainConfig(run_id="triton", attention_backend="triton").attention_backend, "triton")
        with self.assertRaisesRegex(ValueError, "distributed_strategy"):
            TrainConfig(run_id="bad", distributed_strategy="deepspeed")
        with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
            TrainConfig(run_id="bad", gradient_accumulation_steps=0)
        with self.assertRaisesRegex(ValueError, "compile_mode"):
            TrainConfig(run_id="bad", compile_mode="fastest")
        with self.assertRaisesRegex(ValueError, "attention_backend"):
            TrainConfig(run_id="bad", attention_backend="xformers")
        with self.assertRaisesRegex(ValueError, "checkpoint_interval"):
            TrainConfig(run_id="bad", checkpoint_interval=0)
        with self.assertRaisesRegex(ValueError, "max_eval_batches"):
            TrainConfig(run_id="bad", max_eval_batches=0)

    def test_train_config_validates_optimizer_and_scheduler_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "optimizer"):
            TrainConfig(run_id="bad", optimizer="lion")
        with self.assertRaisesRegex(ValueError, "weight_decay"):
            TrainConfig(run_id="bad", weight_decay=-0.1)
        with self.assertRaisesRegex(ValueError, "adam_beta1"):
            TrainConfig(run_id="bad", adam_beta1=1.0)
        with self.assertRaisesRegex(ValueError, "muon_momentum"):
            TrainConfig(run_id="bad", muon_momentum=1.0)
        with self.assertRaisesRegex(ValueError, "muon_ns_steps"):
            TrainConfig(run_id="bad", muon_ns_steps=0)
        with self.assertRaisesRegex(ValueError, "lr_schedule"):
            TrainConfig(run_id="bad", lr_schedule="linear")
        with self.assertRaisesRegex(ValueError, "lr_warmup_steps"):
            TrainConfig(run_id="bad", lr_warmup_steps=-1)
        with self.assertRaisesRegex(ValueError, "min_learning_rate"):
            TrainConfig(run_id="bad", learning_rate=0.001, min_learning_rate=0.002)

    def test_build_config_validates_prepared_split_fractions(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation_fraction"):
            BuildConfig(validation_fraction=-0.1)
        with self.assertRaisesRegex(ValueError, "backtest_fraction"):
            BuildConfig(backtest_fraction=1.0)
        with self.assertRaisesRegex(ValueError, "validation_fraction \\+ backtest_fraction"):
            BuildConfig(validation_fraction=0.5, backtest_fraction=0.5)

    def test_event_builder_writes_token_shards_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalObjectStore(root)

            result = EventBuilder(
                store,
                BuildConfig(block_size=8, stride=4, min_events_per_ticker=5),
            ).build_events(_order_flow_events())
            profile = json.loads(root.joinpath(result.profile_path).read_text(encoding="utf-8"))
            tokenizer = json.loads(root.joinpath(result.tokenizer_path).read_text(encoding="utf-8"))
            tokens = np.load(root / result.numpy_tokens_path, mmap_mode="r")
            ticker_ids = np.load(root / result.numpy_ticker_ids_path, mmap_mode="r")
            numpy_metadata = json.loads(root.joinpath(result.numpy_metadata_path).read_text(encoding="utf-8"))

            self.assertEqual(profile["stream_contract"], "paper-order-flow-token-v1")
            self.assertEqual(profile["feature_order"], ["action", "side", "relative_price", "price_depth", "size", "time"])
            self.assertEqual(profile["event_size"], 1)
            self.assertEqual(tokenizer["type"], "paper-order-flow-composite")
            self.assertGreater(profile["event_count"], 10)
            self.assertGreater(profile["sequence_count"], 1)
            self.assertEqual(tokens.shape, (profile["sequence_count"], profile["block_size"] + 1))
            self.assertEqual(ticker_ids.shape, (profile["sequence_count"],))
            self.assertEqual(profile["numpy_dataset"]["format"], "mega-trading-numpy-token-v1")
            self.assertEqual(profile["numpy_dataset"]["splits"]["method"], "per_ticker_time")
            self.assertGreater(profile["numpy_dataset"]["splits"]["totals"]["backtest"], 0)
            self.assertIn("start_time", profile["numpy_dataset"]["splits"]["time_ranges"]["AAPL"]["backtest"])
            self.assertEqual(numpy_metadata["tokens_path"], result.numpy_tokens_path)
            self.assertIn("AAPL", numpy_metadata["ticker_to_id"])

    def test_tokenizer_is_deterministic_and_model_forward_shapes(self) -> None:
        events = [
            {
                "side": "buy" if index % 2 else "sell",
                "action": "add" if index % 3 else "delete",
                "relative_price_bps": float(index - 5),
                "price_depth_bps": float(index + 1),
                "size": float(100 + index),
                "interarrival_seconds": 1.0,
            }
            for index in range(10)
        ]
        tokenizer = MarketEventTokenizer.fit(events, relative_price_bins=4, price_bins=4, size_bins=4, time_bins=2)
        event = {
            "action": "add",
            "side": "buy",
            "relative_price_bps": 4.2,
            "price_depth_bps": 4.2,
            "size": 128.0,
            "interarrival_seconds": 1.0,
        }
        tokens = tokenizer.encode_event(event)
        model = TradingModel(vocab_size=tokenizer.vocab_size, block_size=8, hidden_dim=16, layers=1, attention_heads=2)
        logits = model(torch.tensor([tokens[:8]], dtype=torch.long))

        self.assertEqual(tokens, tokenizer.encode_event(event))
        self.assertEqual(len(tokens), 1)
        self.assertIsNotNone(tokenizer.price_depth_value(tokens[0]))
        self.assertIsNotNone(tokenizer.relative_price_value(tokens[0]))
        self.assertEqual(logits.shape, (1, len(tokens[:8]), tokenizer.vocab_size))

    def test_model_uses_llama_style_decoder_blocks(self) -> None:
        tokenizer = MarketEventTokenizer()
        model = TradingModel(
            vocab_size=tokenizer.vocab_size,
            block_size=8,
            hidden_dim=16,
            layers=1,
            attention_heads=4,
            kv_heads=2,
            intermediate_dim=32,
        )
        block = model.blocks[0]

        self.assertIsInstance(model.norm, RMSNorm)
        self.assertFalse(hasattr(model, "position_embedding"))
        self.assertIsInstance(block.attention, LlamaAttention)
        self.assertEqual(block.attention.kv_heads, 2)
        self.assertIsInstance(block.feed_forward, SwiGLU)
        self.assertEqual(model(torch.randint(0, tokenizer.vocab_size, (2, 8))).shape, (2, 8, tokenizer.vocab_size))

    def test_tied_output_initialization_keeps_initial_next_token_loss_reasonable(self) -> None:
        torch.manual_seed(7)
        model = TradingModel(vocab_size=512, block_size=16, hidden_dim=512, layers=1, attention_heads=8, dropout=0.0)
        input_ids = torch.randint(0, 512, (4, 16))
        labels = torch.randint(0, 512, (4, 16))

        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

        self.assertIs(model.output.weight, model.token_embedding.weight)
        self.assertLess(float(model.token_embedding.weight.detach().std()), 0.05)
        self.assertLess(float(loss.detach()), 20.0)

    def test_training_chart_aligns_validation_points_to_metric_steps(self) -> None:
        html = _training_chart(
            [
                {"step": 1, "train_loss": 10.0},
                {"step": 100, "train_loss": 9.0, "validation_loss": 8.0},
                {"step": 200, "train_loss": 7.0, "validation_loss": 6.0},
            ]
        )

        match = re.search(r'stroke="#73a7ff"[^>]*points="([^"]+)"', html)

        self.assertIsNotNone(match)
        first_validation_x = float(match.group(1).split()[0].split(",")[0])
        self.assertGreater(first_validation_x, 300.0)

    def test_rotary_embedding_precomputes_and_slices_positions(self) -> None:
        rope = RotaryEmbedding(head_dim=8, max_sequence_length=16, theta=10_000.0)
        query = torch.randn(2, 4, 6, 8)
        key = torch.randn(2, 2, 6, 8)

        rotated_query, rotated_key = rope(query, key)
        expected_query, expected_key = _apply_rope(query, key, theta=10_000.0)

        self.assertEqual(rope.cos.shape, (1, 1, 16, 4))
        self.assertEqual(rotated_query.shape, query.shape)
        self.assertEqual(rotated_key.shape, key.shape)
        self.assertTrue(torch.allclose(rotated_query, expected_query))
        self.assertTrue(torch.allclose(rotated_key, expected_key))

    def test_grouped_query_attention_uses_native_sdpa_gqa(self) -> None:
        attention = LlamaAttention(
            hidden_dim=16,
            block_size=8,
            attention_heads=4,
            kv_heads=2,
            dropout=0.0,
            rope_theta=10_000.0,
        )
        hidden = torch.randn(1, 8, 16)
        attended = torch.zeros(1, 4, 8, 4)

        with patch("mega_trading.model.F.scaled_dot_product_attention", return_value=attended) as sdpa:
            attention(hidden)

        _query, key, value = sdpa.call_args.args[:3]
        self.assertEqual(key.shape[1], 2)
        self.assertEqual(value.shape[1], 2)
        self.assertTrue(sdpa.call_args.kwargs["enable_gqa"])

    def test_triton_attention_backend_calls_local_kernel(self) -> None:
        attention = LlamaAttention(
            hidden_dim=16,
            block_size=8,
            attention_heads=4,
            kv_heads=2,
            dropout=0.0,
            rope_theta=10_000.0,
            attention_backend="triton",
        )
        hidden = torch.randn(1, 8, 16)
        attended = torch.zeros(1, 4, 8, 4)

        with patch("mega_trading.model._triton_attention", return_value=attended) as triton_attention:
            output = attention(hidden)

        _query, key, value = triton_attention.call_args.args[:3]
        self.assertEqual(output.shape, hidden.shape)
        self.assertEqual(key.shape[1], 2)
        self.assertEqual(value.shape[1], 2)
        self.assertEqual(triton_attention.call_args.kwargs["repeats"], 2)

    def test_triton_backend_routes_local_operator_kernels(self) -> None:
        hidden = torch.randn(1, 8, 16)
        with patch("mega_trading.model._triton_rms_norm", return_value=hidden) as rms_norm:
            norm = RMSNorm(16, operator_backend="triton")
            self.assertEqual(norm(hidden).shape, hidden.shape)
        rms_norm.assert_called_once()

        query = torch.randn(1, 4, 8, 4)
        key = torch.randn(1, 2, 8, 4)
        with patch("mega_trading.model._triton_apply_rope", return_value=(query, key)) as rope_kernel:
            rope = RotaryEmbedding(head_dim=4, max_sequence_length=8, theta=10_000.0, operator_backend="triton")
            self.assertEqual(rope(query, key)[0].shape, query.shape)
        rope_kernel.assert_called_once()

        with patch("mega_trading.model._triton_swiglu_gate", side_effect=lambda gate, up: torch.zeros_like(gate)) as swiglu_gate:
            swiglu = SwiGLU(hidden_dim=16, intermediate_dim=32, dropout=0.0, operator_backend="triton")
            self.assertEqual(swiglu(hidden).shape, hidden.shape)
        swiglu_gate.assert_called_once()

    def test_triton_attention_kernel_lives_in_kernel_module(self) -> None:
        self.assertEqual(_triton_attention.__module__, "mega_trading.kernels.triton_attention")
        self.assertIn("tl.make_tensor_descriptor", inspect.getsource(triton_attention_module))

    def test_triton_operator_kernels_live_in_kernel_module(self) -> None:
        self.assertEqual(triton_ops_module.triton_rms_norm.__module__, "mega_trading.kernels.triton_ops")
        self.assertEqual(triton_ops_module.triton_apply_rope.__module__, "mega_trading.kernels.triton_ops")
        self.assertEqual(triton_ops_module.triton_swiglu_gate.__module__, "mega_trading.kernels.triton_ops")

    def test_triton_attention_benchmark_defaults_match_server_rtx6000_shape(self) -> None:
        benchmark = _load_script("benchmark_triton_attention.py")
        args = benchmark._parse_args([])

        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.sequence_length, 512)
        self.assertEqual(args.query_heads, 16)
        self.assertEqual(args.kv_heads, 4)
        self.assertEqual(args.head_dim, 64)
        self.assertEqual(args.dtype, "bfloat16")
        self.assertEqual(args.mode, "forward")
        self.assertEqual(triton_attention_module._attention_tile_shape(args.sequence_length, args.head_dim), (64, 64))

    def test_triton_ops_benchmark_defaults_match_server_rtx6000_shape(self) -> None:
        benchmark = _load_script("benchmark_triton_ops.py")
        args = benchmark._parse_args([])

        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.sequence_length, 512)
        self.assertEqual(args.hidden_dim, 1024)
        self.assertEqual(args.intermediate_dim, 2816)
        self.assertEqual(args.query_heads, 16)
        self.assertEqual(args.kv_heads, 4)
        self.assertEqual(args.head_dim, 64)
        self.assertEqual(args.dtype, "bfloat16")
        self.assertEqual(args.mode, "forward")

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("triton") is not None,
        "Triton attention parity requires CUDA and triton",
    )
    def test_triton_attention_matches_torch_causal_gqa(self) -> None:
        torch.manual_seed(11)
        query = torch.randn(1, 4, 17, 32, device="cuda", dtype=torch.float16, requires_grad=True)
        key = torch.randn(1, 2, 17, 32, device="cuda", dtype=torch.float16, requires_grad=True)
        value = torch.randn(1, 2, 17, 32, device="cuda", dtype=torch.float16, requires_grad=True)
        expected_query = query.detach().clone().requires_grad_(True)
        expected_key = key.detach().clone().requires_grad_(True)
        expected_value = value.detach().clone().requires_grad_(True)
        grad_output = torch.randn_like(query)

        actual = _triton_attention(query, key, value, repeats=2)
        expected = F.scaled_dot_product_attention(
            expected_query,
            expected_key,
            expected_value,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
        )
        actual.backward(grad_output)
        expected.backward(grad_output)

        self.assertTrue(torch.allclose(actual, expected, atol=5e-2, rtol=5e-2))
        self.assertTrue(torch.allclose(query.grad, expected_query.grad, atol=6e-2, rtol=6e-2))
        self.assertTrue(torch.allclose(key.grad, expected_key.grad, atol=6e-2, rtol=6e-2))
        self.assertTrue(torch.allclose(value.grad, expected_value.grad, atol=6e-2, rtol=6e-2))

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("triton") is not None,
        "server-shape Triton attention parity requires CUDA and triton",
    )
    def test_server_shape_triton_attention_matches_torch_causal_gqa(self) -> None:
        torch.manual_seed(13)
        query = torch.randn(1, 16, 512, 64, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(1, 4, 512, 64, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key)

        actual = _triton_attention(query, key, value, repeats=4)
        expected = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
        )

        self.assertTrue(torch.allclose(actual, expected, atol=6e-2, rtol=6e-2))

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("triton") is not None,
        "server-shape Triton attention backward parity requires CUDA and triton",
    )
    def test_server_shape_triton_attention_backward_matches_torch_causal_gqa(self) -> None:
        torch.manual_seed(17)
        query = torch.randn(1, 16, 512, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        key = torch.randn(1, 4, 512, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        value = torch.randn_like(key, requires_grad=True)
        expected_query = query.detach().clone().requires_grad_(True)
        expected_key = key.detach().clone().requires_grad_(True)
        expected_value = value.detach().clone().requires_grad_(True)
        grad_output = torch.randn_like(query)

        actual = _triton_attention(query, key, value, repeats=4)
        expected = F.scaled_dot_product_attention(
            expected_query,
            expected_key,
            expected_value,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
        )
        actual.backward(grad_output)
        expected.backward(grad_output)

        self.assertTrue(torch.allclose(query.grad, expected_query.grad, atol=8e-2, rtol=8e-2))
        self.assertTrue(torch.allclose(key.grad, expected_key.grad, atol=8e-2, rtol=8e-2))
        self.assertTrue(torch.allclose(value.grad, expected_value.grad, atol=8e-2, rtol=8e-2))

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("triton") is not None,
        "Triton RMSNorm parity requires CUDA and triton",
    )
    def test_triton_rms_norm_matches_torch(self) -> None:
        torch.manual_seed(19)
        hidden = torch.randn(2, 512, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        expected_hidden = hidden.detach().clone().requires_grad_(True)
        expected_weight = weight.detach().clone().requires_grad_(True)
        grad_output = torch.randn_like(hidden)

        actual = triton_ops_module.triton_rms_norm(hidden, weight, eps=1e-5)
        expected = F.rms_norm(expected_hidden, (expected_hidden.shape[-1],), expected_weight, eps=1e-5)
        actual.backward(grad_output)
        expected.backward(grad_output)

        self.assertTrue(torch.allclose(actual, expected, atol=6e-2, rtol=6e-2))
        self.assertTrue(torch.allclose(hidden.grad, expected_hidden.grad, atol=8e-2, rtol=8e-2))
        self.assertTrue(torch.allclose(weight.grad, expected_weight.grad, atol=8e-2, rtol=8e-2))

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("triton") is not None,
        "Triton RoPE parity requires CUDA and triton",
    )
    def test_triton_rope_matches_torch(self) -> None:
        torch.manual_seed(23)
        rope = RotaryEmbedding(head_dim=64, max_sequence_length=512, theta=500_000.0)
        query = torch.randn(1, 16, 512, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        key = torch.randn(1, 4, 512, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        expected_query = query.detach().clone().requires_grad_(True)
        expected_key = key.detach().clone().requires_grad_(True)
        grad_query = torch.randn_like(query)
        grad_key = torch.randn_like(key)
        cos = rope.cos.to(device="cuda", dtype=torch.bfloat16)
        sin = rope.sin.to(device="cuda", dtype=torch.bfloat16)

        actual_query, actual_key = triton_ops_module.triton_apply_rope(query, key, cos, sin)
        expected_query_out = _apply_rope(expected_query, expected_key, theta=500_000.0)[0]
        expected_key_out = _apply_rope(expected_query, expected_key, theta=500_000.0)[1]
        torch.autograd.backward((actual_query, actual_key), (grad_query, grad_key))
        torch.autograd.backward((expected_query_out, expected_key_out), (grad_query, grad_key))

        self.assertTrue(torch.allclose(actual_query, expected_query_out, atol=6e-2, rtol=6e-2))
        self.assertTrue(torch.allclose(actual_key, expected_key_out, atol=6e-2, rtol=6e-2))
        self.assertTrue(torch.allclose(query.grad, expected_query.grad, atol=8e-2, rtol=8e-2))
        self.assertTrue(torch.allclose(key.grad, expected_key.grad, atol=8e-2, rtol=8e-2))

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("triton") is not None,
        "Triton SwiGLU parity requires CUDA and triton",
    )
    def test_triton_swiglu_gate_matches_torch(self) -> None:
        torch.manual_seed(29)
        gate = torch.randn(2, 512, 2816, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        up = torch.randn_like(gate, requires_grad=True)
        expected_gate = gate.detach().clone().requires_grad_(True)
        expected_up = up.detach().clone().requires_grad_(True)
        grad_output = torch.randn_like(gate)

        actual = triton_ops_module.triton_swiglu_gate(gate, up)
        expected = F.silu(expected_gate) * expected_up
        actual.backward(grad_output)
        expected.backward(grad_output)

        self.assertTrue(torch.allclose(actual, expected, atol=6e-2, rtol=6e-2))
        self.assertTrue(torch.allclose(gate.grad, expected_gate.grad, atol=8e-2, rtol=8e-2))
        self.assertTrue(torch.allclose(up.grad, expected_up.grad, atol=8e-2, rtol=8e-2))

    def test_compile_and_attention_backend_helpers_are_config_driven(self) -> None:
        model = TradingModel(vocab_size=MarketEventTokenizer().vocab_size, block_size=8, hidden_dim=16, layers=1, attention_heads=2)
        config = TrainConfig(run_id="compile-test", compile=True, compile_mode="reduce-overhead")

        with patch("mega_trading.trainer.torch.compile", return_value="compiled") as compile_model:
            self.assertEqual(_maybe_compile_model(model, config, torch.device("cpu")), "compiled")
            compile_model.assert_called_once_with(model, mode="reduce-overhead")
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            with _attention_kernel_context("flash", torch.device("cpu")):
                pass

    def test_mixed_precision_uses_bfloat16_on_cuda(self) -> None:
        config = TrainConfig(run_id="precision-test", mlflow_enabled=False)

        with patch("mega_trading.trainer.Accelerator", return_value="accelerator") as accelerator:
            self.assertEqual(_accelerator(torch.device("cuda"), "mixed", config), "accelerator")

        self.assertEqual(accelerator.call_args.kwargs["mixed_precision"], "bf16")

    def test_cudagraph_mark_step_begin_skipped_without_cuda_or_compile(self) -> None:
        mark = Mock()
        with patch.object(torch.compiler, "cudagraph_mark_step_begin", mark, create=True):
            _maybe_cudagraph_mark_step_begin(TrainConfig(run_id="x", compile=False), torch.device("cuda"))
            _maybe_cudagraph_mark_step_begin(TrainConfig(run_id="x", compile=True), torch.device("cpu"))
        mark.assert_not_called()

    def test_cudagraph_mark_step_begin_called_when_compile_and_cuda(self) -> None:
        mark = Mock()
        with patch.object(torch.compiler, "cudagraph_mark_step_begin", mark, create=True):
            _maybe_cudagraph_mark_step_begin(TrainConfig(run_id="x", compile=True), torch.device("cuda"))
        mark.assert_called_once()

    def test_muon_optimizer_partitions_hidden_matrices_from_adamw_params(self) -> None:
        model = TradingModel(vocab_size=MarketEventTokenizer().vocab_size, block_size=8, hidden_dim=16, layers=1, attention_heads=2)
        optimizer = _build_optimizer(model, TrainConfig(run_id="muon-test", optimizer="muon"))

        self.assertIsInstance(optimizer, MuonAdamW)
        muon_names = set(optimizer.param_groups[0]["param_names"])
        adamw_names = set(optimizer.param_groups[1]["param_names"])
        self.assertIn("blocks.0.attention.q_proj.weight", muon_names)
        self.assertIn("blocks.0.feed_forward.w1.weight", muon_names)
        self.assertIn("token_embedding.weight", adamw_names)
        self.assertIn("norm.weight", adamw_names)

    def test_cosine_scheduler_warms_up_then_decays_to_min_lr(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=1.0)
        config = TrainConfig(
            run_id="schedule-test",
            learning_rate=1.0,
            max_steps=4,
            lr_schedule="cosine",
            lr_warmup_steps=2,
            min_learning_rate=0.1,
        )
        scheduler = _build_lr_scheduler(optimizer, config)

        lrs = [_current_learning_rate(optimizer)]
        for _ in range(3):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            lrs.append(_current_learning_rate(optimizer))

        self.assertEqual(lrs, [0.5, 1.0, 1.0, 0.1])

    def test_dataset_row_maps_to_next_token_example(self) -> None:
        example = tokens_to_example([1, 3, 4, 5])

        self.assertEqual(example["input_ids"].tolist(), [1, 3, 4])
        self.assertEqual(example["labels"].tolist(), [3, 4, 5])

    def test_numpy_dataset_maps_mmap_rows_to_next_token_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalObjectStore(root)
            result = EventBuilder(
                store,
                BuildConfig(block_size=8, stride=4, min_events_per_ticker=5),
            ).build_events(_order_flow_events())
            profile = store.read_json(result.profile_path)
            dataset = NumpyTickerTimeDataset(
                store,
                dict(profile["numpy_dataset"]),
                dict(profile["numpy_dataset"]["splits"]["counts"]),
                split="train",
            )
            example = next(iter(dataset))

            self.assertEqual(example["input_ids"].shape[0], 8)
            self.assertEqual(example["labels"].shape[0], 8)

    def test_partitioned_numpy_dataset_maps_rows_to_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalObjectStore(root)
            result = EventBuilder(
                store,
                BuildConfig(block_size=8, stride=4, min_events_per_ticker=5, numpy_partition_rows=2),
            ).build_events(_order_flow_events())
            profile = store.read_json(result.profile_path)
            numpy_metadata = dict(profile["numpy_dataset"])
            dataset = NumpyTickerTimeDataset(
                store,
                numpy_metadata,
                dict(numpy_metadata["splits"]["counts"]),
                split="train",
            )
            example = next(iter(dataset))

            self.assertTrue(numpy_metadata["partitioned"])
            self.assertGreater(len(numpy_metadata["partitions"]), 1)
            self.assertEqual(example["input_ids"].shape[0], 8)

    def test_numpy_dataset_exposes_chronological_backtest_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalObjectStore(root)
            result = EventBuilder(
                store,
                BuildConfig(block_size=8, stride=4, min_events_per_ticker=5, validation_fraction=0.25, backtest_fraction=0.25),
            ).build_events(_order_flow_events())
            profile = store.read_json(result.profile_path)
            numpy_metadata = dict(profile["numpy_dataset"])
            split_counts = dict(numpy_metadata["splits"]["counts"])
            train_rows = list(NumpyTickerTimeDataset(store, numpy_metadata, split_counts, split="train"))
            validation_rows = list(NumpyTickerTimeDataset(store, numpy_metadata, split_counts, split="validation"))
            backtest_rows = list(NumpyTickerTimeDataset(store, numpy_metadata, split_counts, split="backtest"))

            self.assertGreater(len(train_rows), 0)
            self.assertGreater(len(validation_rows), 0)
            self.assertGreater(len(backtest_rows), 0)
            self.assertEqual(
                len(train_rows) + len(validation_rows) + len(backtest_rows),
                int(profile["sequence_count"]),
            )

    def test_trainer_and_eval_smoke_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalObjectStore(root)
            EventBuilder(
                store,
                BuildConfig(block_size=8, stride=4, min_events_per_ticker=5),
            ).build_events(_order_flow_events())

            train_result = Trainer(
                store,
                TrainConfig(
                    run_id="train-test",
                    max_steps=2,
                    batch_size=2,
                    validation_fraction=0.25,
                    eval_interval=1,
                    hidden_dim=16,
                    layers=1,
                    attention_heads=2,
                    device="cpu",
                    mlflow_enabled=False,
                    progress_bar=False,
                    checkpoint_interval=1,
                    max_eval_batches=1,
                ),
            ).train("datasets/mixture=public/tokens.npy")
            checkpoint_path = root / train_result.checkpoint_path
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint["model_state_dict"] = {
                f"_orig_mod.{key}": value for key, value in checkpoint["model_state_dict"].items()
            }
            torch.save(checkpoint, checkpoint_path)

            eval_result = run_eval(store, "train-test", rollouts=2, generated_tokens=8, device="cpu")
            backtest_result = run_backtest(store, "train-test", rollouts=2, generated_tokens=8, max_batches=1, device="cpu")
            report_result = run_report(store, "train-test")

            metrics = json.loads(root.joinpath(train_result.metrics_path).read_text(encoding="utf-8"))["metrics"]
            manifest = store.read_manifest(train_result.manifest_path)
            checkpoint = torch.load(root / train_result.checkpoint_path, map_location="cpu", weights_only=False)
            report = json.loads(root.joinpath(eval_result.report_path).read_text(encoding="utf-8"))
            backtest_report = json.loads(root.joinpath(backtest_result.report_path).read_text(encoding="utf-8"))
            self.assertTrue(root.joinpath(train_result.checkpoint_path).exists())
            self.assertTrue(all(key.startswith("_orig_mod.") for key in checkpoint["model_state_dict"]))
            self.assertTrue(root.joinpath("runs/train-test/checkpoints/step-000001.pt").exists())
            self.assertIn("validation_loss", metrics[-1])
            self.assertEqual(metrics[-1]["validation_batches"], 1.0)
            self.assertGreater(metrics[-1]["backtest_sequence_count"], 0)
            self.assertEqual(metrics[-1]["distributed_strategy"], "ddp")
            self.assertEqual(metrics[-1]["dataset_format"], "numpy")
            self.assertEqual(metrics[-1]["world_size"], 1)
            self.assertFalse(metrics[-1]["compile_enabled"])
            self.assertEqual(metrics[-1]["optimizer"], "adamw")
            self.assertEqual(metrics[-1]["lr_schedule"], "constant")
            self.assertIn("learning_rate", metrics[-1])
            self.assertEqual(manifest.metadata["distributed_strategy"], "ddp")
            self.assertEqual(manifest.metadata["dataset_format"], "numpy")
            self.assertEqual(manifest.metadata["gradient_accumulation_steps"], 1)
            self.assertEqual(manifest.metadata["attention_backend"], "auto")
            self.assertEqual(manifest.metadata["max_eval_batches"], 1)
            self.assertEqual(manifest.metadata["optimizer"], "adamw")
            self.assertEqual(manifest.metadata["lr_schedule"], "constant")
            self.assertGreater(manifest.metadata["backtest_sequence_count"], 0)
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertIn("scheduler_state_dict", checkpoint)
            self.assertEqual(checkpoint["step"], 2)
            self.assertEqual(report["stage"], "eval")
            self.assertIn("generated", report)
            self.assertEqual(backtest_report["stage"], "backtest")
            self.assertGreater(backtest_report["backtest_tokens"], 0)
            report_html = root.joinpath(report_result.report_path).read_text(encoding="utf-8")
            self.assertIn("Backtest Dashboard", report_html)
            self.assertIn("Ticker Candles + Forecast", report_html)

    def test_trainer_resumes_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalObjectStore(root)
            EventBuilder(
                store,
                BuildConfig(block_size=8, stride=4, min_events_per_ticker=5),
            ).build_events(_order_flow_events())

            first = Trainer(
                store,
                TrainConfig(
                    run_id="resume-test",
                    max_steps=1,
                    batch_size=2,
                    validation_fraction=0.25,
                    eval_interval=1,
                    hidden_dim=16,
                    layers=1,
                    attention_heads=2,
                    device="cpu",
                    mlflow_enabled=False,
                    progress_bar=False,
                ),
            ).train("datasets/mixture=public/tokens.npy")
            resumed = Trainer(
                store,
                TrainConfig(
                    run_id="resume-test",
                    max_steps=2,
                    batch_size=2,
                    validation_fraction=0.25,
                    eval_interval=1,
                    hidden_dim=16,
                    layers=1,
                    attention_heads=2,
                    device="cpu",
                    mlflow_enabled=False,
                    progress_bar=False,
                    resume_from_checkpoint=first.checkpoint_path,
                ),
            ).train("datasets/mixture=public/tokens.npy")

            metrics = json.loads(root.joinpath(resumed.metrics_path).read_text(encoding="utf-8"))["metrics"]
            checkpoint = torch.load(root / resumed.checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual([row["step"] for row in metrics], [1, 2])
            self.assertEqual(checkpoint["step"], 2)

    def test_cli_build_train_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ingest_config = root / "ingest-demo.toml"
            ingest_config.write_text(
                f"""
[ingest]
output_dir = "{root}"

[[ingest.sources]]
name = "fixture"
""".strip()
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                main(
                    [
                        "prepare",
                        "--ingest-config",
                        str(ingest_config),
                        "build.block_size=8",
                        "build.stride=4",
                        "build.min_events_per_ticker=5",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "train",
                        f"data.data_dir={root}",
                        "run.run_id=train-cli",
                        "training.max_steps=1",
                        "training.batch_size=2",
                        "training.validation_fraction=0.25",
                        "training.eval_interval=1",
                        "training.device=cpu",
                        "training.mlflow_enabled=false",
                        "training.progress_bar=false",
                        "training.gradient_accumulation_steps=1",
                        "training.attention_backend=auto",
                        "model.hidden_dim=16",
                        "model.layers=1",
                        "model.attention_heads=2",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "eval",
                        "--run-id",
                        "train-cli",
                        f"data.data_dir={root}",
                        "eval.rollouts=1",
                        "eval.generated_tokens=4",
                        "eval.device=cpu",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "backtest",
                        "--run-id",
                        "train-cli",
                        "--max-batches",
                        "1",
                        f"data.data_dir={root}",
                        "eval.rollouts=1",
                        "eval.generated_tokens=4",
                        "eval.device=cpu",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "report",
                        "--run-id",
                        "train-cli",
                        f"data.data_dir={root}",
                    ]
                ),
                0,
            )

            self.assertTrue((root / "datasets/mixture=public/tokens.npy").exists())
            self.assertTrue((root / "runs/train-cli/checkpoint.pt").exists())
            self.assertTrue((root / "evals/train-cli/report.json").exists())
            self.assertTrue((root / "evals/train-cli/backtest.json").exists())
            self.assertTrue((root / "reports/train-cli/backtest.html").exists())


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _order_flow_events() -> dict[str, list[dict[str, object]]]:
    rows = []
    for ticker_index, ticker in enumerate(("AAPL", "MSFT")):
        for index in range(20):
            rows.append(
                {
                    "event_id": f"event-{ticker}-{index:04d}",
                    "ticker": ticker,
                    "timestamp": f"2024-01-02T14:{30 + index:02d}:00Z",
                    "date": "2024-01-02",
                    "action": "add" if index % 3 else "delete",
                    "side": "buy" if (index + ticker_index) % 2 == 0 else "sell",
                    "midprice": 100.0 + ticker_index * 10.0 + index * 0.01,
                    "relative_price_bps": float((1 if (index + ticker_index) % 2 == 0 else -1) * (1 + index % 8)),
                    "price_depth_bps": float(1 + (index % 8) * 2 + ticker_index),
                    "size": float(1.0 + index * 0.1 + ticker_index * 0.05),
                    "interarrival_seconds": 60.0,
                    "provider": "fixture",
                    "source_ids": [f"fixture:{ticker}:{index:04d}"],
                    "midprice_return_bps": float((1 if index % 2 else -1) * (index % 5)),
                }
            )
    return events_by_ticker_from_rows(rows)


if __name__ == "__main__":
    unittest.main()
