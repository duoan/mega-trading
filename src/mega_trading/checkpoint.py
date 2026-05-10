"""Checkpoint helpers shared by training and evaluation paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


_COMPILED_MODULE_PREFIX = "_orig_mod."


def normalized_model_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Return a model state dict loadable by the uncompiled model definition."""
    keys = tuple(state_dict.keys())
    if keys and all(isinstance(key, str) and key.startswith(_COMPILED_MODULE_PREFIX) for key in keys):
        return {key[len(_COMPILED_MODULE_PREFIX) :]: value for key, value in state_dict.items()}
    return dict(state_dict)


def load_checkpoint_model_state(model: torch.nn.Module, checkpoint: Mapping[str, Any]) -> None:
    model.load_state_dict(normalized_model_state_dict(checkpoint["model_state_dict"]))
