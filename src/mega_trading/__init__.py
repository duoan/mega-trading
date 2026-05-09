"""Mega-Trading package."""

from mega_trading.config import BuildConfig, TrainConfig
from mega_trading.events import BuildResult, EventBuilder
from mega_trading.model import TradingModel
from mega_trading.trainer import Trainer

__version__ = "0.1.0"

__all__ = [
    "BuildConfig",
    "BuildResult",
    "EventBuilder",
    "TradingModel",
    "TrainConfig",
    "Trainer",
    "__version__",
]
