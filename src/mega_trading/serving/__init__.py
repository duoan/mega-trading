"""Serving-compatible local inference interfaces."""

from mega_trading.serving.server import FeatureRequest, ModelServer, PredictionResponse

__all__ = ["FeatureRequest", "ModelServer", "PredictionResponse"]
