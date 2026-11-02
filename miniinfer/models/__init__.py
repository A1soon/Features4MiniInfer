"""Model package exports."""

from .base import BaseModelOutput
import importlib


def __getattr__(name):
  if name not in __all__:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  if name.startswith("Qwen3Moe"):
    module = "fused_qwen3_moe"
  elif name.startswith("Qwen3"):
    module = "fused_qwen3"
  else:
    module = "fused_qwen2"
  value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
  globals()[name] = value
  return value

__all__ = [
  "BaseModelOutput",
  "Qwen2Attention",
  "Qwen2ForCausalLM",
  "Qwen2MLP",
  "Qwen2Model",
  "Qwen2TransformerBlock",
  "Qwen3Attention",
  "Qwen3ForCausalLM",
  "Qwen3MLP",
  "Qwen3Model",
  "Qwen3TransformerBlock",
  "Qwen3MoeDecoderLayer",
  "Qwen3MoeForCausalLM",
  "Qwen3MoeModel",
  "Qwen3MoeSparseMoeBlock",
]
