"""Engine package exports."""

import importlib

_MODULES = {
  "cuda_graph_runner": "CapturedGraph CudaGraphRunner",
  "detokenizer": "ChatTemplateHandler DecodeState IncrementalDecoder",
  "expert_parallel": "BatchPacket ExpertParallelExecutor RequestMirror RequestState",
  "future_map": "FutureIndices FutureMap",
  "incremental_batch_tokenizer": "IncrementalBatchTokenizer",
  "llm_engine": "LLMEngine RequestOutput StepOutput",
  "model_runner": "ModelRunner",
  "overlap_executor": "OverlapBatchRecord OverlapExecutor",
}


def __getattr__(name):
  for module, names in _MODULES.items():
    if name in names.split():
      value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
      globals()[name] = value
      return value
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
  "CapturedGraph",
  "ChatTemplateHandler",
  "CudaGraphRunner",
  "DecodeState",
  "BatchPacket",
  "ExpertParallelExecutor",
  "FutureIndices",
  "FutureMap",
  "IncrementalBatchTokenizer",
  "IncrementalDecoder",
  "LLMEngine",
  "ModelRunner",
  "OverlapBatchRecord",
  "OverlapExecutor",
  "RequestOutput",
  "StepOutput",
  "RequestMirror",
  "RequestState",
]
