"""
LLMEngine - 对外暴露的统一接口

运行模式:
1. 单进程模式 (默认): 所有组件在同一进程内运行
2. 多进程模式: 预留接口（当前未实现）

Usage:
    # 单进程模式 (默认)
    engine = LLMEngine(model_path="...")
    outputs = engine.generate(prompts, sampling_params)

    # 多进程模式 (当前未实现，会抛 NotImplementedError)
    # engine = LLMEngine(model_path="...", use_multiprocess=True)
"""

import atexit
import os
from miniinfer.utils import get_logger
import queue
from dataclasses import dataclass, field, fields
from time import perf_counter
from typing import List, Optional, Union, Dict, Any, Generator, Sequence
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch

from miniinfer.config.engine.config import EngineConfig
from miniinfer.utils.memory_utils import calc_max_total_tokens
from miniinfer.utils.profiler_utils import profile_methods, stage
from .model_runner import ModelRunner
from .overlap_executor import OverlapExecutor
from .incremental_batch_tokenizer import IncrementalBatchTokenizer
from .metrics import MetricsRecorder
from miniinfer.scheduler.scheduler import Scheduler
from miniinfer.scheduler.scheduler_batch import (
  Req,
  ScheduledBatch,
  ForwardBatch,
  BatchResult,
)

from ..utils.sampling_params import SamplingParams
from ..utils.crash_logger import log_inference_crash
from .detokenizer import IncrementalDecoder

logger = get_logger(__name__)


@dataclass
class RequestOutput:
  """单个请求的输出结果"""

  request_id: int
  delta_text: str
  full_text: str
  token_id: int
  output_token_ids: List[int]
  finished: bool
  finish_reason: Optional[str] = None


@dataclass
class StepOutput:
  """单步推理的输出"""

  outputs: List[RequestOutput]
  num_prefill_tokens: int = 0
  num_decode_tokens: int = 0
  # When this step runs a prefill/extend batch, expose per-request cache stats.
  prefill_prefix_lens: Dict[int, int] = field(default_factory=dict)
  prefill_extend_lens: Dict[int, int] = field(default_factory=dict)

  @property
  def has_output(self) -> bool:
    return len(self.outputs) > 0


@profile_methods("LLMEngine")
class LLMEngine:
  """
  LLM 推理引擎

  当前仅支持单进程模式。
  多进程模式接口保留，但暂未实现（use_multiprocess=True 会抛异常）。
  """

  def __init__(
    self,
    model: str,
    use_multiprocess: bool = False,
    **kwargs,
  ):
    """
    初始化 LLM Engine

    Python 语法提示:
    - __init__ 是“构造函数”。当你写 LLMEngine(...) 时，Python 会自动调用它。
    - self 表示“当前这个 LLMEngine 对象本身”。给 self.xxx 赋值，就是把状态保存
      到这个 engine 对象里，后面的 generate()/step() 都能继续使用。
    - **kwargs 表示接收任意额外关键字参数。例如 LLMEngine(model, max_num_seqs=4)
      里的 max_num_seqs 会进入 kwargs 这个 dict。

    推理系统视角:
    这个函数只负责“搭好 engine 的外壳和基础配置”。真正加载模型、初始化
    Scheduler/KVCache/Tokenizer/ModelRunner 的工作在 _init_single_process_mode() 中。

    Args:
        model: 模型路径或 HuggingFace model id
        use_multiprocess: 是否使用多进程模式
        **kwargs: 其他配置参数
    """
    # EngineConfig 是 engine 的核心配置类，比如 max_num_seqs、max_model_len、
    # dtype、attention_backend、page_size 等都属于这里。
    #
    # 用户可能会传入一些“不属于 EngineConfig 的参数”，例如 use_chat_template、
    # debug、vocab_mask_fn。直接把所有 kwargs 都交给 EngineConfig 可能会报错。
    # 所以这里先读取 EngineConfig 声明过的字段名，只保留它认识的参数。
    config_fields = {field.name for field in fields(EngineConfig)}
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}

    # 创建 EngineConfig 对象。后续几乎所有组件都会从 self.config 读取配置：
    # - ModelRunner 用它决定加载什么模型、使用什么 dtype。
    # - KVCacheManager 用它决定 page_size、最大上下文长度等。
    # - Scheduler 用它决定最大 batch size、chunked prefill 参数等。
    self.config = EngineConfig(model, **config_kwargs)

    # Prompt formatting (chat template) config.
    # - True: always use tokenizer.apply_chat_template when available
    # - False: never use chat template (raw text completion)
    # - "auto": enable for likely chat/instruct models
    # chat template 决定 prompt 在送入 tokenizer 之前是否要包装成聊天格式。
    # 例如 instruct/chat 模型通常希望看到 user/assistant/system 这样的结构；
    # base model 则更像普通文本补全，可以直接使用原始 prompt。
    # 这里使用字典的 .get(key, default_value) 方法。相比于直接用
    # kwargs["use_chat_template"]，.get 的好处是：如果字典里没有这个 key，
    # 它会返回默认值，而不会抛出 KeyError 异常导致程序崩溃。
    self.use_chat_template = kwargs.get("use_chat_template", "auto")
    self.system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")

    # debug_logit_topk 用于调试每步 logits 中概率最高的候选 token。
    # logits 可以先粗略理解为“模型对词表中每个 token 的打分”。
    self.debug = bool(kwargs.get("debug", True))
    self.debug_logit_topk = int(kwargs.get("debug_logit_topk", 5))

    # Optional callback to generate per-batch vocab masks for grammar-constrained sampling.
    # Signature: fn(forward_batch: ForwardBatch) -> Optional[torch.Tensor]
    # vocab_mask_fn 是一个可选钩子，用来限制采样时允许输出哪些 token。
    # 例如做 JSON/grammar constrained decoding 时，可以把不合法 token mask 掉。
    self.vocab_mask_fn = kwargs.get("vocab_mask_fn", None)

    # Metrics are opt-in so ordinary inference keeps the existing overhead and
    # behavior.  The recorder itself is torch/CUDA independent; only the
    # lifecycle hooks below observe the engine runtime.
    self.metrics = MetricsRecorder(enabled=bool(kwargs.get("enable_metrics", False)))

    # 保存运行模式。use_multiprocess=False 是最适合学习的单进程路径：
    # Scheduler、ModelRunner、Detokenizer 都在同一个 Python 进程中。
    self.use_multiprocess = use_multiprocess
    if use_multiprocess and self.config.expert_parallel_size > 1:
      raise ValueError("EP is not supported inside the tokenizer/scheduler ZMQ pipeline")
    self.ep_executor = None

    # _started 表示 engine 生命周期状态。generate()/stream_generate() 会检查它，
    # 如果还没启动，会自动调用 start()。
    self._started = False

    # 流式模式下每步 detokenize 产出 delta_text；非流式（generate）跳过每步 detok
    # （避免 O(seq_len) × num_steps 的开销，长序列下占 CPU ~60%），改为结束时对
    # 每个请求 decode 一次全量 token。
    #
    # detokenize 是把 token ids 转回字符串。流式输出必须每步转一点文本；
    # 非流式 generate 只需要最后转一次，所以可以省很多 CPU 时间。
    self._streaming = False

    if use_multiprocess:
      # 多进程模式：本对象仅作 client 侧壳，推理在三个 worker 进程中完成。
      # 委托公共 API 给 MultiProcessEngine；__enter__/__exit__ 仍走 self.start/stop。
      #
      # Python 语法提示:
      # 这里的 self.start = self._mp_engine.start 是“方法转发/替换”。
      # 之后用户调用 self.start()，实际执行的是 MultiProcessEngine.start()。
      from miniinfer.engine.multi_process_engine import MultiProcessEngine

      self._mp_engine = MultiProcessEngine(model, engine_kwargs=config_kwargs)
      self.start = self._mp_engine.start
      self.stop = self._mp_engine.stop
      self.generate = self._mp_engine.generate
      self.stream_generate = self._mp_engine.stream_generate

      # atexit.register 表示 Python 程序退出时自动调用 self.stop，
      # 尽量避免进程结束时后台资源没有释放。
      atexit.register(self.stop)
      return

    # 单进程初始化主路径。这里会真正创建:
    # - ModelRunner: 加载模型权重并执行 forward。
    # - KVCacheManager: 分配和管理 KV cache 显存。
    # - Scheduler: 管理 waiting/running 请求并组织 batch。
    # - Tokenizer/Detokenizer: 文本和 token ids 互转。
    try:
      self._init_single_process_mode()
    except BaseException:
      if self.ep_executor is not None:
        self.ep_executor.close(abort=True)
      raise

    # request_id -> Req 的索引表。Req 是调度器中的请求对象，里面保存
    # prompt token、已生成 token、采样参数、完成状态、KV cache 位置等。
    self._req_by_id: Dict[int, Req] = {}

    # 注册退出清理。即使用户忘记手动 stop()，进程结束时也尽量清理 engine。
    atexit.register(self.stop)

  # ======================== private helper function ========================
  def _debug_print_topk_next_tokens(
    self, logits: torch.Tensor, forward_batch: ForwardBatch
  ) -> None:
    topk = self.debug_logit_topk
    if topk <= 0:
      return
    if logits is None:
      return

    with torch.no_grad():
      if forward_batch.forward_mode.is_extend():
        if logits.size(0) == forward_batch.batch_size:
          logits_for_sampling = logits
        else:
          extend_lens = forward_batch.extend_seq_lens_cpu or []
          last_token_indices: List[int] = []
          cumsum = 0
          for length in extend_lens:
            last_token_indices.append(cumsum + int(length) - 1)
            cumsum += int(length)
          if not last_token_indices:
            return
          # 优化：使用 torch.as_tensor 避免 CUDA 同步
          idx_cpu = torch.as_tensor(last_token_indices, dtype=torch.int64)
          idx = idx_cpu.to(logits.device, non_blocking=True)
          logits_for_sampling = logits[idx]
      else:
        logits_for_sampling = logits

      k = min(int(topk), int(logits_for_sampling.shape[-1]))
      top_vals, top_ids = torch.topk(logits_for_sampling, k=k, dim=-1)

      for i, req in enumerate(forward_batch.all_seqs):
        candidates = []
        for val, tid in zip(top_vals[i].tolist(), top_ids[i].tolist()):
          text = self.tokenizer.decode([int(tid)], skip_special_tokens=False)
          candidates.append((int(tid), val, repr(text)))
        logger.debug("req=%s top%d next_token candidates: %s", req.req_id, k, candidates)

  # ======== prompt encoding (compat) ========
  def _encode_prompt(self, prompt: str) -> List[int]:
    """
    Encode a user prompt into token ids.

    Kept for backward-compat/tests; production code uses `self.prompt_tokenizer`.
    """
    prompt_tokenizer = getattr(self, "prompt_tokenizer", None)
    if prompt_tokenizer is None:
      prompt_tokenizer = IncrementalBatchTokenizer(
        self.tokenizer,
        use_chat_template=self.use_chat_template,
        model_name=str(getattr(self.config, "model", "")),
        system_prompt=self.system_prompt,
      )
    with stage("stage::Tokenizer.encode_one"):
      return prompt_tokenizer.encode_one(prompt)

  # ======== init helper function =========
  def _init_single_process_mode(self):
    """初始化单进程模式"""
    logger.info("Initializing LLMEngine in single-process mode")

    # TP 子进程（暂未启用）
    self.ps = []
    self.events = []
    # ======== engine config =============
    config = self.config

    if config.expert_parallel_size > 1:
      from .expert_parallel import ExpertParallelExecutor

      self.ep_executor = ExpertParallelExecutor(config)
      self.model_runner = self.ep_executor
      self.kv_cache_mgr = self.ep_executor.kv_cache_mgr
      self.max_total_tokens = self.ep_executor.max_total_tokens
    else:
      self._init_local_model_and_cache(config)

    self._init_scheduler_and_tokenizer(config)

  def _init_local_model_and_cache(self, config):
    from miniinfer.kvcache.kv_cache_manager import KVCacheManager

    # 步骤 1: 先创建 ModelRunner 并加载模型（不传入 kv_cache_mgr）
    logger.info("Step 1: Loading model...")
    with stage("stage::LLMEngine.init.model_runner"):
      self.model_runner = ModelRunner(
        self.config,
        kv_cache_mgr=None,
        rank=0,
        events=self.events,
        attn_backend=config.attention_backend,
      )

    # 步骤 2: 模型加载完成后，基于实际已分配显存计算 max_total_tokens
    logger.info("Step 2: Calculating max_total_tokens based on actual GPU memory usage...")
    with stage("stage::LLMEngine.init.calc_max_total_tokens"):
      self.max_total_tokens = calc_max_total_tokens(self.config, log=logger)
      if config.max_total_tokens is not None:
        self.max_total_tokens = min(self.max_total_tokens, config.max_total_tokens)
        self.max_total_tokens = self.max_total_tokens // config.page_size * config.page_size

    # 步骤 3: 初始化 KV Cache 管理器
    logger.info("Step 3: Initializing KV Cache Manager...")
    with stage("stage::LLMEngine.init.kv_cache_mgr"):
      self.kv_cache_mgr = KVCacheManager(
        size=self.max_total_tokens,
        max_requests=config.max_num_seqs,
        max_context_len=config.max_context_len,
        num_layers=config.hf_config.num_hidden_layers,
        num_heads=config.hf_config.num_key_value_heads,  # 使用 KV head 数量，支持 GQA
        head_dim=getattr(
          config.hf_config,
          "head_dim",
          config.hf_config.hidden_size // config.hf_config.num_attention_heads,
        ),
        dtype=config.dtype,
        device="cuda",
        enable_prefix_cache=config.enable_prefix_cache,
        page_size=config.page_size,
        max_extend_tokens=config.max_extend_len,
      )

    # 步骤 4: 设置 ModelRunner 的 kv_cache_mgr 并初始化 attn_backend
    logger.info("Step 4: Initializing attention backend...")
    with stage("stage::LLMEngine.init.set_kv_cache_mgr"):
      self.model_runner.set_kv_cache_mgr(self.kv_cache_mgr)

  def _init_scheduler_and_tokenizer(self, config):
    # Tokenizer
    with stage("stage::Tokenizer.from_pretrained"):
      self.tokenizer = AutoTokenizer.from_pretrained(
        self.config.model,
        use_fast=True,
        trust_remote_code=True,
      )
    with stage("stage::LLMEngine.init.prompt_tokenizer"):
      self.prompt_tokenizer = IncrementalBatchTokenizer(
        self.tokenizer,
        use_chat_template=self.use_chat_template,
        model_name=str(getattr(self.config, "model", "")),
        system_prompt=self.system_prompt,
      )
    # Stream detokenizer
    with stage("stage::LLMEngine.init.detokenizer"):
      self.detokenizer = IncrementalDecoder(
        tokenizer=self.tokenizer,
        skip_special_tokens=True,
      )
    self.config.eos = self.tokenizer.eos_token_id

    # Scheduler (传入 tokenizer 和 model_runner)
    with stage("stage::LLMEngine.init.scheduler"):
      self.scheduler = Scheduler(self.config, self.tokenizer, self.kv_cache_mgr)

    # 步骤 5: 初始化 CUDA Graph (如果未禁用)
    if not self.config.enforce_eager or self.config.enable_ep_cuda_graph:
      logger.info("Step 5: Initializing CUDA Graph for decode acceleration...")
      cuda_graph_max_bs = self.config.cuda_graph_max_bs
      if cuda_graph_max_bs <= 0:
        cuda_graph_max_bs = self.config.max_num_seqs
      with stage("stage::LLMEngine.init.cuda_graph"):
        self.model_runner.init_cuda_graph(
          max_batch_size=cuda_graph_max_bs,
          max_context_len=self.config.max_context_len,
        )
    else:
      logger.info("Step 5: CUDA Graph disabled (enforce_eager=True)")

    # 步骤 6: 初始化 Overlap Executor (双 batch 交替执行)
    enable_overlap = self.config.enable_overlap
    logger.info(f"Step 6: Initializing Overlap Executor (enabled={enable_overlap})...")
    with stage("stage::LLMEngine.init.overlap_executor"):
      self.overlap_executor = (
        None
        if self.ep_executor is not None
        else OverlapExecutor(
          max_running_requests=self.config.max_num_seqs,
          max_chunks_per_request=4,
          device="cuda",
          enable_overlap=enable_overlap,
        )
      )

    # 步骤 7: 预热 release path，避免第一次 drain_pending_releases() 触发
    # Runtime Triggered Module Loading。
    with stage("stage::LLMEngine.init.release_path_warmup"):
      self.kv_cache_mgr.warmup_release_path()

    # 步骤 8: 预热 schedule path，覆盖 prefill / decode / mixed 中首次触发的
    # CUDA/Triton 模块加载。
    with stage("stage::LLMEngine.init.schedule_path_warmup"):
      self.scheduler.warmup_schedule_path(device=torch.device("cuda"))

    # 追踪上一个 batch 用于 overlap 判断
    self._last_batch: Optional[ScheduledBatch] = None

    # 异步请求队列：支持在推理过程中动态添加请求
    # 线程安全，可从任意线程调用 submit_request()
    self._pending_requests: queue.Queue = queue.Queue()

    self._started = True

  # ======== add request and step function ========
  def add_requests(
    self,
    prompts: Sequence[Union[str, List[int]]],
    sampling_params: Sequence[SamplingParams],
  ) -> List[int]:
    """
    批量添加请求到 Scheduler。

    这是 generate() 主链路里的关键一步：
    - generate() 接收的是用户层面的 prompts，可以是字符串，也可以是 token ids。
    - Scheduler 不直接处理字符串，它处理的是 Req 对象。
    - Req 里保存 token ids、采样参数、生成状态、KV cache 相关位置等。

    所以 add_requests() 做的事可以概括为：
      prompts -> token_ids_list -> Req -> scheduler.waiting_queue

    Args:
        prompts:
            一批输入。每个元素可以是:
            - str: 原始文本，需要 tokenizer 编码。
            - List[int]: 已经编码好的 token ids，可以直接使用。
        sampling_params:
            和 prompts 一一对应的一批采样参数。generate() 里如果用户只传一个
            SamplingParams，会先扩展成等长列表再传进来。

    Returns:
        request_ids:
            每个新建 Req 的 request id。后续可以用这个 id 查询/跟踪请求。
    """
    # 空输入直接返回空列表。这样调用方不需要额外处理 len(prompts)==0 的情况。
    if not prompts:
      return []

    # 每个 prompt 必须有一套对应的 sampling params。
    # 如果数量不一致，后面 zip(prompts, sampling_params) 会悄悄截断较长的一边，
    # 导致请求丢失或参数错配，所以这里提前报错。
    if len(prompts) != len(sampling_params):
      raise ValueError(
        f"prompts and sampling_params length mismatch: {len(prompts)} vs {len(sampling_params)}"
      )

    # token_ids_list 最终会保存每个请求的 token ids。
    # 先创建等长的空列表，占位用；如果某个 prompt 已经是 List[int]，
    # 下面会直接填入；如果是 str，则先记录下来，稍后批量 tokenize。
    token_ids_list: List[List[int]] = [[] for _ in range(len(prompts))]

    # to_encode 保存所有需要 tokenizer 处理的字符串 prompt。
    # to_encode_indices 保存这些字符串在原 prompts 中的位置。
    # 为什么要记录 indices？
    # 因为批量 tokenizer 返回的是 encoded 列表，需要放回 token_ids_list
    # 的原始位置，保持 outputs 顺序和输入 prompts 顺序一致。
    to_encode: List[str] = []
    to_encode_indices: List[int] = []

    # enumerate(prompts) 会同时给出下标 i 和元素 prompt。
    # 这里把两类输入分开：
    # - str: 后面统一 batch tokenize，效率更高。
    # - List[int]: 用户已经 tokenize 好了，直接使用。
    for i, prompt in enumerate(prompts):
      if isinstance(prompt, str):
        to_encode.append(prompt)
        to_encode_indices.append(i)
      else:
        # list(prompt) 会复制一份 token ids，避免外部列表后续被修改影响 Req。
        token_ids_list[i] = list(prompt)

    # 对所有字符串 prompt 做批量编码。
    # batch tokenize 通常比一个个 encode_one 更高效，也更接近线上 serving 的用法。
    if to_encode:
      with stage("stage::Tokenizer.encode_batch"):
        encoded = self.prompt_tokenizer.encode_batch(to_encode)

      # zip(to_encode_indices, encoded) 会把“原始位置”和“编码结果”配对。
      # 例如 prompts = ["a", [1,2], "b"] 时：
      #   to_encode_indices = [0, 2]
      #   encoded = [ids_of_a, ids_of_b]
      # 所以要分别填回 token_ids_list[0] 和 token_ids_list[2]。
      for i, ids in zip(to_encode_indices, encoded):
        token_ids_list[i] = ids

    # 到这里，所有 prompt 都已经变成 token ids。
    # 接下来为每个请求创建 Req 对象，并加入 Scheduler。
    request_ids: List[int] = []
    for token_ids, param in zip(token_ids_list, sampling_params):
      # Req 是调度器里真正流动的请求对象。
      # 它不只包含 prompt token ids，还会在后续逐步记录：
      # - output_ids: 已生成 token。
      # - req_pool_idx/out_cache_loc: KV cache 位置。
      # - finished/finished_reason: 请求是否结束。
      req = Req(token_ids, param)

      # scheduler.add(req) 只是把请求放入 waiting_queue。
      # 真正的模型计算会在后续 step()/step_overlap() 里发生。
      self.scheduler.add(req)

      self.metrics.record_request_arrival(req.req_id, len(token_ids))

      # _req_by_id 是 engine 侧的索引表，方便通过 request_id 找回 Req。
      self._req_by_id[req.req_id] = req
      request_ids.append(req.req_id)

    return request_ids

  def submit_request(
    self,
    prompt: Union[str, List[int]],
    sampling_params: Optional[SamplingParams] = None,
  ) -> int:
    """
    异步提交请求（线程安全）

    请求会被加入待处理队列，在下一次 step_overlap() 调用时处理。
    可以从任意线程调用此方法，适用于 online serving 场景。

    Args:
        prompt: 输入 prompt（文本或 token ids）
        sampling_params: 采样参数

    Returns:
        request_id: 请求 ID
    """
    if sampling_params is None:
      sampling_params = SamplingParams()

    # 预先 tokenize（如果需要）
    if isinstance(prompt, str):
      token_ids = self.prompt_tokenizer.encode_one(prompt)
    else:
      token_ids = list(prompt)

    req = Req(token_ids, sampling_params)
    self.metrics.record_request_arrival(req.req_id, len(token_ids))
    self._req_by_id[req.req_id] = req
    self._pending_requests.put(req)
    return req.req_id

  def _recv_requests(self) -> int:
    """
    处理异步提交的请求（SGLang 风格 recv_requests）

    从 _pending_requests 队列中取出所有待处理请求，加入 scheduler。
    此方法在 step_overlap() 开头调用。

    Returns:
        处理的请求数量
    """
    count = 0
    while not self._pending_requests.empty():
      try:
        req = self._pending_requests.get_nowait()
        self.scheduler.add(req)
        count += 1
      except Exception:
        break
    return count

  def _is_disable_overlap_for_batch(self, batch: Optional[ScheduledBatch]) -> bool:
    """
    判断是否应该禁用当前 batch 的 overlap（SGLang 风格）

    禁用条件：
    1. overlap 未启用
    2. 连续两个 prefill/extend batch（优化 TTFT）
    3. 当前 batch 为空

    对于连续 prefill 禁用 overlap 的理由：
    - 假设 batch A 是 prefill，batch B 也是 prefill
    - 如果启用 overlap，process(A) 会在 forward(B) 期间执行
    - 但 batch A 的用户想尽快看到首个 token（TTFT）
    - 禁用 overlap 后，process(A) 在 forward(B) 之前完成
    - 这样 batch A 的首个 token 更早返回

    Args:
        batch: 当前要执行的 batch

    Returns:
        True 如果应该禁用 overlap
    """
    if not self.config.enable_overlap:
      return True

    if batch is None:
      return True

    # 连续两个 prefill/extend batch 禁用 overlap 以优化 TTFT
    if (
      self._last_batch is not None
      and self._last_batch.forward_mode.is_extend()
      and batch.forward_mode.is_extend()
    ):
      return True

    return False

  def _build_vocab_mask(self, forward_batch: ForwardBatch) -> Optional[torch.Tensor]:
    fn = getattr(self, "vocab_mask_fn", None)
    if fn is None:
      return None
    mask = fn(forward_batch)
    if mask is None:
      return None
    if not isinstance(mask, torch.Tensor):
      mask = torch.as_tensor(mask)
    if mask.dtype != torch.bool:
      mask = mask.to(torch.bool)
    return mask

  def step(self) -> StepOutput:
    """
    执行单步推理（非阻塞式）

    每次调用执行一个推理步骤：调度 -> forward -> sample -> 增量解码。
    新请求可以在任意时刻通过 add_request() 添加，不会阻塞此方法。

    初学者可以把一次 step 理解成“推理引擎的一拍”:
    1. Scheduler 选出本轮要跑的一批请求。
    2. ForwardBatch 把调度结果整理成模型能吃的 tensor。
    3. ModelRunner 跑模型前向，拿到 logits。
    4. Sampler 从 logits 中选出每个请求的下一个 token。
    5. _process_step_result 把 token 写回 Req，并判断请求是否结束。

    注意:
    - step() 只启动/完成一轮 batch，不保证所有请求都结束。
    - generate() 会在 while 循环里反复调用 step_overlap()/step()，直到结束。
    - 这个函数是不带 overlap 的清晰版本，适合先读；step_overlap() 是性能优化版。

    Returns:
        StepOutput: 包含本步所有请求的增量输出
    """
    iteration_started_at = perf_counter()
    ep = getattr(self, "ep_executor", None)
    if ep is not None:
      self._recv_requests()

    # 一些已完成请求的 KV cache 释放可能被延迟到后面做，以便和 GPU 计算重叠。
    # 但在新一轮 schedule 之前，prefix/radix cache 状态必须是最新的。
    # 典型场景：req1 刚结束，req2 立刻提交了相同 prompt。如果不先释放/回写，
    # req2 可能无法命中 req1 留下的 prefix cache。
    if getattr(self.scheduler, "pending_release_reqs", None):
      released_ids = [req.req_id for req in self.scheduler.pending_release_reqs]
      self.scheduler.drain_pending_releases()
      if ep is not None:
        ep.release(released_ids)

    # 1. 调度获取 batch。
    #
    # scheduler.schedule(...) 会查看:
    # - waiting_queue: 还没 prefill 的新请求。
    # - running_batch: 已经完成 prefill、正在 decode 的请求。
    # - chunking_reqs: 长 prompt 被拆成 chunk 后还没处理完的请求。
    #
    # 它返回 ScheduledBatch，描述“本轮要跑哪些请求、跑 prefill 还是 decode、
    # 输入 token 是哪些、KV cache 写到哪里”等元数据。
    try:
      with stage("stage::LLMEngine.step.schedule"):
        batch = self.scheduler.schedule(self.model_runner.device, skip_decode_input_ids=False)
    except Exception as exc:
      # 如果调度阶段异常，记录尽量多的上下文，方便复现 OOM/状态不一致等问题。
      log_inference_crash(
        exc, scheduler=self.scheduler, model_runner=self.model_runner, stage="schedule"
      )
      if ep is not None:
        ep.close(abort=True)
      raise
    if batch is None or len(batch.reqs) == 0:
      # 没有可执行请求，返回一个空 StepOutput。
      # 返回空并不一定是错误，可能只是队列暂时没有请求。
      # 同时再次 flush pending release，避免已经完成的请求占着 KV cache 不放。
      if getattr(self.scheduler, "pending_release_reqs", None):
        released_ids = [req.req_id for req in self.scheduler.pending_release_reqs]
        self.scheduler.drain_pending_releases()
        if ep is not None:
          ep.release(released_ids)
      return StepOutput(outputs=[])

    iteration = self.metrics.begin_iteration(
      batch,
      self.scheduler,
      started_at=iteration_started_at,
      scheduled_at=perf_counter(),
    )

    # 2. 构造 ForwardBatch 并执行模型前向。
    #
    # ScheduledBatch 是“调度器视角”的 batch，里面有很多 CPU/GPU 混合元数据。
    # ForwardBatch 是“模型视角”的 batch，会补齐:
    # - input_ids: 本轮送进模型的 token ids。
    # - positions: 每个 token 在序列中的位置，用于 RoPE。
    # - req_pool_indices/out_cache_loc: attention 写读 KV cache 需要的位置。
    # - attn_backend metadata: FlashAttention/FlashInfer 需要的索引信息。
    forward_batch = None
    try:
      with stage("stage::LLMEngine.step.forward_batch_init"):
        forward_batch = ForwardBatch.init_new(batch, self.model_runner.attn_backend)

      with stage("stage::LLMEngine.step.forward"):
        # model_runner.forward(...) 会根据 forward_mode 走两条路径之一：
        # - EXTEND/MIXED: 处理 prompt 或 chunked prefill，用 extend attention。
        # - DECODE: 每个请求只输入 1 个 token，用 decode attention/CUDA Graph。
        #
        # 返回的 logits 可以理解为“每个请求下一 token 的词表打分”。
        self.metrics.mark_forward_start(iteration)
        output = self.model_runner.forward(forward_batch)
        logits = output.logits
      # if self.debug_logit_topk > 0:
      # self._debug_print_topk_next_tokens(logits, forward_batch)

      # 3. 采样。
      #
      # logits 本身只是分数，还不是 token。sample() 会结合 SamplingParams：
      # temperature/top_k/top_p/greedy 等，从词表里选出 next_tokens。
      # next_tokens 通常是一维 tensor，长度约等于本轮参与生成的请求数。
      with stage("stage::LLMEngine.step.sample"):
        next_tokens = self.model_runner.sample(
          logits, forward_batch, vocab_mask=self._build_vocab_mask(forward_batch)
        )
      self.metrics.mark_forward_end(iteration)
    except Exception as exc:
      # forward/sample 涉及模型、attention backend、KV cache、CUDA Graph，
      # 是最容易出现 shape/device/dtype 问题的地方，所以这里把 batch 信息也记录下来。
      log_inference_crash(
        exc,
        scheduler=self.scheduler,
        model_runner=self.model_runner,
        forward_batch=forward_batch,
        scheduled_batch=batch,
        stage="forward/sample",
      )
      if ep is not None:
        ep.close(abort=True)
      raise

    # 4. 处理结果并增量解码。
    #
    # BatchResult 是一个小容器，把 logits 和 next_token_ids 包在一起传下去。
    # _process_step_result 会做几件事：
    # - 把 next token append 到对应 Req.output_ids。
    # - 检查 EOS/max_tokens，判断请求是否 finished。
    # - 流式模式下把新增 token detokenize 成 delta_text。
    # - 完成后通知 Scheduler 释放或回写 KV cache。
    with stage("stage::LLMEngine.step.process_result"):
      result = BatchResult(logits=logits, next_token_ids=next_tokens)
      executed_requests = list(batch.reqs)
      try:
        outputs = self._process_step_result(batch, result)
        if ep is not None:
          ep.commit(executed_requests)
      except BaseException:
        if ep is not None:
          ep.close(abort=True)
        raise

    self.metrics.finish_iteration(iteration, outputs, self.scheduler)

    # 5. 统计本轮处理了多少 token。
    #
    # 这些统计主要用于进度条、benchmark、调试:
    # - num_prefill_tokens: 本轮处理了多少 prompt/extend token。
    # - num_decode_tokens: 本轮为多少请求生成了下一个 token。
    #
    # mixed batch 比较特殊：它同时包含 prefill 请求和 decode 请求。
    # 对 decode 请求来说，即使它走的是 extend kernel，也应该记为 decode token。
    num_prefill = 0
    num_decode = 0
    prefill_prefix_lens: Dict[int, int] = {}
    prefill_extend_lens: Dict[int, int] = {}

    decode_req_set = set()
    if batch.decoding_reqs:
      decode_req_set = {id(r) for r in batch.decoding_reqs}

    if batch.forward_mode.is_extend():
      # is_extend() covers both EXTEND and MIXED modes
      if batch.prefix_lens and batch.extend_lens:
        for req, pre_len, ext_len in zip(batch.reqs, batch.prefix_lens, batch.extend_lens):
          if id(req) in decode_req_set:
            # decode 请求在 mixed batch 中视为 decode token
            num_decode += 1
          else:
            num_prefill += ext_len
            prefill_prefix_lens[int(req.req_id)] = int(pre_len)
            prefill_extend_lens[int(req.req_id)] = int(ext_len)
      else:
        num_prefill = sum(batch.extend_lens) if batch.extend_lens else 0
    elif batch.forward_mode.is_decode():
      num_decode = len(batch.reqs)

    return StepOutput(
      outputs=outputs,
      num_prefill_tokens=num_prefill,
      num_decode_tokens=num_decode,
      prefill_prefix_lens=prefill_prefix_lens,
      prefill_extend_lens=prefill_extend_lens,
    )

  def step_overlap(self) -> StepOutput:
    """
    双 Batch 交替执行的推理步骤（SGLang 风格）

    核心设计（参考 SGLang event_loop_overlap）:
      - Placeholder 路径: schedule(N) → run_async(N) → process(N-1)
                                            ↑               ↑
                                      GPU 先启动    与 forward(N) 并行

      - Conservative 路径: process(N-1) → schedule(N) → run_async(N)
                              ↑
                     schedule 需要正确的 output_ids

    SGLang 的关键优化：
      1. 先启动 GPU，然后在 GPU 执行时处理上一批结果
      2. 连续两个 prefill batch 禁用 overlap，优化 TTFT（首 token 延迟）
      3. recv_requests 在循环开头处理异步提交的新请求

    Conservative 路径用于：首批请求、near-max-tokens 边界、连续 prefill。

    Phases:
      0. recv_requests: 处理 submit_request() 异步提交的请求
      1. 判断是否可以使用 placeholder (can_use_placeholder)
      2. Conservative: 先处理 N-1 (如果需要)
      3. Schedule: 获取下一个 batch
      3.5. 检测连续 prefill，禁用 overlap 优化 TTFT
      4. run_async: 启动 GPU forward
      5. Placeholder: GPU 运行时处理 N-1

    Returns:
        StepOutput: 包含本步所有请求的增量输出
    """
    if getattr(self, "ep_executor", None) is not None:
      return self.step()

    outputs: List[RequestOutput] = []
    num_prefill = 0
    num_decode = 0
    prefill_prefix_lens: Dict[int, int] = {}
    prefill_extend_lens: Dict[int, int] = {}

    # ===================================================================
    # Phase 0: recv_requests — 处理异步提交的请求 (SGLang 风格)
    #
    # 从 _pending_requests 队列中取出所有待处理请求，加入 scheduler。
    # 这允许用户在推理过程中动态添加请求（通过 submit_request()）。
    # ===================================================================
    if getattr(self, "_pending_requests", None) is not None:
      with stage("stage::LLMEngine.step_overlap.recv_requests"):
        self._recv_requests()

    # ===================================================================
    # Phase 1: 判断路径
    #
    # can_use_placeholder 决定是否走 SGLang 风格 overlap：
    # - True: schedule 使用 placeholder input_ids，无需等待 N-1 处理完成
    # - False: schedule 需要正确的 output_ids，必须先处理 N-1
    # ===================================================================
    running_batch = self.scheduler.running_batch
    has_near_max_tokens_req = False
    if running_batch is not None and running_batch.reqs:
      for req in running_batch.reqs:
        if req.finished:
          continue
        max_tokens = int(getattr(req, "max_tokens", 0) or 0)
        if max_tokens > 0 and (len(getattr(req, "output_ids", [])) + 1) >= max_tokens:
          has_near_max_tokens_req = True
          break

    can_use_placeholder = (
      self.config.enable_overlap
      and not has_near_max_tokens_req
      and len(running_batch.reqs) > 0
      and running_batch.output_ids is not None
      # and len(running_batch.output_ids) == len(running_batch.reqs)
    )

    # ===================================================================
    # Phase 2: Conservative 路径 — 先处理 N-1
    #
    # 仅当无法使用 placeholder 时（首批/near-max-tokens），先同步处理上一批。
    # 这确保 schedule(N) 看到正确的 output_ids。
    # ===================================================================
    pending_step_out: Optional[StepOutput] = None

    if not can_use_placeholder and self.overlap_executor.has_pending():
      with stage("stage::LLMEngine.step_overlap.process_conservative"):
        try:
          pending_step_out = self.overlap_executor.process_pending_batch(
            self._process_overlap_result
          )
        except Exception as exc:
          log_inference_crash(
            exc,
            scheduler=self.scheduler,
            model_runner=self.model_runner,
            stage="step_overlap.process_conservative",
          )
          raise

    # ===================================================================
    # Phase 3: Schedule（在 schedule_stream 上执行 CPU→GPU 数据准备）
    #
    # 所有 schedule 和 prepare 操作在 schedule_stream 上执行，
    # forward_stream 会等待 schedule_stream 完成后再启动 GPU 计算。
    # ===================================================================
    schedule_stream = getattr(self.overlap_executor, "schedule_stream", None)
    # When CUDA graph is in use, schedule_stream's GPU prep (H2D copies, position
    # compute, allocations) for step N+1 must not race with forward_stream's graph
    # replay for step N. Make schedule_stream wait for the last forward_stream op
    # (the graph replay) before issuing its own GPU work. This is non-blocking on
    # the CPU (unlike forward_stream.synchronize()), preserving CPU post-processing
    # overlap with GPU forward.
    if schedule_stream is not None and self.model_runner.use_cuda_graph:
      schedule_stream.wait_stream(self.overlap_executor.forward_stream)
    try:
      if schedule_stream is not None:
        with torch.cuda.stream(schedule_stream):
          if can_use_placeholder:
            with stage("stage::LLMEngine.step_overlap.schedule_placeholder"):
              batch = self.scheduler.schedule(self.model_runner.device, skip_decode_input_ids=True)
          else:
            with stage("stage::LLMEngine.step_overlap.schedule"):
              batch = self.scheduler.schedule(self.model_runner.device, skip_decode_input_ids=False)
      else:
        if can_use_placeholder:
          with stage("stage::LLMEngine.step_overlap.schedule_placeholder"):
            batch = self.scheduler.schedule(self.model_runner.device, skip_decode_input_ids=True)
        else:
          with stage("stage::LLMEngine.step_overlap.schedule"):
            batch = self.scheduler.schedule(self.model_runner.device, skip_decode_input_ids=False)
    except Exception as exc:
      log_inference_crash(
        exc, scheduler=self.scheduler, model_runner=self.model_runner, stage="step_overlap.schedule"
      )
      raise

    # ===================================================================
    # Phase 3.5: 检测连续 prefill，决定是否禁用 overlap (SGLang 风格)
    #
    # 连续两个 prefill/extend batch 会禁用 overlap 以优化 TTFT：
    # - 第一个 prefill batch 的用户希望尽快看到首个 token
    # - 禁用 overlap 后，process(N-1) 在 forward(N) 之前完成
    # - 这样 batch N-1 的首个 token 更早返回给用户
    # ===================================================================
    disable_overlap_for_batch = self._is_disable_overlap_for_batch(batch)

    if disable_overlap_for_batch and self.overlap_executor.has_pending():
      # 连续 prefill 或其他禁用场景：立即处理上一批结果
      with stage("stage::LLMEngine.step_overlap.process_consecutive_prefill"):
        try:
          step_out = self.overlap_executor.process_pending_batch(self._process_overlap_result)
        except Exception as exc:
          log_inference_crash(
            exc,
            scheduler=self.scheduler,
            model_runner=self.model_runner,
            stage="step_overlap.process_consecutive_prefill",
          )
          raise
        if step_out:
          if pending_step_out is None:
            pending_step_out = step_out
          else:
            # 合并结果（不应该发生，但防御性处理）
            pending_step_out.outputs.extend(step_out.outputs)

    # ===================================================================
    # Phase 4: run_async — 尽快把 forward(N) 交给 GPU
    # ===================================================================
    # IMPORTANT: avoid GPU sync here (e.g. `(batch.input_ids < 0).any()`),
    # otherwise we will block the CPU scheduler on a CUDA reduction and
    # destroy overlap. Let the scheduler mark placeholder usage explicitly.
    use_placeholder = bool(can_use_placeholder and batch is not None and batch.uses_placeholder)

    current_record = None
    current_forward_batch = None

    if batch is not None and len(batch.reqs) > 0:
      # forward_batch_init 也在 schedule_stream 上执行（构建元数据）
      schedule_stream = getattr(self.overlap_executor, "schedule_stream", None)
      try:
        if schedule_stream is not None:
          with torch.cuda.stream(schedule_stream):
            with stage("stage::LLMEngine.step_overlap.forward_batch_init"):
              current_forward_batch = ForwardBatch.init_new(batch, self.model_runner.attn_backend)
        else:
          with stage("stage::LLMEngine.step_overlap.forward_batch_init"):
            current_forward_batch = ForwardBatch.init_new(batch, self.model_runner.attn_backend)

        with stage("stage::LLMEngine.step_overlap.run_forward_async"):
          current_record = self.overlap_executor.run_forward_async(
            batch,
            current_forward_batch,
            self.model_runner,
            use_placeholder=use_placeholder,
          )
      except Exception as exc:
        log_inference_crash(
          exc,
          scheduler=self.scheduler,
          model_runner=self.model_runner,
          forward_batch=current_forward_batch,
          scheduled_batch=batch,
          stage="step_overlap.forward_init/run_forward_async",
        )
        raise

      if not batch.forward_mode.is_decode():
        # Propagate placeholder metadata for the next schedule() call.
        # This is "scheduler-side" CUDA work; keep it on schedule_stream to avoid
        # accidental default-stream serialization with compute.
        schedule_stream = getattr(self.overlap_executor, "schedule_stream", None)
        if schedule_stream is not None:
          with torch.cuda.stream(schedule_stream):
            self._propagate_future_to_running_batch(batch)
        else:
          self._propagate_future_to_running_batch(batch)

    # ===================================================================
    # Phase 5: Placeholder 路径 — GPU 启动后处理 N-1
    #
    # 此时 GPU 正在执行 forward(N)+sample(N)（如果无 vocab_mask）
    # 或 forward(N)（如果有 vocab_mask）。
    # CPU 并行处理 N-1 的结果，实现真正的 overlap。
    #
    # 注意：如果 disable_overlap_for_batch=True，Phase 3.5 已经处理过，这里跳过
    # ===================================================================
    if (
      can_use_placeholder and not disable_overlap_for_batch and self.overlap_executor.has_pending()
    ):
      with stage("stage::LLMEngine.step_overlap.process_overlap"):
        try:
          pending_step_out = self.overlap_executor.process_pending_batch(
            self._process_overlap_result
          )
        except Exception as exc:
          log_inference_crash(
            exc,
            scheduler=self.scheduler,
            model_runner=self.model_runner,
            stage="step_overlap.process_overlap",
          )
          raise

    # 收集 pending 处理结果
    if pending_step_out is not None:
      outputs.extend(pending_step_out.outputs)
      num_prefill += pending_step_out.num_prefill_tokens
      num_decode += pending_step_out.num_decode_tokens
      prefill_prefix_lens.update(pending_step_out.prefill_prefix_lens)
      prefill_extend_lens.update(pending_step_out.prefill_extend_lens)

    # ===================================================================
    # Phase 4.5: vocab mask generate (CPU-heavy) + launch sample (GPU-heavy)
    #
    # Only executed when vocab_mask_fn is configured (grammar-constrained sampling).
    # This stage is intentionally after processing (N-1) so any per-request
    # grammar state updates can be reflected in the sampling mask for batch N.
    # ===================================================================
    if batch is not None and len(batch.reqs) > 0 and current_record is not None:
      vocab_mask = None
      with stage("stage::LLMEngine.step_overlap.vocab_mask_generate"):
        if current_forward_batch is not None:
          vocab_mask = self._build_vocab_mask(current_forward_batch)
          if vocab_mask is not None:
            # Transfer the mask on schedule_stream so forward_stream can wait on it.
            if getattr(self.overlap_executor, "schedule_stream", None) is not None:
              with torch.cuda.stream(self.overlap_executor.schedule_stream):
                vocab_mask = vocab_mask.to(self.model_runner.device, non_blocking=True)
            else:
              vocab_mask = vocab_mask.to(self.model_runner.device, non_blocking=True)

      with stage("stage::LLMEngine.step_overlap.launch_sample"):
        try:
          current_record = self.overlap_executor.run_sample_async(
            record=current_record,
            model_runner=self.model_runner,
            vocab_mask=vocab_mask,
          )
        except Exception as exc:
          log_inference_crash(
            exc,
            scheduler=self.scheduler,
            model_runner=self.model_runner,
            forward_batch=current_forward_batch,
            scheduled_batch=batch,
            stage="step_overlap.run_sample_async",
          )
          raise

    # Non-overlap mode: process immediately
    if batch is not None and current_record is not None:
      if not getattr(self.overlap_executor, "enable_overlap", True):
        try:
          step_out = self._process_overlap_result(current_record.batch, current_record.batch_result)
        except Exception as exc:
          log_inference_crash(
            exc,
            scheduler=self.scheduler,
            model_runner=self.model_runner,
            forward_batch=current_forward_batch,
            scheduled_batch=current_record.batch,
            stage="step_overlap.process_immediate",
          )
          raise
        if step_out:
          outputs.extend(step_out.outputs)
          num_prefill += step_out.num_prefill_tokens
          num_decode += step_out.num_decode_tokens
          prefill_prefix_lens.update(step_out.prefill_prefix_lens)
          prefill_extend_lens.update(step_out.prefill_extend_lens)

    # 确保 overlap 被禁用或无 batch 时不堆积 pending
    if (not self.config.enable_overlap or batch is None) and self.overlap_executor.has_pending():
      with stage("stage::LLMEngine.step_overlap.sync_pending"):
        while self.overlap_executor.has_pending():
          try:
            step_out = self.overlap_executor.process_pending_batch(self._process_overlap_result)
          except Exception as exc:
            log_inference_crash(
              exc,
              scheduler=self.scheduler,
              model_runner=self.model_runner,
              scheduled_batch=batch,
              stage="step_overlap.sync_pending",
            )
            raise
          if step_out:
            outputs.extend(step_out.outputs)
            num_prefill += step_out.num_prefill_tokens
            num_decode += step_out.num_decode_tokens
            prefill_prefix_lens.update(step_out.prefill_prefix_lens)
            prefill_extend_lens.update(step_out.prefill_extend_lens)

    # Drain deferred releases only after the current batch has already launched
    # its sample/copy work. Cache state must be fresh before the next schedule(),
    # but it does not need to block this step's launch path.
    if getattr(self.scheduler, "pending_release_reqs", None):
      with stage("stage::LLMEngine.step_overlap.drain_pending_releases"):
        self.scheduler.drain_pending_releases()

    self._last_batch = batch

    return StepOutput(
      outputs=outputs,
      num_prefill_tokens=num_prefill,
      num_decode_tokens=num_decode,
      prefill_prefix_lens=prefill_prefix_lens,
      prefill_extend_lens=prefill_extend_lens,
    )

  def generate(
    self,
    prompts: Union[List[str], List[List[int]]],
    sampling_params: Union[SamplingParams, List[SamplingParams]] = None,
    use_tqdm: bool = True,
  ) -> List[Dict[str, Any]]:
    """
    批量生成文本（阻塞式）

    等待所有请求完成后返回结果。
    如需流式输出，请使用 stream_generate()。

    初学者可以把 generate() 理解成“帮你写好的推理主循环”:
    1. 规范化 prompts 和 sampling_params。
    2. 调用 add_requests() 把 prompt 变成 Req 并加入 Scheduler。
    3. while not is_finished(): 反复驱动 step_overlap()。
    4. 收集每个请求的输出 token ids。
    5. 最后一次性 detokenize 成文本并返回。

    generate() 是阻塞式接口：调用它的线程会一直等到这一批请求全部完成。
    如果你想边生成边拿到 delta_text，应该看 stream_generate()。

    Args:
        prompts: 输入 prompts
        sampling_params: 采样参数
        use_tqdm: 是否显示进度条

    Returns:
        生成结果列表，每个元素包含 text 和 token_ids
    """
    # 如果用户创建了 engine 但没有显式调用 start()，这里自动启动。
    # 在 minimal_engine.py 里使用 with LLMEngine(...) as engine 时，__enter__()
    # 已经会调用 start()，所以这里通常只是兜底。
    if not self._started:
      self.start()

    # 1. 归一化输入。
    #
    # 为了让后续逻辑统一处理，这里把几种输入形式都转成“列表”：
    # - "hello"               -> ["hello"]
    # - [1, 2, 3]             -> [[1, 2, 3]]
    # - ["hello", "world"]    -> 保持不变
    # - [[1, 2], [3, 4]]      -> 保持不变
    #
    # prompts 既可以是字符串，也可以是已经 tokenize 好的 token id 列表。
    if isinstance(prompts, str):
      prompts = [prompts]
    elif isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], int):
      prompts = [prompts]

    # 2. 归一化采样参数。
    #
    # 如果用户没有传 sampling_params，就使用默认 SamplingParams()。
    # 如果用户只传了一个 SamplingParams 对象，就复制成和 prompts 等长的列表，
    # 表示所有请求使用同一套采样策略。
    #
    # 注意这里是浅复制同一个对象引用。当前 SamplingParams 基本只读，所以没问题；
    # 如果未来每个请求会修改 sampling_params，就需要为每个请求创建独立副本。
    if sampling_params is None:
      sampling_params = SamplingParams()
    if not isinstance(sampling_params, list):
      sampling_params = [sampling_params] * len(prompts)

    # tqdm 是进度条库。这里 total=len(prompts)，表示每完成一个请求，进度条 +1。
    pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True) if use_tqdm else None

    # 清理上一轮残留的 running_batch（overlap 延迟处理可能留下 stale finished reqs，
    # 导致 _append_running_batch_metadata 的 torch.cat device 不匹配）。
    #
    # generate() 可能被同一个 engine 连续调用多次。上一轮如果有 overlap 延迟释放，
    # running_batch 里可能还残留已经 finished 的 Req。新一轮开始前先清理掉，
    # 避免调度器误以为它们还在运行。
    self.scheduler._filter_batch(self.scheduler.running_batch)
    if not self.scheduler.running_batch.reqs:
      # 如果 running_batch 已经空了，把相关 tensor 元数据也置空。
      # 这样下一轮 schedule 会从干净状态重建这些字段。
      self.scheduler.running_batch.req_pool_indices = None
      self.scheduler.running_batch.seq_lens = None
      self.scheduler.running_batch.seq_lens_cpu = None
      self.scheduler.running_batch.output_ids = None

    # 3. 添加所有请求。
    #
    # add_requests() 会做:
    # - 对字符串 prompt 批量 tokenize，得到 token ids。
    # - 为每个 prompt 创建 Req 对象。
    # - 把 Req 加入 Scheduler.waiting_queue。
    #
    # 这一步还不会跑模型，只是把请求放进队列。
    self.add_requests(prompts, sampling_params)

    # 4. 准备结果容器。
    #
    # results: request_id -> {"text": ..., "token_ids": ..., "request_id": ...}
    # request_ids 记录本次 generate 中出现过的请求 id，用于最后按 id 排序返回。
    # decode_throughput 用于进度条展示每秒 decode token 数。
    results: Dict[int, Dict[str, Any]] = {}
    request_ids = []
    decode_throughput = 0.0

    # 5. 主推理循环。
    #
    # is_finished() 本质上问 Scheduler：是否还有 waiting/running/chunking 请求？
    # 只要还有，就继续调用 step_overlap() 推进一轮。
    #
    # 为什么这里用 step_overlap() 而不是 step()？
    # - step() 是清晰的串行路径，适合理解。
    # - step_overlap() 是性能路径，会把 GPU forward 和 CPU 结果处理重叠起来。
    # 对外输出仍然都是 StepOutput。
    while not self.is_finished():
      t = perf_counter()
      step_output = self.step_overlap()
      step_time = max(perf_counter() - t, 1e-9)

      if pbar:
        # 只有 decode token 才通常用于衡量生成吞吐，因为 prefill 和 decode
        # 的计算形态差异很大，混在一起看会误导。
        if step_output.num_decode_tokens > 0:
          decode_throughput = step_output.num_decode_tokens / step_time
        pbar.set_postfix(
          {
            "Decode": f"{int(decode_throughput)}tok/s",
          }
        )

      # step_output.outputs 里是这一轮产生的输出片段。
      # 流式模式下 delta_text 会逐步累加；非流式 generate 中为了省 CPU，
      # 每步可能不做真实 detokenize，最后会统一 decode token_ids。
      for output in step_output.outputs:
        if output.request_id not in results:
          # 第一次看到某个 request_id 时，创建结果 entry。
          results[output.request_id] = {
            "text": "",
            "token_ids": [],
            "request_id": output.request_id,
          }
          request_ids.append(output.request_id)

        # delta_text 是本轮新增文本；output_token_ids 是该请求当前完整输出 token ids。
        # 对非流式 generate 来说，最终 text 会在循环后被一次性 decode 覆盖。
        results[output.request_id]["text"] += output.delta_text
        results[output.request_id]["token_ids"] = output.output_token_ids

        if output.finished and pbar:
          pbar.update(1)

    # 6. 清空 overlap executor 中还没处理完的 pending batch。
    #
    # step_overlap() 为了重叠 CPU/GPU，可能出现“GPU 已经启动了下一批，
    # 但上一批结果还没完全 process”的情况。主循环结束后必须把这些 pending
    # 结果处理完，否则 results 可能缺最后几个 token 或 finished 状态。
    while self.overlap_executor is not None and self.overlap_executor.has_pending():
      try:
        step_output = self.overlap_executor.process_pending_batch(self._process_overlap_result)
      except Exception as exc:
        # flush 阶段异常也记录上下文，方便定位 overlap 相关问题。
        log_inference_crash(
          exc,
          scheduler=self.scheduler,
          model_runner=self.model_runner,
          stage="generate.flush_pending",
        )
        raise
      if step_output:
        for output in step_output.outputs:
          if output.request_id not in results:
            results[output.request_id] = {
              "text": "",
              "token_ids": [],
              "request_id": output.request_id,
            }
            request_ids.append(output.request_id)
          results[output.request_id]["text"] += output.delta_text
          results[output.request_id]["token_ids"] = output.output_token_ids
          if output.finished and pbar:
            pbar.update(1)

    # 进度条对象需要显式 close，避免终端显示残留。
    if pbar:
      pbar.close()

    # 非流式：每步跳过了 detok，这里对每个请求的 output_token_ids 一次性 decode。
    # 总开销 O(sum seq_len)（每请求一次），远小于每步 decode 的 O(seq_len × num_steps)。
    #
    # 这里使用 detokenizer._decode(...) 是项目内部实现细节；你可以先理解为:
    # token_ids -> 文本字符串。
    if not self._streaming:
      for rid in request_ids:
        entry = results[rid]
        toks = entry["token_ids"]
        entry["text"] = self.detokenizer._decode(list(toks)) if toks else ""

    # 7. 按请求 ID 顺序返回结果。
    #
    # 注意：这里不是按 prompts 原始列表下标排序，而是按 request_id 排序。
    # 当前 add_requests() 是顺序创建 Req，request_id 单调递增，所以通常等价于输入顺序。
    return [results[rid] for rid in sorted(request_ids)]

  # ========== process helper function ============
  def _process_overlap_result(self, batch: ScheduledBatch, result: BatchResult) -> StepOutput:
    """
    处理 overlap 执行中的单个 batch 结果

    作为 OverlapExecutor.process_pending_batch 的回调函数。
    """
    # 处理结果并增量解码
    outputs = self._process_step_result(batch, result)

    # 统计 token 数
    num_prefill = 0
    num_decode = 0
    prefill_prefix_lens: Dict[int, int] = {}
    prefill_extend_lens: Dict[int, int] = {}

    decode_req_set = set()
    if batch.decoding_reqs:
      decode_req_set = {id(r) for r in batch.decoding_reqs}

    if batch.forward_mode.is_extend():
      if batch.prefix_lens and batch.extend_lens:
        for req, pre_len, ext_len in zip(batch.reqs, batch.prefix_lens, batch.extend_lens):
          if id(req) in decode_req_set:
            num_decode += 1
          else:
            num_prefill += ext_len
            prefill_prefix_lens[int(req.req_id)] = int(pre_len)
            prefill_extend_lens[int(req.req_id)] = int(ext_len)
      else:
        num_prefill = sum(batch.extend_lens) if batch.extend_lens else 0
    elif batch.forward_mode.is_decode():
      num_decode = len(batch.reqs)

    return StepOutput(
      outputs=outputs,
      num_prefill_tokens=num_prefill,
      num_decode_tokens=num_decode,
      prefill_prefix_lens=prefill_prefix_lens,
      prefill_extend_lens=prefill_extend_lens,
    )

  def _propagate_future_to_running_batch(self, batch: ScheduledBatch) -> None:
    """
    将非 DECODE batch 上的 output_ids (future placeholder) 传播回 running_batch。

    DECODE batch 就是 running_batch 本身，run_batch_async 直接写入 output_ids。
    但 EXTEND / MIXED batch 是新建对象，需要把 batch.output_ids 中 decode 请求对应的 future indices
    以及新加入 running_batch 的 extend 请求的 future indices 传播回 running_batch.output_ids。
    这样下一轮 schedule(N+1) 可以从 running_batch.output_ids 获取 placeholder。

    对于纯 EXTEND batch（chunked_prefill=False），batch 仅包含新 prefill 请求。
    此时 running_batch 中的旧 decode 请求不在 batch 中，但它们在 running_batch.output_ids
    中的旧 future indices 仍然有效（FutureMap 循环 buffer 空间足够），需要保留。
    """
    running_batch = self.scheduler.running_batch
    if batch.output_ids is None or len(running_batch.reqs) == 0:
      running_batch.output_ids = None
      return

    # Save old output_ids before overwriting.
    # _get_new_batch_prefill appends new reqs to running_batch, so:
    #   running_batch.reqs = [old_surviving_reqs..., new_prefill_reqs...]
    #   old_output_ids corresponds to old_surviving_reqs (same order, post-_filter_batch)
    old_output_ids = running_batch.output_ids

    # Build a CPU-side map from running_batch order -> batch indices.
    # Avoid Python for-loop driving per-element GPU writes; do one masked gather on GPU.
    batch_req_map = {id(r): i for i, r in enumerate(batch.reqs)}
    idx_in_batch = [batch_req_map.get(id(req), -1) for req in running_batch.reqs]

    # Validate any "not-in-batch" entries can be sourced from old_output_ids.
    if old_output_ids is None:
      if any(i < 0 for i in idx_in_batch):
        running_batch.output_ids = None
        return
    else:
      old_len = int(old_output_ids.numel())
      for i, idx in enumerate(idx_in_batch):
        if idx < 0 and i >= old_len:
          running_batch.output_ids = None
          return

    device = batch.output_ids.device
    idx_cpu = torch.as_tensor(idx_in_batch, dtype=torch.int64)

    # 快速路径：running_batch.reqs 与 batch.reqs 完全同序同集（decode 步的常见情况）。
    # 此时无需 nonzero + masked gather，直接 clone 即可，省掉 2 次 nonzero + 多次
    # index/empty kernel（profiler 显示这两步 nonzero+index 占 CPU ~31%）。
    bs = len(running_batch.reqs)
    if bs == len(batch.reqs) and all(i == j for i, j in zip(idx_in_batch, range(bs))):
      running_batch.output_ids = batch.output_ids.clone()
      return

    idx_dev = idx_cpu.to(device, non_blocking=True)
    in_batch = idx_dev >= 0

    new_output_ids = torch.empty(
      (bs,),
      dtype=batch.output_ids.dtype,
      device=device,
    )

    pos_in_batch = in_batch.nonzero(as_tuple=True)[0]
    if pos_in_batch.numel() > 0:
      src = idx_dev[pos_in_batch]
      new_output_ids[pos_in_batch] = batch.output_ids[src]

    pos_old = (~in_batch).nonzero(as_tuple=True)[0]
    if pos_old.numel() > 0:
      if old_output_ids is None:
        running_batch.output_ids = None
        return
      new_output_ids[pos_old] = old_output_ids[pos_old]

    running_batch.output_ids = new_output_ids

  def _process_step_result(self, batch: ScheduledBatch, result: BatchResult) -> List[RequestOutput]:
    """
    处理单步推理结果，返回每个请求的增量输出

    对于 chunked prefill 请求：
    - 将已推理的 token 插入 radix cache（通过 update_unfinished_req_radix_cache）
    - 不进行 output_ids 解码和 finished 判断
    - 保持 is_chunked 标志直到完成所有 chunk
    """
    if batch is None or len(batch.reqs) == 0:
      return []

    next_token_ids = result.next_token_ids
    if isinstance(next_token_ids, torch.Tensor):
      # In overlap mode, next_token_ids is already on CPU (cloned from pinned buffer).
      # Check device to avoid unnecessary .cpu() call which could disguise issues.
      if next_token_ids.device.type != "cpu":
        logger.debug(
          "next_token_ids on %s (non-overlap path does .cpu() here)", next_token_ids.device
        )
        next_token_ids = next_token_ids.cpu()
      next_token_ids = next_token_ids.tolist()

    outputs = []
    finished_req_ids = []

    # Decode tokens in batch to reduce Python overhead.
    active_reqs: List[Req] = []
    active_req_ids: List[int] = []
    active_token_ids: List[int] = []

    for i, req in enumerate(batch.reqs):
      # ============ Chunked Prefill 特殊处理 ============
      # 对于未完成 prefill 的 chunked 请求，只更新 radix cache，
      # 不进行 output_ids 解码和 finished 判断
      if req.is_chunked:
        # 将已推理的 token 插入 radix cache
        # update_unfinished_req_radix_cache 会：
        # 1. 将当前 fill_ids 对应的 KV 写入 radix cache
        #    若未开启 radix cache，则在 req.prefix_indices 中保留私有 KV 映射。
        # 2. 调用方恢复完整 fill_ids，以便后续 chunk 正确切片。
        # 3. 调用方释放临时 RequestPool 行；物理 KV 仍由缓存或当前请求持有。
        self.kv_cache_mgr.update_unfinished_req_radix_cache(req)
        req.fill_ids = req.origin_input_ids + req.output_ids
        self.kv_cache_mgr.request_pool.free([req.req_pool_idx])
        req.req_pool_idx = -1
        logger.debug(
          f"Updated radix cache for chunked prefill req_id={req.req_id}, "
          f"cache_protected_len={req.cache_protected_len}, "
          f"prefix_indices={req.prefix_indices}, "
          f"last_node={req.last_node}, "
          f"req_pool_idx=" + str(req.req_pool_idx)
          if req.req_pool_idx != -1
          else "no req_pool_idx"
        )
        continue

      # ============ 正常请求处理 ============
      # overlap placeholder 路径下可能出现 "stale batch"：
      # schedule(N) 先于 process(N-1)，导致已经在 (N-1) 结束的请求仍出现在 batch(N) 的 record 中。
      # 这时必须跳过该请求，避免 output_ids 超过 max_tokens 或重复解码。
      if req.finished:
        continue

      # Legacy overlap path: schedule(N) may retract a request before
      # process(N-1) has consumed the stale result.  That result must not be
      # appended after the request's old logical state was reset.  The
      # preservation path is non-overlap and clears this one-shot marker when
      # the request is admitted to its valid resume batch.
      if req.is_retracted:
        self.detokenizer.cleanup(req.req_id)
        req.is_retracted = False
        continue

      # The current forward has computed KV for the logical sequence that was
      # present before sampling.  Mark that boundary before appending the new
      # token: the sampled token itself has no KV until the next forward.
      req.mark_kv_committed(req.current_seq_len)
      token_id = next_token_ids[i]
      req.append_output_token(token_id)

      active_reqs.append(req)
      active_req_ids.append(req.req_id)
      active_token_ids.append(token_id)

    if active_reqs:
      eos_id = self.scheduler.eos_token_id
      if self._streaming:
        # 流式：每步 detokenize 产出 delta_text（O(seq_len)/step，流式必需）。
        decoded = self.detokenizer.decode_batch(
          req_ids=active_req_ids,
          token_ids=active_token_ids,
          eos_token_id=eos_id,
          return_full_text=True,
        )
      else:
        # 非流式：跳过每步 detok（仅用 token 判 eos/finished），结束时一次性 decode。
        # decoded = [(delta_text, is_eos, full_text)]
        decoded = [("", (tid == eos_id), "") for tid in active_token_ids]
      for req, token_id, (delta_text, is_eos, full_text) in zip(
        active_reqs, active_token_ids, decoded
      ):
        # 检查是否达到最大 token 数
        if req.ignore_eos:
          is_eos = False
        is_finished = is_eos or len(req.output_ids) >= req.max_tokens
        finish_reason = None

        if is_finished:
          # flush() is typically a no-op after decode(); keep for safety.
          delta_text += self.detokenizer.flush(req.req_id)

          req.finished = True
          if is_eos:
            finish_reason = "eos"
            req.finished_reason = "eos"
          else:
            finish_reason = "max_tokens"
            req.finished_reason = "max_tokens"
          finished_req_ids.append(req.req_id)

        outputs.append(
          RequestOutput(
            request_id=req.req_id,
            delta_text=delta_text,
            full_text=full_text,
            token_id=token_id,
            output_token_ids=req.output_ids,
            finished=is_finished,
            finish_reason=finish_reason,
          )
        )
        if is_finished:
          # Drop detokenizer state to reduce per-step CPU/memory overhead.
          self.detokenizer.cleanup(req.req_id)

    # 处理完成的请求
    self.scheduler._handle_finished_requests(batch, finished_req_ids)

    return outputs

  def is_finished(self) -> bool:
    """检查是否所有请求都已完成"""
    return not self.scheduler.has_unfinished()

  def reset_metrics(self) -> None:
    """Clear collected L3 records before a new measured run."""
    self.metrics.reset()

  def get_metrics_summary(self) -> Dict[str, Any]:
    """Return the current L3 metrics summary."""
    return self.metrics.summary()

  def export_metrics(self, path: Union[str, "os.PathLike[str]"]) -> None:
    """Export iteration/request records and the summary as JSONL."""
    self.metrics.export_jsonl(path)

  # =========== start and stop ========
  def start(self):
    """启动引擎"""
    if self._started:
      return

    self._started = True
    logger.info("Engine started")

  def stop(self):
    """停止引擎"""
    if not self._started:
      return

    logger.info("Stopping engine...")
    if getattr(self, "ep_executor", None) is not None:
      self.ep_executor.close()
    # 单进程模式
    if hasattr(self, "model_runner"):
      # ModelRunner currently has no IPC; just drop the reference
      del self.model_runner

    for p in self.ps:
      p.join()

    self._started = False
    logger.info("Engine stopped")

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.stop()

  # ================= for test =================
  def add_request(
    self,
    prompt: Union[str, List[int]],
    sampling_params: Optional[SamplingParams] = None,
  ) -> int:
    """
    添加请求到调度队列（非阻塞）

    新请求会被添加到等待队列，不会阻塞当前正在进行的推理。
    调用 step() 或 stream_generate() 来驱动推理。

    Args:
        prompt: 输入 prompt (文本或 token ids)
        sampling_params: 采样参数

    Returns:
        request_id: 请求 ID，用于后续查询状态和结果
    """
    if sampling_params is None:
      sampling_params = SamplingParams()

    if isinstance(prompt, str):
      with stage("stage::Tokenizer.encode_one"):
        token_ids = self.prompt_tokenizer.encode_one(prompt)
    else:
      token_ids = prompt

    req = Req(token_ids, sampling_params)
    self.metrics.record_request_arrival(req.req_id, len(token_ids))
    self.scheduler.add(req)
    self._req_by_id[req.req_id] = req
    return req.req_id

  def stream_generate(
    self,
    prompts: Union[str, List[str], List[int], List[List[int]]],
    sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None,
  ) -> Generator[RequestOutput, None, None]:
    """
    流式生成接口（生成器）

    每次 yield 一个请求的增量输出。可以在生成过程中通过
    add_request() 添加新请求，新请求会被自动纳入调度。

    Args:
        prompts: 输入 prompts（支持单个或列表，文本或 token ids）
        sampling_params: 采样参数

    Yields:
        RequestOutput: 每个请求的增量输出

    Example:
        >>> for output in engine.stream_generate(["Hello", "World"]):
        ...   print(f"[{output.request_id}] {output.delta_text}", end="")
        ...   if output.finished:
        ...     print()
    """
    if not self._started:
      self.start()

    # 归一化输入`
    if isinstance(prompts, str):
      prompts = [prompts]
    elif isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], int):
      prompts = [prompts]  # 单个 token ids 列表

    if sampling_params is None:
      sampling_params = SamplingParams()
    if not isinstance(sampling_params, list):
      sampling_params = [sampling_params] * len(prompts)

    # 添加所有初始请求（batch tokenize）
    self.add_requests(prompts, sampling_params)

    # 流式：每步 detokenize 产出 delta_text
    self._streaming = True
    try:
      # 持续推理直到所有请求完成
      while self.scheduler.has_unfinished():
        step_output = self.step_overlap()

        for output in step_output.outputs:
          yield output

      # Flush any remaining pending batches from overlap executor
      while self.overlap_executor is not None and self.overlap_executor.has_pending():
        try:
          step_output = self.overlap_executor.process_pending_batch(self._process_overlap_result)
        except Exception as exc:
          log_inference_crash(
            exc,
            scheduler=self.scheduler,
            model_runner=self.model_runner,
            stage="stream_generate.flush_pending",
          )
          raise
        if step_output:
          for output in step_output.outputs:
            yield output
    finally:
      self._streaming = False
