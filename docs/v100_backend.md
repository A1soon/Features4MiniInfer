# V100 attention backend 接入与验证

## 如何使用

在已有 V100 实验镜像 `miniinfer-v100:py312-torch210-cu126-fa` 中运行项目，
创建引擎时显式选择：

```python
import torch
from miniinfer import LLMEngine, SamplingParams

with LLMEngine(
    model="/models/your-local-qwen2",
    attention_backend="flash_attn_v100",
    dtype=torch.float16,
    page_size=256,
    max_num_seqs=4,
    max_context_len=4096,
    enable_prefix_cache=True,
    enable_chunked_prefill=True,
    chunked_prefill_size=1024,
) as engine:
    results = engine.generate(
        ["Explain paged KV cache."],
        SamplingParams(temperature=0, max_tokens=32),
    )
```

模型仍需位于本地。默认 backend 仍是 FlashInfer；选择 V100 backend 不会改变
其他 backend 的选择方式。这个 backend 要求 SM70、FP16、page_size 为 16 的
正整数倍，以及 head_dim 为 32/64/96/128/192/256。已验证页长为 16 和 256。
Chunked prefill 的 chunk 大小至少要覆盖一页，否则现有调度器无法推进长请求，
配置阶段会提前报错。

此路径使用第三方的推理前向算子，没有提供训练 backward 或 FP8 KV cache。

## 算子来源

使用 [zhinianqin/flash-attention-v100](https://github.com/zhinianqin/flash-attention-v100)，
固定提交 `c2eda5e6115b98c3ba4bfd181570668742eece22`。
实验环境为 Python 3.12.12、PyTorch 2.10.0+cu126、Triton 3.6.0、
CUDA Toolkit 12.6.3、驱动 535.98、Tesla V100-SXM2-32GB。

导入入口是 `vllm_flash_attn.flash_attn_varlen_func`，不需要安装 vLLM。
扩展沿用上游 BSD-3-Clause 许可证。CUDA kernel 未修改；离线构建时仅跳过
setup.py 的 git submodule 拉取，因为固定提交的 CUTLASS 已经提供。

## 调用关系

`LLMEngine` 把配置交给 `ModelRunner.init_attn_backend()`，创建
`FlashAttentionV100Backend`。每轮 forward 先准备 metadata，随后各层的
`AttentionImpl` 调用 backend；EXTEND/MIXED 和 DECODE 共用 paged varlen 算子。

建议按以下顺序 review：

1. `miniinfer/layers/attention_backend/flashattention_v100_backend.py`
   的 `init_forward_metadata()`：RequestPool 到算子 metadata 的转换。
2. 同一文件的 `_forward()`：本轮 K/V 写入，以及完整历史 K/V 的读取。
3. `init_cuda_graph_metadata()` / `update_cuda_graph_metadata()`：固定容量和动态长度。
4. `miniinfer/engine/model_runner.py` 的 backend 选择分支，以及
   `miniinfer/layers/sample.py` 的可选 FlashInfer / PyTorch 采样路径。
5. `tests/unittests/test_flashattention_v100.py` 的 reference 和完整引擎测试。

RequestPool 每格保存的是 token slot，因此每隔 page_size 取一个 slot，
再整除 page_size，才得到物理 page id。新的 backend 传给算子的是：

| 参数 | 含义 |
| --- | --- |
| q | 仅包含本轮新 token，形状 `[total_new_tokens, Q_heads, D]` |
| k / v | 完整物理 cache 的 view，形状 `[pages, page_size, KV_heads, D]` |
| cu_seqlens_q | 本轮每个请求 Q 长度的累计和，int32 GPU tensor |
| seqused_k | 每个请求完整 KV 长度，包含本轮刚写入的 token |
| block_table | 每个请求的逻辑 page 到物理 page 的映射，int32 GPU tensor |

例如 mixed batch 的 Q 长度为 `[17, 1]`、完整 KV 长度为 `[273, 511]`，则
`cu_seqlens_q=[0,17,18]`。第一个请求复用 256-token prefix，Q 对应位置
256 到 272；第二个请求的 Q 对应位置 510。算子的右下对齐 causal mask
负责这个位置偏移，不能按 Q 的局部行号直接使用左上三角 mask。

本轮 KV 通过已有 `KVCacheManager.set_kv_buffer()` 写入 `out_cache_loc`。
Attention 直接按 page table 读历史 KV；这里没有将完整 prefix gather 成连续副本。

## CUDA Graph

Decode 的 Q 长度固定为 1，但 KV 长度会增长。捕获时的 Python 标量也会被冻结，
因此 `max_seqlen_k` 必须固定为整张静态 block_table 的容量，不能固定在预热时
常用的长度 2。每次 replay 仅更新 GPU 上的实际长度和 page table 内容。

每个 capture batch size 保存独立 metadata；eager prefill 即使替换了当前
metadata，也不会覆盖其他 decode graph 捕获的缓冲区。

## 可选依赖

旧 FA2/FA3 算子推迟到选择对应 backend 时导入，基础 attention wrapper 也只在
调用时导入官方 flash_attn。这样只有 V100 扩展时仍能导入 MiniInfer。

没有 FlashInfer 时，Sampler 使用 PyTorch 完成 temperature、top-k、top-p
以及 greedy/mixed 采样。过滤按 top-k 后 top-p 执行；greedy 行保持 argmax。
这些是兼容路径，没有对其吞吐做性能承诺。

## 验证与复现

2026-10-26 在 V100 GPU 0 上运行：

```bash
bash scripts/run_v100_validation.sh 0
```

镜像通过 `V100_IMAGE` 指定；结果目录通过 `V100_RESULTS_DIR` 指定，默认
`/tmp/miniinfer-v100-validation-$UID/`。脚本使用只读源码挂载和关闭网络的
容器，生成 `pytest.log`、`junit.xml`、`image.json`，不下载模型。

新增验证共 87 项，全部通过：

- 67 项直接算子验证：六种 head_dim、页长 16/256、非连续物理页、共享 prefix、
  QKV split view、交错 KV stride、未使用区域 NaN 哨兵、causal prefill、prefix、
  MHA/MQA/GQA decode、mixed 和 CUDA Graph。较长用例在 head_dim 64/128 下包含 1057-token
  prefill、2097-token KV 的 prefix prefill、4097-token KV 的 decode。
- 16 项 backend/采样/配置验证：实际 RequestPool 映射与 KV 写入、复用缓存、
  graph 预热后跨页增长、metadata 切换、采样过滤、无效配置拒绝。
- 4 项完整引擎验证：临时创建两层随机初始化 Qwen2 checkpoint 和 tokenizer，
  使用真实权重加载、RoPE、norm、MLP、调度器、KV allocator、RadixCache 和 sampler。
  Eager、CUDA Graph、overlap、prefix 命中及 chunked mixed 的 greedy token ids
  与 Hugging Face eager 模型逐 token 一致；prefix 测试还断言命中 256 个 token，
  mixed 测试断言实际出现 prefill/decode 混合轮次。

算子 reference 使用独立的逻辑 K/V 做 FP32 attention，不通过被测 page table
反向构造期望值，避免地址映射错误被 reference 一起掩盖。
数值阈值为 rtol=0.01、atol=0.01。

另有 26 个已有测试通过，覆盖 prefill budget、decode input ids、chunked KV 生命周期
和 overlap 调用顺序。

这些测试证明上述路径的正确性。随机小模型测试不代表预训练模型的生成质量，
也没有测量 TTFT、TPOT、吞吐、长时间并发稳定性或 Qwen3/MoE。本接入为单卡路径，
不新增多卡模型执行能力。
