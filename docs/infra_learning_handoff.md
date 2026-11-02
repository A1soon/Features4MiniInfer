# Infra Learning Handoff

本文档用于交接你的 MiniInfer / LLM inference infra 学习过程。

目标是把两个 session 分工清楚：

- 本 session：负责整体学习规划、阶段复盘、概念路线、简历项目设计、论文/工程/research 方向建议。
- 新 session：专门负责代码细节、CUDA/Triton/PyTorch 语法、某个函数逐行解释、调试和实验实现。

## 当前学习目标

你目前的目标不是单纯“读完 MiniInfer”，而是通过 MiniInfer 建立 LLM inference infra 的系统理解，并逐步形成可以写进简历、面试能讲清楚、后续能迁移到 vLLM / SGLang / FlashInfer 的能力。

核心能力目标：

1. 能讲清楚一个 request 从 prompt 到 output token 的完整生命周期。
2. 能解释 prefill / decode / mixed batch / chunked prefill 的区别和调度目的。
3. 能理解 KV cache 的物理存储、page allocator、request pool、prefix cache。
4. 能读懂 attention backend 需要哪些 metadata，比如 `input_ids`、`seq_lens`、`req_pool_indices`、`out_cache_loc`、page table。
5. 能用 profiling / benchmark 证明一个 inference 系统瓶颈在哪里。
6. 最终产出一个可写入简历的 inference infra 项目总结。

## 已完成内容

你已经读过并基本理解：

- `examples/minimal_engine.py`
- `LLMEngine.__init__`
- `LLMEngine.add_requests`
- `LLMEngine.generate`
- `LLMEngine.step`
- `Req`
- `ScheduledBatch`
- `ForwardBatch`
- `Scheduler.schedule`
- `Scheduler._get_new_batch_prefill`
- `Scheduler._get_decode_batch`
- `KVCacheManager` 的主要入口

你目前的理解已经包括：

- 用户提交 prompt 后，`add_requests()` 创建 `Req` 并加入 `waiting_queue`。
- `generate()` 不断调用 `step()` 或 `step_overlap()` 直到请求完成。
- `Req` 是 request 生命周期状态。
- `ScheduledBatch` 是 scheduler 这一轮决定要跑什么。
- `ForwardBatch` 是模型 forward / attention backend 真正消费的执行元数据。
- prefill 阶段也会输出 logits，因为 prompt 最后一个位置的 logits 用于采样第一个 output token。

## 已添加详细注释的文件

这些文件已经加过较详细的中文注释，适合优先阅读：

- `examples/minimal_engine.py`
- `miniinfer/engine/llm_engine.py`
- `miniinfer/scheduler/scheduler_batch.py`
- `miniinfer/scheduler/scheduler.py`
- `miniinfer/scheduler/prefill_adder.py`
- `miniinfer/kvcache/kv_cache_manager.py`
- `miniinfer/kvcache/memory_pool.py`
- `miniinfer/kvcache/radix_cache.py`

注意：注释的目的不是替代代码，而是帮助你建立“读这段代码时应该抓什么”的方向。

## 两个 Session 的分工

### 本 Session：规划 / 复盘 / 概念主线

适合问：

- 我这周应该读哪些内容？
- 我读完这些函数后，理解是否正确？
- 这些概念应该先补哪些，哪些可以后补？
- MiniInfer 和 vLLM / SGLang / Genllm 有什么区别？
- 某个系统设计在业界有什么意义？
- 这部分能不能写进简历？
- 我应该做什么 benchmark / 小实验？
- 某篇论文和 MiniInfer 哪段代码对应？
- 我想成为 infra 高手，还缺哪块能力？

本 session 不建议长期做：

- 某个 Triton kernel 的逐行解释。
- CUDA 语法逐项展开。
- 具体代码 debug。
- 单个函数卡很久的局部推导。

### 新 Session：代码 / CUDA / Triton / PyTorch 细节

适合问：

- 请逐行解释某个函数。
- 这个 Triton 语法是什么意思？
- 这个 tensor shape 为什么是这样？
- 这个 kernel 如何改成 Python 伪代码？
- 这段代码有没有 bug？
- 请帮我写一个 toy example 验证 page allocator。
- 请帮我用 profiler 跑一个实验。
- 请帮我实现一个小功能或 benchmark。

新 session 可以从下面这段提示词开始：

```text
我正在学习当前仓库中的 MiniInfer LLM inference 项目。另一个 session 负责整体 infra 学习规划；这个 session 专门负责代码、CUDA、Triton、PyTorch 细节解释和实验实现。

我的基础比较弱，请你解释代码时：
1. 先说明这个函数在系统里的作用。
2. 再说明输入/输出是什么。
3. 再把关键变量和 tensor shape 讲清楚。
4. 对复杂 CUDA/Triton 代码，请先改写成 Python 伪代码。
5. 最后用一个小例子走一遍。

我现在想看这个文件/函数：...
```

## 第二周阅读顺序

第二周主线是：KV cache 和 scheduler 如何把请求变成可执行 batch。

建议顺序：

1. `miniinfer/kvcache/memory_pool.py`
   - `MHAKVCacheStorage`
   - `get_num_new_pages`
   - `PagedTokenAllocator`
   - `alloc_extend`
   - `alloc_decode`
   - `RequestPool`

   这一层回答：KV cache 物理上怎么存，token 位置怎么映射到 page。

2. `miniinfer/kvcache/radix_cache.py`
   - `TreeNode`
   - `RadixCache`
   - `match_prefix`
   - `insert`
   - `evict`
   - `inc_lock_ref`
   - `dec_lock_ref`
   - `_split_node`
   - `_insert_helper`

   这一层回答：prefix cache 如何复用已有 KV，为什么需要 radix tree 和 lock。

3. `miniinfer/kvcache/kv_cache_manager.py`
   - `prefix_for_waiting_req`
   - `prepare_for_extend`
   - `prepare_for_mixed`
   - `prepare_for_decode`
   - `release_request`
   - `update_finished_req_radix_cache`
   - `get_page_table`
   - `write_cache_indices`

   这一层回答：scheduler 选出的请求如何被准备成 attention backend 能用的元数据。

4. `miniinfer/scheduler/prefill_adder.py`
   - `PrefillAdder`
   - `try_add_prefill`
   - `add_chunked_req`

   这一层回答：为什么某个请求能进 batch，为什么某个请求会被 chunked prefill。

5. `miniinfer/scheduler/scheduler.py`
   - `schedule`
   - `_get_new_batch_prefill`
   - `_get_decode_batch`
   - `_check_decode_mem`
   - `drain_pending_releases`

   这一层回答：系统每一轮到底选择 prefill、decode 还是 mixed batch。

## 当前最需要补的概念

不要一次性补完所有东西。按当前 MiniInfer 阅读顺序，先补这些：

1. KV cache
   - K/V 是 attention 中每层保存的历史 key/value。
   - prefill 会为 prompt 中的 token 写入 K/V。
   - decode 每生成一个新 token，也会追加这个 token 的 K/V。

2. Paged KV cache
   - 不把每个请求的 KV 当作连续大数组管理。
   - 把 KV cache 切成 page/block，减少碎片并方便复用。
   - `RequestPool` 保存 token position 到 physical KV location 的映射。

3. Prefix cache
   - 如果两个请求有相同前缀，后来的请求可以复用前面已经算好的 K/V。
   - radix tree 用来高效保存和查找 token prefix。
   - lock/ref count 用来防止正在使用的 prefix 被 evict。

4. Chunked prefill
   - 长 prompt 不一定一次 prefill 完。
   - 可以切成多个 chunk，避免 prefill 长时间占用 GPU，使 decode 请求饿死。

5. Attention backend
   - 它不是一个固定函数，而是一组负责高效 attention 计算的 backend。
   - backend 需要 input token、seq length、KV cache buffer、page table 等元数据。

## 4-5 周 MiniInfer 简历路线

### Week 1：Engine / Request / Batch 主线

目标：

- 讲清楚 request 生命周期。
- 讲清楚 `generate -> step -> scheduler -> model_runner -> sample -> process_result`。

产出：

- 一张 request lifecycle 图。
- 一篇 1-2 页笔记：MiniInfer engine 主循环。

### Week 2：Scheduler / KV Cache / Prefix Cache

目标：

- 讲清楚 waiting/running/chunking 三类请求。
- 讲清楚 prefill/decode/mixed batch。
- 讲清楚 KV page allocator、RequestPool、RadixCache。

产出：

- 一张 KV cache 数据结构图。
- 一个 page allocator toy example。
- 一个 prefix cache 命中例子。

### Week 3：Model Forward / Attention Backend

目标：

- 读懂模型 forward 如何调用 attention。
- 读懂 attention backend 需要哪些 metadata。
- 建立 FlashAttention / paged attention 的基本概念。

产出：

- 一张 `ForwardBatch -> attention backend` 数据流图。
- 一个小笔记：prefill attention 和 decode attention 的区别。

### Week 4：性能实验 / Benchmark / Profiling

目标：

- 学会测 TTFT、TPOT、throughput、GPU memory。
- 跑不同 batch size、prompt length、max_tokens、chunk size 的实验。
- 用 profiling 解释瓶颈。

产出：

- 一个 benchmark 脚本或实验 notebook。
- 一份 profiling report。

### Week 5：简历项目包装

目标：

- 把 MiniInfer 阅读和实验包装成一个可讲项目。
- 准备 3-5 个面试故事。

可写方向：

- Implemented and analyzed an LLM inference engine covering request scheduling, paged KV cache, prefix caching, and prefill/decode batching.
- Built benchmarks to evaluate TTFT/TPOT tradeoffs under different prompt lengths and chunked prefill settings.
- Studied KV cache allocation and prefix reuse mechanisms inspired by vLLM/SGLang-style serving systems.

## 想成为 Infra 高手的长期路线

你需要同时补三类能力：

### 1. 系统能力

重点：

- operating systems 基础
- networking / RPC / queueing
- distributed systems
- scheduling
- memory management
- observability

学习方式：

- 不要只看书，结合 MiniInfer / vLLM / SGLang 的具体设计学。
- 每学一个概念，都问它在 inference serving 中对应什么问题。

### 2. GPU / Kernel 能力

重点：

- CUDA memory hierarchy
- warp / block / occupancy
- memory coalescing
- shared memory
- Tensor Core
- Triton 编程模型
- profiling with Nsight / PyTorch profiler

学习方式：

- 先写小 kernel，不要直接手写 paged attention。
- 推荐顺序：vector add -> reduction -> layernorm -> softmax -> matmul toy -> attention toy。

### 3. Research / Paper 能力

优先论文：

- FlashAttention
- Orca
- vLLM / PagedAttention
- Sarathi-Serve
- DistServe
- FlashInfer

读论文方法：

1. 这篇论文解决什么系统瓶颈？
2. 它的核心 design 是什么？
3. 它牺牲了什么？
4. MiniInfer / vLLM / SGLang 中哪里体现了类似思想？
5. 我能不能用一个小实验复现其中一个结论？

## 每周时间分配建议

你每周大约 30-40 小时，可以这样分：

- 12-15h：读 MiniInfer / vLLM / SGLang 代码。
- 6-8h：补 CUDA / Triton / PyTorch 基础。
- 5-6h：读论文和写笔记。
- 5-8h：做 benchmark / profiling / 小实验。
- 2-3h：整理简历材料和复盘。

关键不是时间长，而是每周必须有产出：

- 一张图。
- 一篇短笔记。
- 一个 toy example。
- 一个 benchmark 结果。
- 一个可以讲给面试官的 tradeoff。

## 每次复盘可以用的问题

每周结束时，在本 session 问：

```text
我这一周读了这些文件/函数：...
我的理解是：...
我还卡在：...
请你评估我是否可以进入下一周，并指出我应该补哪些概念。
```

如果是代码细节，在新 session 问：

```text
我正在读仓库中的 `...` 这个函数。
我不理解这些变量：...
请你先解释函数作用，再解释输入输出，再用 Python 伪代码和具体例子讲一遍。
```

## 当前注意事项

1. 不要因为一个 Triton kernel 卡住，就停止整个主线学习。
   - 对复杂 kernel，先理解输入/输出和 Python 等价逻辑。
   - Triton 语法可以在新 session 单独补。

2. 不要过早切到 vLLM 源码。
   - MiniInfer 先读出系统骨架。
   - vLLM / SGLang 后面作为工业级对照。

3. 不要只读不做实验。
   - infra 能力最终要靠 benchmark、profile、tradeoff 来证明。

4. 概念补习要服务于当前代码。
   - 读 KV cache 时补 page allocator / prefix cache。
   - 读 attention backend 时补 FlashAttention / paged attention。
   - 读 scheduler 时补 queueing / batching / latency-throughput tradeoff。

## 下一步建议

你现在可以继续第二周：

1. 在本 session 里确认第二周整体顺序和复盘。
2. 在新 session 里逐个解决代码细节，比如 `alloc_extend_kernel`、`write_cache_indices`、`RadixCache.match_prefix`。
3. 读完 `memory_pool.py` 和 `radix_cache.py` 后，回到本 session 复盘：

```text
我读完了 memory_pool.py 和 radix_cache.py，我的理解是：...
请你评估我的理解，并告诉我是否可以进入 kv_cache_manager.py。
```
