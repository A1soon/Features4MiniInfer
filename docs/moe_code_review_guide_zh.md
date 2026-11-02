# MiniInfer MoE 代码 Review 路线

这份文档是当前 MiniInfer 工作树的阅读导航。目标不是逐行背代码，而是沿着一条完整数据流完成 review：

```text
Qwen3-MoE layer
  -> router / top-k
  -> token-expert 分桶
  -> 本地专家 grouped GEMM
  -> top-k 加权求和
  -> TP/EP collective
  -> rank 0 sampling / request history
```

建议每读完一个阶段，只记录三件事：

1. 输入和输出的 shape。
2. 这一层拥有或修改的状态。
3. 这一层必须满足的不变量。

## 文件映射

截图中的文件名和实际仓库不完全一致：

| 你要找的内容 | 实际文件 |
| --- | --- |
| 并行 dispatcher | `miniinfer/layers/fused_moe/dispatcher.py` |
| 专家权重和专家 forward | `miniinfer/layers/fused_moe/experts.py` |
| MoE 主层、router、top-k | `miniinfer/layers/fused_moe/layer.py` |
| Triton 分桶和 grouped GEMM | `miniinfer/layers/fused_moe/fused_moe_kernel.py` |
| Qwen3-MoE 模型集成 | `miniinfer/models/fused_qwen3_moe.py` |
| 流式 safetensors 权重加载 | `miniinfer/loader/moe.py` |
| EP 引擎和 rank 间协议 | `miniinfer/engine/expert_parallel.py` |
| batch 长度、位置和 last-token metadata | `miniinfer/scheduler/scheduler_batch.py` |

项目中没有 `layers/fused_moe/moe.py`。如果资料里写 `moe.py`，通常对应低层的 `fused_moe_kernel.py` 或权重加载器 `loader/moe.py`。

## 第一阶段：先建立数学基线

### 1. 独立 reference

先读：

- `tests/moe_reference.py:9`：`moe_reference_with_routes`
- `tests/moe_reference.py:23`：`moe_reference`

这里不要跳过。它不调用被测 router、分桶或 Triton kernel，而是用独立实现计算：

```text
router logits
  -> FP64 softmax
  -> top-k
  -> 可选 top-k 归一化
  -> 对每个专家执行 gate/up/down
  -> 按 routing weight 加权求和
```

手算一个最小例子：2 个 token、4 个专家、top-2。确认每个 token 的两个专家输出如何相加。

### 2. 模型集成位置

按这个顺序读：

1. `miniinfer/models/fused_qwen3_moe.py:41`：`Qwen3MoeSparseMoeBlock`
2. `miniinfer/models/fused_qwen3_moe.py:96`：`Qwen3MoeDecoderLayer`
3. `miniinfer/models/fused_qwen3_moe.py:153`：`Qwen3MoeDecoderLayer.forward`
4. `miniinfer/models/fused_qwen3_moe.py:177`：`Qwen3MoeModel`
5. `miniinfer/models/fused_qwen3_moe.py:219`：模型层循环

需要回答：

- 哪些层是 dense MLP，哪些层是 MoE？
- `first_k_dense_replace`、`decoder_sparse_step`、`mlp_only_layers` 如何决定 `is_moe`？
- attention 输出之后，hidden state 如何进入 `self.mlp`？
- 3D 输入 `[B, T, H]` 何时展平为 `[B*T, H]`？

配置来源：

- `miniinfer/config/model/qwen3_moe.py:75-83`：`num_experts`、`num_experts_per_tok`、`moe_intermediate_size`、`norm_topk_prob`
- `miniinfer/config/model/qwen3_moe.py:92-126`：`from_hf_config`

## 第二阶段：Review 单卡 MoE 主流程

### 3. `FusedMoE` 主层

精读：

- `miniinfer/layers/fused_moe/layer.py:31`：`FusedMoE.__init__`
- `miniinfer/layers/fused_moe/layer.py:75`：`_route`
- `miniinfer/layers/fused_moe/layer.py:83`：`forward`
- `miniinfer/layers/fused_moe/layer.py:114`：`load_expert_weight`

重点检查 `FusedMoE._route`：

```python
router_logits = self.gate(hidden)
weights = F.softmax(router_logits.float(), dim=-1)
topk_weights, topk_ids = torch.topk(weights, self.top_k, dim=-1)
topk_weights = topk_weights / topk_weights.sum(...)
```

要确认：

- router softmax 是否使用 FP32；
- top-k 是对 softmax 后的概率做的；
- `norm_topk_prob=False` 时是否保留原始 top-k 概率；
- `topk_ids.shape == [T, K]`，`topk_weights.shape == [T, K]`；
- `topk_ids` 使用的是全局 expert id；
- `routed_scaling_factor` 是否应该参与输出。当前配置中有这个字段，但 `_route()` 没有使用；若模型配置不是 1.0，需要确认是否会造成正确性差异；
- 当前没有训练用的 auxiliary loss、capacity dropping 或 load balancing，这对推理不一定是问题，但不能把本实现描述成训练版 MoE。

### 4. 专家布局和 CPU fallback

读：

- `miniinfer/layers/fused_moe/experts.py:19`：`FusedExperts.__init__`
- `miniinfer/layers/fused_moe/experts.py:83`：`_ensure_workspace`
- `miniinfer/layers/fused_moe/experts.py:98`：`forward`
- `miniinfer/layers/fused_moe/experts.py:133`：`_eager_forward`

必须记住权重 shape：

```text
w13_weight: [E_local, 2 * I_local, H]
w2_weight:  [E_local, H, I_local]
expert_map: [E_global]
```

`expert_map` 中：

```text
本地 global expert -> local expert row
远程 expert       -> -1
```

CPU fallback 是最容易理解的专家语义：

1. 找出某个本地专家命中的 `(token, top-k choice)`；
2. `gate_up = F.linear(x, w13)`；
3. `silu(gate) * up`；
4. `F.linear(mid, w2)`；
5. 乘 routing weight；
6. `index_add_` 回 token 输出。

先确保 `_eager_forward` 与 reference 一致，再读 GPU kernel。不要反过来。

对应测试：

- `tests/unittests/test_moe_cpu.py:14`：数学 parity，覆盖空 token、不同 top-k、归一化开关和 2D/3D 输入；
- `tests/unittests/test_moe_cpu.py:33`：TP/EP 权重切片重构；
- `tests/unittests/test_moe_cpu.py:59`：非法布局；
- `tests/unittests/test_moe_cpu.py:65`：未初始化 distributed 时应在 collective 前失败。

## 第三阶段：读 token 分桶和 Triton kernel

### 5. 先读 reference 分桶

读：

- `miniinfer/layers/fused_moe/fused_moe_kernel.py:35`：`moe_align_block_size_torch`

它把每个 `(token, k)` 展平成：

```text
flat_pair = token_id * K + choice_id
```

然后按 expert 排序，并把每个 expert 的桶 padding 到 `BLOCK_M` 的倍数。

确认：

- `sorted_token_ids` 存的是 `token*K+k`，不是单纯 token id；
- padding 使用 sentinel `T*K`；
- `expert_ids` 是每个 tile 对应的 expert；
- `num_tokens_post_padded` 是真实 padding 后长度；
- padding 行不能读到真实 hidden 或写入真实输出。

### 6. 再读 Triton 对应实现

顺序：

1. `fused_moe_kernel.py:290`：`moe_align_block_size_triton`
2. `fused_moe_kernel.py:121`：`_small_batch_align_kernel`
3. `fused_moe_kernel.py:192`：`_count_tokens_kernel`
4. `fused_moe_kernel.py:206`：`_meta_kernel`
5. `fused_moe_kernel.py:231`：`_fill_kernel`
6. `fused_moe_kernel.py:274`：`_scatter_tokens_kernel`
7. `fused_moe_kernel.py:506`：`fused_experts_triton`
8. `fused_moe_kernel.py:383`：`_fused_moe_kernel`
9. `fused_moe_kernel.py:459`：`_invoke_grouped_gemm`

`fused_experts_triton` 的逻辑是：

```text
1. 按 expert 分桶
2. GEMM1: hidden @ w13
3. SiLU(gate) * up
4. GEMM2: mid @ w2
5. 乘 top-k weight
6. 对 K 维求和
```

review 时重点问：

- `expert_map` 为远程 expert 时，哪些 tile 被 mask？
- sentinel 行是否可能访问越界？
- `cache1/cache2/cache3` 的容量是否覆盖 `T*K`？
- workspace 复用时，上一次调用残留值会不会进入本次输出？
- 空 token、只有远程命中、某个 expert 没有 token 时是否安全？

GPU 测试：

- `tests/unittests/test_moe_cuda.py:15`：Triton 与独立 reference；
- `tests/unittests/test_moe_cuda.py:63`：远程 expert 必须产生全零 partial 输出；
- `tests/moe_nccl_probe.py`：平衡路由、集中到 rank 0、大消息 all-reduce。

## 第四阶段：Review 权重加载

实际生产路径不是完整 `state_dict` 加载，而是：

- `miniinfer/engine/model_runner.py:97-112`：检测 Qwen3-MoE 并调用 `load_qwen3_moe_weights`；
- `miniinfer/loader/moe.py:64`：`_targets`，建立目标参数和 TP/EP slice；
- `miniinfer/loader/moe.py:125`：`_remote_expert`，判断远程专家；
- `miniinfer/loader/moe.py:140`：`load_qwen3_moe_weights`。

检查：

- router、embedding、attention、norm 是否 replicated；
- 专家是否按 EP 切 expert id；
- gate/up 是否沿 intermediate 输出维切片；
- down 是否沿 intermediate 输入维切片；
- 远程专家是否在读取 tensor 前就跳过；
- shape、finite、dtype overflow、missing/unexpected key 是否明确失败；
- tied embedding 时 `lm_head.weight` 是否复用 embedding。

对应测试：

- `tests/unittests/test_moe_loader_cpu.py:20`：TP/EP/tied/dense layer；
- `tests/unittests/test_moe_loader_cpu.py:76`：坏 checkpoint；
- `tests/unittests/test_moe_loader_cpu.py:103`：有索引/无索引和旧加载器 parity；
- `tests/unittests/test_moe_rank_checkpoint_cuda.py:19`：真实 checkpoint 的 rank-local stream load。

## 第五阶段：Review TP/EP 并行语义

### 7. 并行配置和 collective

读：

- `miniinfer/layers/fused_moe/dispatcher.py:28`：`MoEParallelConfig`
- `dispatcher.py:49`：`expert_range`
- `dispatcher.py:55`：`validate_group`
- `dispatcher.py:67`：`create_moe_parallel_config`
- `dispatcher.py:98`：`MoEDispatcher`
- `dispatcher.py:112`：`combine`
- `dispatcher.py:133`：`prepare_mlp`

当前 EP 的真实语义是：

```text
每个 rank 持有完整 token
-> 每个 rank 只计算本地专家
-> EP group all-reduce partial output
-> MoE-TP group all-reduce partial output
```

这不是经典的 token all-to-all EP。它的优点是实现简单、容易保持 rank 间 shape 一致；代价是每个 rank 都复制 `[T, H]` hidden，并对完整输出做 collective。

手画 `TP=2, EP=2, E=8, I=24`：

```text
rank 0: EP 0, TP 0, experts 0-3, I slice 0-11
rank 1: EP 0, TP 1, experts 0-3, I slice 12-23
rank 2: EP 1, TP 0, experts 4-7, I slice 0-11
rank 3: EP 1, TP 1, experts 4-7, I slice 12-23
```

确认所有 rank 的 collective 顺序完全一致；没有本地 token 的 rank 也必须参加 collective 并贡献零。

对应测试：

- `tests/unittests/test_moe_distributed_cpu.py:19`：四进程 Gloo、balanced/concentrated、TP/EP/TP+EP；
- `tests/moe_nccl_probe.py:62`：1 MiB all-reduce 和集中路由；
- `tests/unittests/test_ep_engine_cpu.py:101`：完整引擎协议和 HF logits。

## 第六阶段：Review EP 引擎控制面

### 8. 请求状态和 batch packet

按顺序读：

1. `miniinfer/engine/expert_parallel.py:43`：`RequestState`
2. `expert_parallel.py:64`：`BatchPacket`
3. `expert_parallel.py:76`：`BatchPacket.from_forward`
4. `expert_parallel.py:98`：`BatchPacket.validate`
5. `expert_parallel.py:151`：`BatchPacket.to_forward`

检查 packet 是否完整携带：

```text
request id
RequestPool row
input_ids / positions
seq_lens / q_lens
本轮 out_slots
完整 token -> physical KV slot 映射
```

这解释了为什么 prefix、chunked prefill、mixed 和 decode 能在 follower 上复现同一 KV 地址，而 follower 不需要自己调度或分配 KV。

### 9. History、采样和生命周期

读：

- `expert_parallel.py:186`：`RequestMirror`
- `RequestMirror.prepare`
- `RequestMirror.sample`
- `RequestMirror.commit`
- `RequestMirror.release`
- `expert_parallel.py:371`：`_follower`
- `expert_parallel.py:436`：`ExpertParallelExecutor`
- `expert_parallel.py:531`：`execute`
- `expert_parallel.py:588`：`sample`
- `expert_parallel.py:602`：`commit`
- `expert_parallel.py:614`：`release`
- `expert_parallel.py:627`：`close`

跟踪一次完整控制流：

```text
READY -> INIT_KV -> PREPARE -> RUN -> SAMPLE -> COMMIT -> RELEASE -> STOP
```

必须确认：

- 只有 rank 0 调 sampler；
- follower 不自行追加生成 token；
- chunk-only forward 不追加 output history；
- 普通 decode/prefill 的采样 token 在 COMMIT 时同步；
- worker PREPARE 失败、退出、stall 时 rank 0 能有界退出；
- RUN 阶段 collective 内部崩溃是否可能先让 rank 0 卡在 NCCL。

### 10. Engine 和 Scheduler 接入

最后读：

- `miniinfer/config/engine/config.py:117-137`：EP 配置限制和 Qwen3-MoE 校验；
- `miniinfer/engine/llm_engine.py:274-294`：EP executor 初始化；
- `miniinfer/engine/llm_engine.py:627-770`：`step` 中 forward、sample、commit、release；
- `miniinfer/scheduler/scheduler_batch.py:693-747`：EXTEND/MIXED metadata；
- `miniinfer/scheduler/scheduler_batch.py:780-816`：`compute_positions_extend`。

重点看 prefix/chunk/mixed 如何维护：

```text
extend_seq_lens
extend_prefix_lens
positions
last_token_indices
out_cache_loc
```

## 第七阶段：CUDA Graph 最后看

只有前面 eager EP 正确后才读：

- `miniinfer/engine/cuda_graph_runner.py:177`：padding scratch slot；
- `cuda_graph_runner.py:209`：decode graph capture；
- `cuda_graph_runner.py:329`：decode replay；
- `miniinfer/engine/piecewise_cuda_graph.py:17`：V100 prefill/mixed graph runner；
- `piecewise_cuda_graph.py:93`：query padding；
- `piecewise_cuda_graph.py:198`：capture；
- `piecewise_cuda_graph.py:239`：replay；
- `miniinfer/engine/expert_parallel.py:297`：所有 rank 协调 graph capture；
- `expert_parallel.py:329`：graph release。

当前要特别区分：

- decode full graph 已有真实验证路径；
- prefill/mixed graph 是 V100 backend 的 opt-in 实验路径；
- 新增 prefill/mixed graph 必须有真实 capture/replay parity 证据，不能只凭 bucket/layout 单测；
- scratch KV pages 在 release 时是否归还 allocator，是必须检查的生命周期问题；
- 目前这条路径仍是整模型 full-forward capture，不等同于 vLLM/SGLang 的 attention eager break + dense segments。

## Review 记录模板

每个函数用下面格式写笔记：

```text
函数：
输入 shape：
输出 shape：
读取的状态：
修改的状态：
跨 rank 的通信：
必须满足的不变量：
我发现的风险/疑问：
对应测试：
```

## 当前优先检查项

按优先级，不要一开始扩展范围：

1. `routed_scaling_factor` 是否应该进入 router 输出；
2. FP32 router softmax、top-k 和 top-k 归一化是否与目标 Qwen3-MoE 语义一致；
3. sentinel、remote expert 和 padding 行是否会读写越界或污染真实输出；
4. TP/EP collective 顺序和空本地 token rank；
5. prefix/chunk/mixed 的 `RequestState.history` 和 KV slot 映射；
6. worker 在 RUN/collective 阶段失败时是否能有界退出；
7. CUDA Graph scratch page 的分配、复用和释放；
8. all-reduce baseline 与经典 all-to-all EP 的通信代价差异。

## 磁盘使用统计（2026-10-28）

以下统计只读执行，没有删除或清理任何文件。

### APFS Data 卷

`diskutil info /System/Volumes/Data`：

```text
容器总空间：494.4 GB
Data 卷已用：247.1 GB
容器可用：219.9 GB
```

`df -h` 显示：

```text
Data 挂载点容量：460 GiB
已用：230 GiB
可用：205 GiB
占用率：53%
```

两组数字的单位和 APFS 统计口径不同，不能直接相减；结论一致：当前仍有约 220 GB 可用空间，Data 卷约半满。

### MiniInfer 源码仓库

源码仓库应只包含程序、测试和文档。模型 checkpoint、环境目录、运行缓存和性能 trace
通常是本地实验产物，不应提交到源码仓库；大模型权重应按独立的数据管理流程保存。

### 复现命令

```bash
df -h /System/Volumes/Data
diskutil info /System/Volumes/Data
du -hd 1 . | sort -h
du -hd 2 ../experiments | sort -h
```

检查空间前确认当前工作目录；清理、移动权重或删除缓存前应核实具体路径和内容。
