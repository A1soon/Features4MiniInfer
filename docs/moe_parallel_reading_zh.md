# MiniInfer MoE 与多卡推理阅读清单

目标：在 V100 上验证现有 MoE 算子，打通 Qwen3-30B-A3B-Instruct-2507 的四卡专家并行，
随后研究 Attention TP、TP/EP 组合和性能瓶颈。资料按当前实现所需知识排序。
先完成第 1、2 项即可开始 review EP 基线；第 3、4 项在对应开发阶段补。

## 1. MoE 前向：第一优先级

入门：[Hugging Face — Mixture of Experts Explained](https://huggingface.co/blog/moe)。
正文暂时无法直接访问时，可以从
[官方 GitHub 文稿](https://github.com/huggingface/blog/blob/main/moe.md)阅读。
这篇文章以 Mixtral、Switch 等模型举例；它们的专家数、top-k、容量限制不等于 Qwen3 的配置。

优先读这些小节：

- What is a Mixture of Experts (MoE)?
- What is Sparsity?
- Load balancing tokens for MoEs：理解负载不均的后果，训练 loss 公式可略读。
- Making MoEs go brrr 下的 Parallelism、Capacity Factor and communication costs、
  Serving techniques：理解专家部署与通信。

模型架构：[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)，
先读 [第 2 节 Architecture](https://arxiv.org/html/2505.09388v1#S2)。
再对照 [2507 模型卡的 Model Overview](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507#model-overview)。
2507 的具体配置以我们固定 revision 的 config.json 为准，训练配方和评测表暂时略过。

需要掌握的计算过程：

```text
hidden [T,H]
  -> router logits [T,E]
  -> softmax + top-k + 概率归一化
  -> topk_ids / topk_weights [T,K]
  -> 每个 token 进入 K 个专家
  -> 各专家 SwiGLU MLP
  -> 按路由权重求和，输出 [T,H]
```

对我们的目标模型，H=2048、E=128、K=8、专家 intermediate_size=768。
不考虑 padding 时，T 个 token 产生 T*K 个 token/expert 对。
专家 MLP 的三个矩阵在 PyTorch 权重布局下分别是
`gate/up [I,H]`、`down [H,I]`；它们计算 `silu(x @ gate.T) * (x @ up.T)`，
然后乘 `down.T`。Attention、norm、router 等部分仍是共享计算。

对应代码：

- [layer.py](../miniinfer/layers/fused_moe/layer.py#L68)：路由、加权、专家调用。
- [experts.py](../miniinfer/layers/fused_moe/experts.py#L134)：先读 CPU eager reference，追踪一个 token 的结果。
- [fused_qwen3_moe.py](../miniinfer/models/fused_qwen3_moe.py#L93)：MoE 如何放进 Transformer block。

读完能回答：

1. “总参数 30B、激活约 3B”分别影响显存还是计算量？为什么显存不能按 3B 算？
2. 一个 token 选择 8 个专家，最终为什么仍只有一个 H 维输出？
3. top-k 权重在哪里归一化、在哪里乘到输出上？重复乘或漏乘会怎样？
4. 为什么同样 token 数，不同路由分布可能产生不同耗时？

## 2. 分布式与 EP：四卡基线的直接前置

第一份资料：[NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)。
先看 AllReduce、Broadcast、AllGather、AlltoAll 的图和输入/输出。
不需要先学习 NCCL ring/tree 的底层算法或 C API。

第二份资料：[PyTorch 2.10 Distributed](https://docs.pytorch.org/docs/2.10/distributed.html)。
版本与实验镜像对应。按 API 名搜索下面的条目即可，无需通读全部文档：

- `init_process_group`、`get_rank`、`get_world_size`、`new_group`、`destroy_process_group`。
- `all_reduce`、`broadcast`、`all_gather`、`all_to_all_single`。
- 区分 global rank、local rank、进程组内 rank、GPU 的物理编号。
- 理解 collective 的参与者、调用顺序和张量数量/类型为何必须匹配。

我们先实现的 EP 数据流：

```text
rank 0/1/2/3 持有相同输入 token
  -> router 选择相同专家（共享 router 权重与输入）
  -> 每个 rank 仅保存、计算自己的 32 个专家
  -> 得到各自的 partial output [T,H]
  -> all_reduce(sum)
  -> 各 rank 得到完整输出 [T,H]
```

这是复制 token 的 EP 基线。它可以在 Attention 暂时复制的情况下独立验证。
输出合并需要 SUM；如果使用平均值，会把结果错误地缩小四倍。
某个 rank 没有 token 路由到本地专家时，也必须按约定参与 collective，贡献零结果。

后续 all-to-all 方案会先按专家归属发送 token，再计算、回传并恢复原 token 顺序。
这涉及不同的 token 数据分布，不能只替换一次通信 API。
all-to-all 的速度优势需要结合 batch、路由和数据布局实测。

对应代码：

- [dispatcher.py](../miniinfer/layers/fused_moe/dispatcher.py#L29)：TP/EP 配置和 `combine`。
- [experts.py](../miniinfer/layers/fused_moe/experts.py#L23)：本地专家范围与 `expert_map`。
- [layer.py](../miniinfer/layers/fused_moe/layer.py#L107)：专家权重归属与切片。

读完能回答：

1. GPU 只看到编号 0，是否意味着它一定是宿主机 GPU 0？
2. 四个 rank 的 partial output 相加，为什么可以恢复单卡完整专家计算？
3. 某个 rank 提前 return，其他 rank 正在 all-reduce，可能发生什么？
4. rank 0 采样后为什么需要把 token 同步给其他参与推理的 rank？

## 3. TP：理解矩阵切分，随后补 Attention 并行

核心论文：[Megatron-LM](https://arxiv.org/abs/1909.08053)，
优先读 [第 3 节 Model Parallel Transformers](https://arxiv.org/html/1909.08053v4#S3) 的 MLP 和 Attention 切分图。
暂时略过优化器、训练集、反向传播和大规模训练结果。

更贴近 SwiGLU 的讲解：[PyTorch Tensor Parallel Tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)。
看 `FeedForward`、`ColwiseParallel`、`RowwiseParallel` 和 Attention 部分。
这是原理参考，不表示我们必须换成 DTensor 实现；教程的新版 API 也不应直接假定与镜像一致。

推荐用 PyTorch 的 `F.linear(x,W)=x @ W.T` 理解维度，避免“行/列并行”名称造成混淆：

| 权重 | PyTorch shape | 沿哪个维度切 | 本 rank 得到什么 |
| --- | --- | --- | --- |
| gate/up | [I,H] | I，也就是输出特征维度 | [T,I/TP] |
| down | [H,I] | I，也就是输入特征维度 | [T,H] 的部分和 |

每个 rank 独立计算自己的 gate/up 和逐元素乘法，再计算 down 的部分输出，
最后一次 all-reduce 恢复完整 [T,H]。不同 rank 的 gate 与 up 必须切到相同 I 区间。
融合 gate/up 后也要分别切各自半段，不能随意把拼接矩阵直接均分。

对应代码：[load_expert_weight](../miniinfer/layers/fused_moe/layer.py#L107)。
当前 Attention 使用完整 QKV/O 投影，后续才补 head 切分和输出归约。

读完能回答：

1. 为什么 gate/up 后不用立刻通信，down 后却需要求和？
2. 为什么每张卡各放完整模型，只能得到多副本，无法降低单副本的权重占用？
3. EP 切“专家编号”、TP 切“专家内部维度”，两者分别如何改变权重 shape？
4. Qwen3-30B-A3B 只有 4 个 KV heads，Attention TP=8 时为何需要特殊处理？

## 4. Triton 与 V100：算子验证、优化阶段阅读

主资料：[Triton Matrix Multiplication 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)。
先读 Block-Level Matrix Multiplication、Pointer Arithmetic 和 kernel 主循环；
再读 L2 cache / program 重排，autotune 参数表最后看。

重点是理解一个 program 计算一个输出 tile，M/N 方向选择输出范围，K 方向在 program
内部循环累加。`tl.arange` 表示逻辑元素集合；元素到线程/warp 的映射由编译器和布局决定，
不能把二维索引直接理解为两组独立的“行线程、列线程”。

V100 对照资料：[NVIDIA Volta Tuning Guide](https://docs.nvidia.com/cuda/archive/12.6.3/volta-tuning-guide/index.html)。
优先看 1.4.1.3 Occupancy、1.4.2 Tensor Core Operations、1.4.3 Memory Throughput。
理解 FP16 乘法、FP32 累加、寄存器/共享内存占用和访存合并。
现代教程中的硬件配置不能直接当成 V100 最优参数；BF16/FP8/TF32 路径也不能照搬到 V100。

对应代码：

- [token 分桶参考](../miniinfer/layers/fused_moe/fused_moe_kernel.py#L35)。
- [grouped GEMM kernel](../miniinfer/layers/fused_moe/fused_moe_kernel.py#L383)。
- [专家前向入口](../miniinfer/layers/fused_moe/fused_moe_kernel.py#L506)。

推荐顺序：eager 专家计算 -> `moe_align_block_size_torch` -> 分桶后的索引 -> GEMM tile。
先弄清 sorted token ID 是 `token*K+k` 还是原 token ID，再追踪两段 GEMM。

读完能回答：

1. 为什么将同一专家的 token 放在一起有助于 GEMM？
2. padding sentinel、remote expert=-1 和 store mask 分别保护什么？
3. token 很少时，大 BLOCK_M 会浪费什么？大 tile 又可能节省什么？
4. 原始 BF16 权重转换为 FP16 后，为何还需要有限值、logits 和生成结果验证？

## 5. 性能分析：基线正确后再补

实践入口：[PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。
重点看 CPU/CUDA activity、`record_function`、shape 记录和 trace；
先按 router、token 分桶、GEMM1、激活、GEMM2、合并、通信划分时间线。
profile 有额外开销，正式 latency benchmark 应单独运行。

进阶论文：[MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)。
关注 token dropping、padding 浪费与 block-sparse 计算的动机。
它是 MoE 训练系统论文，不是我们的推理基线，也不是之前提到的所有 MegaMoE 工作的同义词；
先借鉴如何识别不规则专家计算的浪费，不要求复现整篇系统。

读完能回答：

1. 计算时间、通信时间、CPU 调度时间分别在哪里？能否重叠？
2. 为什么八卡吞吐、单请求 TPOT 和四卡比起来可能出现不同趋势？
3. 为什么“激活 3B”不能保证与 3B Dense 一样快？
4. 一次优化究竟减少了计算、访存、padding、launch，还是通信？证据是什么？

## 建议阅读节奏

| 次序 | 本轮只完成什么 | 用什么确认理解 |
| --- | --- | --- |
| 1 | MoE 入门 + Qwen3 Architecture | 手算 2 个 token、4 个专家、top-2 的加权输出 |
| 2 | NCCL 图 + PyTorch rank/group | 手画复制 token 的四卡 EP 数据流 |
| 3 | Megatron 第 3 节 + SwiGLU TP | 标出每个权重和通信前后张量的 shape |
| 4 | Triton matmul + Volta 指定小节 | 解释一个 GEMM program 的 load/dot/store |
| 5 | Profiler + MegaBlocks 动机 | 用性能数据判断一个实际瓶颈 |

目前不把分布式训练、ZeRO/FSDP、优化器分片、复杂 backward 或 FP8 量化作为前置要求。
学习到能够解释对应数据流即可继续 review，不必读完所有资料才开始实验。

链接核对日期：2026-10-27。Hugging Face 博客使用官方仓库文稿核对；模型卡使用已下载
并校验的固定版本核对。
