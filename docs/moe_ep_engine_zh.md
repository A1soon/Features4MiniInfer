# MiniInfer 四卡 EP 引擎接入

本轮把 Qwen3-MoE 的专家并行从算子级验证接到引擎执行边界。实现位于
`miniinfer/engine/expert_parallel.py`，默认采用 eager、无 overlap 的第一版基线。

## 执行模型

- rank 0 独占 Scheduler、RequestPool、RadixCache、detokenizer 和采样。
- rank 1..N-1 只加载本 rank 的专家，执行 rank 0 下发的同一批 Attention 和 token。
- Attention 权重在各 rank 复制；MoE 专家按 EP rank 切分，专家输出通过已有 EP
  process group 做 all-reduce。
- Pipe 只传 CPU 控制元数据和采样 token，模型计算和 all-reduce 使用同一个进程组。
- 每个 worker 都要在 `PREPARE` 阶段确认批次合法，所有 rank 准备完成后才进入
  `RUN`，避免一侧参数错误导致另一侧卡在 collective。

## 批次与生命周期

`BatchPacket` 携带 forward mode、请求 ID、RequestPool 行、输入/位置、序列长度、
本轮写入槽位和完整物理 KV 映射。因此 prefix、chunked prefill、mixed batch 和
decode 都使用 rank 0 的物理映射；worker 不独立分配 KV。

一次请求的控制顺序是：

`READY -> INIT_KV -> PREPARE -> RUN -> SAMPLE -> COMMIT -> RELEASE -> STOP`

只有 rank 0 调用 `sample`。chunked prefill 的 `COMMIT` 不追加采样 token，普通
decode/prefill 则由 rank 0 的请求历史作为权威状态，再同步给所有 worker。

## 容量协商与失败处理

每个 rank 在模型加载后报告可用 KV 容量，rank 0 取最小值并按 page 对齐；不足两页
时在分配前失败。worker 启动、Pipe 命令和 distributed collective 均有超时边界。
只终止本 executor 创建的子进程，初始化或运行失败后会关闭进程组、join，必要时
强制结束子进程。

## 当前限制

- 第一版要求 `tensor_parallel_size=1`、`enforce_eager=True`、`enable_overlap=False`。
- 使用 `use_multiprocess=True` 的 tokenizer/scheduler ZMQ 管线暂不与 EP 嵌套。
- CUDA Graph 默认关闭。`enable_ep_cuda_graph=True` 开放已验证的 decode bucket；所有 rank
  同步 capture/release 并按相同 bucket 顺序运行。decode parity 可用
  `MINIINFER_EP_GRAPH_VERIFY=1` 检查。V100 paged backend 的静态 bucket prefill/mixed full-forward
  graph 已实现，默认关闭，最多支持两个真实请求，默认 buckets 为 256/512 tokens；支持
  prefix/page-table 元数据更新，超 bucket/请求数回退 eager。真实 Qwen3-30B EP4 已完成
  256/512 buckets capture/replay、prefill/decode/chunked mixed/shared-prefix smoke；Graph 与
  同 padding eager logits 完全一致，已测 top-1 与 unpadded eager 一致，四 rank teardown 正常。
  但 unpadded eager 的 strict logits parity（`rtol=atol=1e-3`）仍失败，最大实测绝对差
  `1.359375`，没有放宽容差。FP16 shape/reduction rounding 是相关因素，唯一根因未确定；真实
  模型 FP32 reduce 尝试也未消除差异。最终 smoke `status=PASS` 只表示其静态-padding parity、
  top-1 和执行流程断言通过，不代表 strict logits parity。Graph 仍默认关闭，性能未测。

## 证据

`tests/unittests/test_ep_engine_cpu.py` 使用四个真实 Gloo 进程、生产模型加载器、
生产 MoE forward/sampler 和 CPU attention adapter，覆盖完整模型 logits 对齐、
共享 prefix、mixed、decode、chunked commit、请求行复用、采样一次性、worker
失败/超时和 KV 容量协商。

`tests/unittests/test_cuda_graph_plan.py` 覆盖 Graph bucket 和上限边界；GPU/NCCL
正确性仍须在取得四张同组空闲 GPU 后执行。

单卡 GPU 记录：2026-10-27 在 V100 物理 GPU 2、PyTorch 2.10/CUDA 12.6 环境中，
`test_flashattention_v100.py` 与 `test_moe_cuda.py` 合计 95 项通过；另以真实
Qwen3-30B checkpoint 将 rank 0 的 32 个专家和复制权重流式加载为 FP16，5043 个 tensor
加载、13824 个 tensor 跳过，峰值 CUDA allocated 约 22.98 GB。没有运行真实模型 forward、
NCCL 或多 rank collective。

2026-10-28 四张 V100 多卡记录：基础与 1 MiB NCCL collectives 通过；四 rank 合成
Qwen3-MoE kernel/reference（平衡、rank0-only、大消息）通过。真实 30B eager EP4 的
prefill/decode/采样、共享 prefix 与 chunked mixed smoke 通过。真实 EP decode CUDA Graph
bucket 1 通过同 batch eager parity，最大绝对误差 0，三 token 请求完成并正常 teardown。
V100 静态 bucket prefill/mixed full-forward graph 已通过真实四卡 capture/replay、same-padding
logits parity、top-1 对照及 prefill/decode/mixed/prefix smoke；strict unpadded logits parity
仍未通过，未测 Graph 性能，功能默认关闭并需显式 opt-in。完整记录在实验目录
`logs/multi-gpu-20261028/RESULTS.zh-CN.md` 和 `logs/prefill-graph-20261029/README.zh-CN.md`。
