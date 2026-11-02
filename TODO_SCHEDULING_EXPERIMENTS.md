# MiniInfer 调度实验 TODO

本文档跟踪两个相互补充的工作：

1. Decode-aware adaptive prefill budget：通过共享的 iteration token budget 控制实际 chunk，尽量减少 decode 抖动和 retract。
2. Retract state preservation and cache-aware resume：retract 后保留请求逻辑状态，并尽量复用仍存活的 Radix KV page。

最终目标不是只得到一个“更快”的数字，而是形成一套可复现的正确性测试、调度策略对比和性能解释。

## 当前 L3 第一版交付边界

本地第一版已经完成公共观测基础的最小闭环：串行 `LLMEngine.step()` 的 iteration/request 埋点、TTFT/TPOT/ITL/输出吞吐汇总、fixed/aggregate runner、JSONL 导出和 CPU-only 测试。对应代码见 `miniinfer/engine/metrics.py`、`benchmark/bench_l3_budget.py`、`tests/unittests/test_l3_metrics.py` 和 `docs/l3_metrics_guide.md`。

本轮明确不包含 adaptive `B_t`、overlap/CUDA Graph 专用计时和完整 differential correctness；这些仍按本文后续 M2-M8 顺序推进。GPU 上的 benchmark 仍需先通过 V100/4090 的 backend smoke test。non-overlap 的 retract 逻辑状态保留第一版已经落地：保留 `output_ids`，用 `kv_committed_len` 区分已采样但尚未经过 forward 的 token，并在恢复时通过 Radix prefix lookup 复用可存活的 KV。

## 0. 实验原则

- [ ] 所有策略保留配置开关，可以随时切回当前实现作为 baseline。
- [ ] correctness 先于 performance；greedy 输出不一致时不做性能结论。
- [ ] 第一版使用单进程、`temperature=0`、`enable_overlap=False`、`enforce_eager=True`。
- [ ] 基础语义验证完成后，再依次开启 overlap、CUDA Graph 和随机采样。
- [ ] compute budget 和 KV capacity budget 分开：`P + D <= B` 约束本轮计算量，PrefillAdder/KV allocator 继续保证 KV 不超额。
- [ ] 固定模型、软件版本、随机种子和 workload；记录 GPU 型号、显存、CUDA、PyTorch 和 kernel 版本。

## 1. V100 硬件门禁

V100 是 Volta（SM70），支持 FP16 Tensor Core，但不支持 BF16。当前上游 FlashInfer 的支持范围从 SM75 开始，FlashAttention2 的主 CUDA 路径面向 Ampere/Ada/Hopper；因此 MiniInfer 现有的两个 attention backend 都不能作为 V100 可用的前提。先验证远端环境中实际安装的版本，但不要把反复尝试依赖版本当作主方案。

### 1.1 先做兼容性检查，不先改框架

- [ ] 记录 `nvidia-smi`、CUDA driver、GPU 显存是 16 GB 还是 32 GB。
- [ ] 记录 `torch.__version__`、`torch.version.cuda` 和 `torch.cuda.get_device_capability()`。
- [ ] 使用 FP16 和 Qwen2-0.5B 或 Qwen3-0.6B，先运行单请求、短输出 smoke test。
- [ ] 首先测试默认 FlashInfer backend。
- [ ] 若失败，记录是 wheel/编译阶段拒绝 SM70，还是运行 kernel 时失败。
- [ ] 再测试项目的 FlashAttention backend；不要为了“跑起来”盲目更换多个依赖版本。
- [ ] 单独 smoke test RMSNorm、RoPE、SiluAndMul 等 Triton kernel，确认不是只有 attention backend 不兼容。
- [ ] smoke test 通过后，再开启 chunked prefill，运行 prefill、decode 和 mixed batch 各一个最小样例。

### 1.2 兼容性决策

- [ ] 如果现有 backend 可用：不做硬件适配，直接进入实验。
- [ ] 如果 attention backend 不支持 SM70：评估增加一个 correctness-first 的 PyTorch/SDPA fallback backend。
- [ ] fallback 只需要支持 FP16、prefill、decode、mixed batch 和现有 paged KV 映射；第一版不支持 CUDA Graph。
- [ ] 如果多个核心 Triton kernel 也不支持 SM70：停止 V100 适配，优先换 T4/A10/A100 等受支持 GPU，避免硬件兼容工作吞掉项目主线。
- [ ] 即使使用 fallback，所有 A/B/C 策略必须使用同一 backend；结论只解释为 V100 上的相对结果。

硬件门禁完成标准：单请求能稳定生成，greedy 输出可与 Hugging Face 对齐，然后 mixed batch 能连续运行且没有非法访存。

## 2. 公共正确性与观测基础

两个实验先共用同一套 instrumentation 和 workload runner。

### 2.1 每个 iteration 记录

- [ ] iteration id、batch mode（prefill/decode/mixed）和 batch size。
- [ ] `D_t`：本轮 decode token 数。
- [ ] `P_t`：本轮实际 prefill/extend token 数。
- [ ] `B_t`：本轮 aggregate token budget。
- [ ] 每个请求的 remaining prompt、requested chunk 和 actual chunk。
- [ ] KV physical free、Radix evictable、Radix protected 和实际分配量。
- [ ] retract 请求数、request id、释放 token/page 数和重复 retract 次数。
- [ ] forward wall time；计时边界使用 `torch.cuda.synchronize()` 或 CUDA event。

### 2.2 请求级指标

- [ ] TTFT、每个 token timestamp、TPOT/ITL、E2E latency。
- [ ] input/output token 数和最终 token ids。
- [ ] prefill waiting time、decode waiting time和完成原因。
- [ ] restored cached tokens、recomputed tokens 和 resume latency。

### 2.3 固定 workload

- [ ] 短 prompt/长 decode：观察 TPOT 和 throughput。
- [ ] 长 prompt/短 decode：观察 TTFT 和 chunk 效果。
- [ ] 长 prompt 与 active decode 共存：核心 mixed workload。
- [ ] burst arrival 与稳定到达率两种模式。
- [ ] warmup 不计入结果；每组至少重复 3 次，报告 median 和 p95。

## 3. Idea A：Decode-aware adaptive prefill budget

### 3.1 A：保留当前 fixed 策略作为 baseline

- [ ] 增加明确的 scheduler policy 配置，例如 `fixed`、`aggregate`、`adaptive`。
- [ ] `fixed` 必须保持当前 `chunked_prefill_size`、`max_extend_len` 和 PrefillAdder 行为不变。
- [ ] 对固定 chunk 值做 sweep，例如 512、1024、2048、4096；最终范围按 V100 smoke test 调整。
- [ ] 验证只切换 policy、不改变请求输入时，greedy 最终 token 完全一致。

### 3.2 B：实现 vLLM 风格 aggregate iteration budget

定义：

```text
D_t = 本轮所有 decode 请求消耗的 token 数，通常每个请求为 1
P_t = 本轮所有 prefill/extend 请求实际处理的 token 数
B   = max_num_batched_tokens

P_t + D_t <= B
```

- [ ] 在配置中补全并贯通 `max_num_batched_tokens`。
- [ ] 每轮先计算 decode 的计算预算 `D_t`，再把 `B - D_t` 给所有 prefill 请求共享。
- [ ] `chunked_prefill_size` 保留为可选的 per-request safety cap，而不再代表整轮共享预算。
- [ ] actual chunk 同时受以下条件限制：

```text
min(
  request remaining prompt,
  iteration remaining prefill budget,
  KV admission budget,
  max_extend_len,
  optional per-request cap
)
```

- [ ] 保留 page 对齐和最后一个 chunk 的边界语义。
- [ ] 明确多个长 prompt 如何共享剩余预算，第一版沿用队列顺序，暂不引入复杂公平策略。
- [ ] 保证 active decode 不会因为新 prefill 被排除在 mixed batch 之外。
- [ ] 增加不变量断言：`P_t + D_t <= B`、分配 page 不超额、请求不会同时存在于多个队列。

### 3.3 C：实现 latency-aware 动态 `B_t`

第一版使用有界反馈控制，不做强化学习或复杂预测器。

- [ ] 配置 `min_budget`、`max_budget`、`initial_budget`、`adjust_interval` 和 `target_p95_tpot`。
- [ ] 最近窗口 p95 TPOT 超过目标时，减小 `B_t`。
- [ ] TPOT 低于目标且 prefill queue 持续积压时，增大 `B_t`。
- [ ] 没有 active decode 时，允许直接使用较大的 prefill budget。
- [ ] 增加上下界、步长限制和 hysteresis，避免 budget 每轮振荡。
- [ ] 记录每次调整的原因和调整前后数值。
- [ ] 将 retract rate 作为保护信号：连续 retract 时立即收紧 prefill budget。

### 3.4 动态 budget 实验

- [ ] A：current fixed chunk。
- [ ] B：固定 aggregate budget，至少测试 3 个 `B`。
- [ ] C：latency-aware adaptive `B_t`。
- [ ] 对比 TTFT median/p95、TPOT median/p95、throughput、retract rate 和 GPU utilization。
- [ ] 分析 Pareto frontier，不用单一吞吐数字宣布“全面更优”。

Idea A 完成标准：在 mixed workload 下，策略能按预期改变 `P_t/D_t`，保持 token correctness，并给出 TTFT、TPOT、throughput 三者之间可复现的权衡。

## 4. Idea B：保存 retract 请求状态并进行 cache-aware resume

### 4.1 先定义请求状态不变量

- [x] `output_ids` 是请求逻辑历史；retract 后不得清空。
- [x] 新增 `kv_committed_len`，表示已经完成 forward、KV 内容有效的 token 前缀长度。
- [x] 区分“已经采样”与“KV 已提交”，不能从 `len(output_ids)` 猜测 committed boundary。
- [ ] 保留 `max_tokens` 计数、EOS/finished 状态和增量 detokenizer 状态。
- [ ] 将 `ever_retracted` 与“需要丢弃旧 batch stale result”的一次性状态分开。
- [ ] 明确 `req_pool_idx`、`prefix_indices`、`last_node` 等是可重建的 KV 元数据，retract 后应重置。

### 4.2 建立 forced-retract correctness test

- [ ] 第一版关闭 overlap 和 CUDA Graph，使用 greedy decoding。
- [ ] A：足够 KV、不发生 retract，作为 oracle。
- [ ] B：人为缩小 KV pool，在指定 iteration 强制 retract。
- [ ] 比较完整 token ids、流式 delta 拼接结果、EOS 和 `max_tokens` 结束位置。
- [ ] 覆盖 page 边界 `page_size - 1`、`page_size`、`page_size + 1`。
- [ ] 覆盖 prefix cache 开/关、缓存存活/强制驱逐、一次/重复 retract。
- [ ] 覆盖纯 decode、mixed batch 和 chunked prefill 后进入 decode。
- [ ] 在修复前让测试稳定复现当前行为，保存失败样例作为回归证据。

### 4.3 修复 logical state preservation

- [x] retract 时保留 `origin_input_ids + output_ids`。
- [x] `release_request` 只读取并处理 `kv_committed_len` 范围内有效的 KV mapping。
- [x] Radix Cache 只接管 committed prefix 中的完整 page。
- [ ] partial page 正确释放，不能产生 dangling page reference 或 double free。
- [x] 请求回到 waiting queue 后，用完整 token 历史重建 `fill_ids`。
- [x] prefix lookup 只复用恢复语义允许的前缀；最新需要作为输入的 token 必须进入 forward。
- [ ] Radix miss/eviction 时正确退化为 recompute，不影响最终输出。

### 4.4 overlap 生命周期单独处理

- [ ] 为 scheduled batch/result 增加 generation/epoch，或使用等价的一次性 stale-result 标记。
- [ ] retract 不能误删已经对客户端可见的旧输出。
- [ ] 旧 in-flight result 被丢弃时，不能影响恢复后的新 batch result。
- [ ] KV page 释放必须等待仍可能写入这些 page 的 GPU work 完成。
- [ ] non-overlap 测试全部通过后，才开启 overlap 回归测试。

### 4.5 统计 cache-aware resume 收益

- [ ] retract 时记录 logical history tokens、committed tokens、Radix inserted full-page tokens。
- [ ] resume 时记录 Radix hit tokens、recomputed tokens 和 cache survival ratio。
- [ ] 比较以下策略：

```text
A. no-retract oracle
B. 当前 MiniInfer retract
C. preserve state + Radix resume
D. preserve state + full recompute（is_insert=False，correctness control）
```

- [ ] 对比最终 token correctness、resume latency、E2E latency、TPOT、throughput 和 retract 次数。
- [ ] 解释 Radix page 被立即驱逐时为什么 C 会退化接近 D。

### 4.6 可选后续，不进入第一版

- [ ] cost-aware victim selection：综合可释放 page、预计重算 token、priority 和重复 retract 次数。
- [ ] 防止同一个请求反复 retract/resume 的 anti-thrashing 机制。
- [ ] CPU KV offload，仅在测量 `T_recompute` 与 `T_D2H + T_H2D` 后决定是否实现。
- [ ] 不在第一版实现 NVMe/远端 KV、分布式 prefill/decode 或复杂优先级策略。

Idea B 完成标准：forced retract 与 no-retract oracle 的 token/流式输出完全一致；Radix page 存活时重算 token 明显减少，page 被驱逐时能够正确回退。

## 5. 两个 Idea 的联合实验

动态 prefill budget 是预防层，cache-aware resume 是恢复层：

```text
admission / aggregate budget
  -> 尽量减少 KV 压力和 retract
  -> 若仍发生 retract
  -> 保存逻辑状态并尝试 Radix resume
  -> cache miss 时正确 recompute
```

- [ ] 固定 retract 策略，对比 fixed/aggregate/adaptive budget 的 retract rate。
- [ ] 固定 scheduler policy，对比 full recompute/Radix resume 的恢复成本。
- [ ] 最后组合 adaptive budget + Radix resume，确认收益不是只来自 workload 偶然性。
- [ ] 分开报告“避免了多少次 retract”和“每次 retract 少重算多少 token”。
- [ ] 至少保留一个高压 workload，使所有策略都发生一定数量 retract，否则无法比较恢复策略。

## 6. 推荐执行顺序

- [ ] M0：V100 smoke test 与 backend 决策。
- [ ] M1：公共 iteration/request metrics 和可复现 workload。
- [ ] M2：forced-retract differential test，稳定复现状态丢失。
- [ ] M3：non-overlap logical-state 修复与 full-recompute fallback。
- [ ] M4：Radix resume、page-boundary/eviction 测试和收益统计。
- [ ] M5：fixed aggregate budget（`P + D <= B`）。
- [ ] M6：latency-aware adaptive `B_t`。
- [ ] M7：恢复 overlap/CUDA Graph，完成联合实验。
- [ ] M8：整理设计文档、结果表、关键时序图和简历描述。

## 7. 最终交付物

- [ ] 一个 forced-retract differential correctness test suite。
- [ ] 三种 prefill policy：fixed、aggregate、adaptive。
- [ ] 正确的 retract state machine 和 Radix resume fallback。
- [ ] 一份固定环境、固定 workload、可重复运行的 benchmark 脚本。
- [ ] CSV/JSON 原始结果以及 TTFT、TPOT、throughput、retract/recompute 图表。
- [ ] 一份说明 trade-off、失败场景和硬件限制的实验报告。
