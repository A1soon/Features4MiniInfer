# V100 Prefill/Mixed Static-Bucket CUDA Graph

状态：真实 Qwen3-30B EP4 上已完成 256/512 bucket capture/replay、same-padding eager logits
parity、top-1 对照和 prefill/decode/chunked-mixed/shared-prefix smoke，四 rank 正常 teardown。
与 unpadded eager 的逐元素 strict parity（`rtol=atol=1e-3`）仍未通过，Graph 默认关闭；没有
性能或服务模式结果。详细原始记录见实验目录 `logs/prefill-graph-20261029/README.zh-CN.md`。

10-29 最新 run 的 Graph 对同 padding eager 最大绝对误差为 `0.0`，所有已测 top-1 相同；
padded 与 unpadded eager 的最大观测误差为 `1.359375`。FP16 reduce 长度敏感性探针显示
shape-dependent rounding，但真实模型 FP32 reduce 对照仍有差异，因此尚不能把它认定为唯一
根因。最终诊断 run 的 smoke `status=PASS` 不表示 unpadded strict logits parity 通过。

## 实现边界

当前 runner 捕获整段模型 forward，不是把 attention、MoE 切成各自独立的 graph/eager
segments。仅用于 `flash_attn_v100`，通过 `MINIINFER_PREFILL_CG=1` 显式开启；默认 eager。
EP 每个 rank 同步捕获，并在初始化后核对各 rank 的 decode/prefill bucket 签名。

- 最多两个真实请求，默认 token buckets 为 256 和 512；由
  `MINIINFER_PREFILL_CG_MAX_SEQS` 与 `MINIINFER_PREFILL_CG_MAX_TOKENS` 限制。
- 每个 bucket 使用固定的两个真实请求槽和一个 padding 槽。按 query lengths 构造
  `cu_seqlens_q` 与 last-token indices，输出只包含各请求最后 token 的 logits。
- 请求数或 query layout 不能安全填充、超过 bucket、sequence/page table 超容量时返回
  eager fallback。Prefix 命中和 mixed metadata 都根据当前真实 page table/长度更新，不使用
  capture 时冻结的请求状态。
- `page 0` 保持 KV allocator 的 dummy/padding 页；长 padding 行使用从 allocator 单独保留的
  scratch pages，避免写入真实请求的 live KV slots。
- 可设置 `MINIINFER_PREFILL_CG_VERIFY=1`，在 graph replay 前将同一 ForwardBatch eager logits
  快照到 CPU，再对比 graph 结果。CPU snapshot 可避免 graph 后续执行期间 CUDA allocator
  复用 eager 输出的底层 GPU storage。

## 测试与风险

`tests/unittests/test_cuda_graph_plan.py` 覆盖 bucket、固定行数的 padding layout 以及 V100
extend metadata tensor 引用。对应 EP/Graph CPU 回归通过；新增动态 query/KV attention Graph
探针也已在 V100 通过。10-28 首轮非法访存及 10-29 workspace 生命周期修复的全过程保留在
`logs/prefill-graph-20261028/` 与 `logs/prefill-graph-20261029/`。

10-29 的真实 smoke 覆盖 300-token prefill、7-token decode、2 个 mixed steps 和 256-token
shared-prefix hit。Graph 与同 padding eager 的 logits exact match；与 unpadded eager 的 top-1
一致，但 logits strict parity 不满足原容差。`1046-reduce-padding` 的 FP16 shape probe 显示
max abs 差 `0.0078125`，而真实模型尝试 FP32 EP reduce 仍未闭合差异。结果提示数值路径依赖
shape/reduction，但唯一来源尚未隔离。没有放宽容差，也没有以 top-1 替代 strict logits parity。

本轮没有性能测试、更大 decode bucket 或服务模式覆盖。保持功能默认关闭。若目标要求 unpadded
逐元素 parity，后续应统一 eager 与 graph 的 collective shape/运算路径后再做 GPU 对照；不能
只将 padded eager 设为参考来声称 strict parity。

捕获图内仍包含 V100 varlen paged attention、MoE Triton 路由/grouped GEMM 及 EP all-reduce。
Capture/replay 期间各 rank 必须保持相同 bucket、query layout 和 collective 顺序。
任何 capture 失败在 EP 下都应让初始化失败并清理 worker，不允许某些 rank 静默 eager、其它
rank replay graph。单进程模式可跳过失败 bucket，后续 eager fallback。

MoE 的 `_workspace` 是 capture pool 以外的外部 tensor；CUDA Graph 不自动延长其生命周期。
decode `CapturedGraph.moe_workspaces` 和 prefill buffers 必须保留这些对象，即使后来 warmup
更大 bucket 或 eager fallback 触发扩容。销毁前显式 reset 图句柄，避免异常栈持有 graph
对象时 NCCL teardown 等待；61 项 CPU 回归及四 rank 多 bucket 扩容探针已通过。
