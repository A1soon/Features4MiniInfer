# L3：预算实验与推理指标

L3 第一版只测串行 `LLMEngine.step()`。实验变量是调度预算策略：`fixed` 保持原有行为，`aggregate` 用同一个每轮预算 `B` 约束

```text
P_t + D_t <= B
```

其中 `P_t` 是本轮真正处理的 prefill/extend token 数，`D_t` 是 decode 请求数。mixed batch 中 decode 请求的 `extend_len` 通常也是 1，因此统计时必须从 `P_t` 中排除它们。

## 启用指标

普通 engine 默认不记录指标。创建 engine 时显式传入：

```python
engine = LLMEngine(
    model="/path/to/model",
    enable_metrics=True,
    enable_overlap=False,
    enforce_eager=True,
)
```

也可以在一次 warmup 后调用 `engine.reset_metrics()`，只保留正式实验数据。公开接口是：

```python
engine.get_metrics_summary()
engine.export_metrics("results.jsonl")
```

## 运行 fixed/aggregate

在 GPU 服务器上运行。`--help` 不会加载 MiniInfer 的 CUDA/Triton 模块；真正执行时需要完整 GPU 环境。

```bash
python benchmark/bench_l3_budget.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --policy fixed \
  --num-reqs 8 --prompt-len 512 --max-tokens 32 \
  --output results/fixed.jsonl

python benchmark/bench_l3_budget.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --policy aggregate --budget 2048 \
  --num-reqs 8 --prompt-len 512 --max-tokens 32 \
  --output results/aggregate-2048.jsonl
```

建议保持模型、请求数、prompt 长度、输出长度、page size 和 GPU 不变，只改变 `policy`/`budget`。aggregate 的 `budget` 必须不小于 `max_num_seqs`，否则 decode-only batch 本身就无法满足不变量。

## JSONL 记录

文件包含三类记录（runner 还会追加一条包含 CLI 配置的 `benchmark` 记录）：

- `iteration`：本轮 mode、request ids、`prefill_tokens`、`decode_tokens`、budget、KV 可用量、retract 计数和 CPU 侧 schedule/forward/iteration 时间。
- `request`：输入长度、arrival/first-token/finish 时间、输出 token 数、TTFT、TPOT 和 ITL 列表。
- `summary`：总 prefill/decode/output token、完成请求数、吞吐、TTFT/TPOT/ITL 的 median/p95，以及预算违反次数。

指标定义：

```text
TTFT = first_token_time - arrival_time
TPOT = (finish_time - first_token_time) / (output_tokens - 1)
ITL  = 相邻输出 token 的时间差
```

只有至少两个输出 token 时才定义 TPOT；只有至少两个 token 时间点时才有 ITL。输出吞吐是正式运行期间完成的 output token 数除以 recorder 的 elapsed time。

## 计时边界与限制

第一版关闭 overlap 并开启 `enforce_eager`，目的是让一次 `step()` 对应一轮清晰的 schedule → forward/sample → process 生命周期。runner 在实验起止调用 `torch.cuda.synchronize()`，所以 wall time 可以用于端到端比较。

recorder 内部的 `schedule_ms`、`forward_ms` 和 `iteration_ms` 使用 CPU `perf_counter`，不是精确的 GPU kernel 时间；后续若需要拆分 GPU kernel，应改用 CUDA Event。第一版也不统计 overlap 专用指标、CUDA Graph 指标或 adaptive `B_t`。

## 明天的阅读顺序

1. 先读 `miniinfer/engine/metrics.py` 的 `extract_batch_counts()`，确认 mixed batch 为什么要把 decode 请求从 `extend_lens` 中排除；再读 `RequestMetrics` 和 `IterationMetrics` 的派生指标。
2. 回到 `miniinfer/engine/llm_engine.py`：看 `add_requests()`/`add_request()`/`submit_request()` 的 arrival 埋点，然后按 `step()` 中 `schedule → begin_iteration → forward/sample → finish_iteration` 的顺序对照代码。
3. 最后读 `benchmark/bench_l3_budget.py` 的 `main()`，确认 warmup、`reset_metrics()`、串行 step 循环、CUDA synchronize 和 JSONL 导出之间的边界。
4. 用 `tests/unittests/test_l3_metrics.py` 复习五个核心例子：纯 extend、纯 decode、mixed、预算违反和手工时间戳。测试中的 dummy batch 对应真实 `ScheduledBatch` 的最小字段集合。
