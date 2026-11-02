# L2 调度复习笔记

本文只复习三个问题：

1. `available_prefill_budget` 表示什么？
2. 为什么当前 MiniInfer 开启 chunked prefill 后才能构造 mixed batch？
3. prefill 和 decode 长度不同，怎样放进同一次 forward？

## 1. `available_prefill_budget`

位置：`miniinfer/scheduler/prefill_adder.py:157`

```python
budgets = [self.cur_rem_tokens, self.rem_input_tokens]
if self.rem_compute_tokens is not None:
    budgets.append(self.rem_compute_tokens)
return min(budgets)
```

三个预算的含义：

```text
cur_rem_tokens       KV capacity 视角还能使用多少 token
rem_input_tokens     本轮 prefill 输入上限，受 max_extend_len 影响
rem_compute_tokens   aggregate 策略下本轮还能实际计算多少 token
```

因此该函数表达的是“当前 prefill 最多还能使用多少 token”。但它目前只是一个
辅助属性，不是实际 admission 的唯一入口，而且没有包含 `rem_chunk_tokens`。
真正的请求准入仍由 `try_add_prefill()` 和 `add_chunked_req()` 分别检查：

```text
KV budget + input budget + chunk budget + compute budget
```

### KV 与 compute 不能混淆

```text
KV budget       按 page 对齐，保护 KV cache 容量
compute budget  按实际 extend token 记账，保护本轮计算量
```

例如 `page_size=256`、实际 extend 为 257：

```text
KV 记账      = 512
compute 记账 = 257
```

## 2. chunked prefill 与 mixed batch

当前 `fixed` 没有统一的 `P_t + D_t <= B` 限制；`aggregate` 才引入：

```text
B   = max_num_batched_tokens
D_t = 本轮 decode token 数，通常等于 running 请求数
P_t = 本轮 prefill/extend token 数

P_t + D_t <= B
```

例如 `B=1024`、有 8 个 running decode 请求：

```text
本轮 prefill budget = 1024 - 8 = 1016
```

如果新请求还剩 3000 个 prompt token，不切 chunk 就只能整条等待，或者一次处理
3000 token 并违反预算。开启 chunked prefill 后，本轮可以只处理不超过 1016
token，剩余部分下一轮继续。

当前 Scheduler 还把 chunked prefill 当作 mixed batching 的模式开关：

```python
if self.enable_chunked_prefill and running_reqs:
    decode_reqs = list(running_reqs)
```

所以当前实现中：

```text
开启 chunked prefill
    -> 可以切分长 prompt
    -> 可以把 running decode 附加到 prefill batch
    -> 构造 mixed batch
```

理论上 chunked prefill 和 mixed batch 可以是两个独立开关，只是当前 MiniInfer
为了简化调度逻辑而把它们绑定在一起。即使某个短 prompt 实际没有被切块，只要
开启了该模式，它也可以和 running decode 组成 mixed batch。

## 3. 不同长度怎样组成 mixed batch？

MiniInfer 不把请求 padding 成相同长度，而是使用：

```text
扁平 token 数组 + 每个请求的长度 metadata + 独立的 KV page table
```

假设：

```text
请求 A：已有 prefix 2 token，本轮 prefill 3 token
请求 B：已有历史 5 token，本轮 decode 1 token
```

Mixed batch 可以表示为：

```text
input_ids   = [A1, A2, A3, B_decode]   # 按 token，长度 4
prefix_lens = [2, 5]                    # 按请求，长度 2
extend_lens = [3, 1]                    # 按请求，长度 2
seq_lens    = [5, 6]                    # 按请求，长度 2
batch_size  = 2                         # 请求数量
```

decode 请求在 mixed batch 中被视为一种特殊的 extend：

```text
prefix_len = 已有历史长度
extend_len = 1
```

`input_ids` 和 `out_cache_loc` 按 token 一一对应：

```text
input_ids[i] -> 该 token 的 K/V 写入 out_cache_loc[i]
```

## 4. Attention 如何恢复请求边界？

Attention backend 会收到 query offset。例如：

```text
qo_indptr = [0, 3, 4]
```

表示：

```text
请求 A 使用 input_ids[0:3]
请求 B 使用 input_ids[3:4]
```

每个请求还有自己的 `paged_kv_indices` 等 page table metadata，因此 A 只读取 A
的历史 KV，B 只读取 B 的历史 KV。token 虽然物理上被展平，不同请求之间不会
互相 attention。

`MIXED` 在当前代码中被视为 extend 模式，所以整批走 `forward_extend()`：

```text
input_ids [4]
    -> embedding / QKV / MLP 对 4 个 token 一起计算
    -> ragged attention 根据 offset 和 page table 分隔请求
    -> K/V 写入各自的 cache page
```

模型可能得到：

```text
hidden_states = [4, hidden_size]
```

但生成下一个 token 只需要每个请求最后位置的 logits：

```text
A 的最后位置 = 2
B 的最后位置 = 3

logits_for_sampling = logits[[2, 3]]
shape = [2, vocab_size]
```

## 5. 三句记忆

1. `available_prefill_budget` 是预算摘要，不是当前唯一的请求准入判断。
2. 当前 MiniInfer 用 chunked prefill 同时开启 prompt 切块和 mixed batching。
3. Mixed batch 的核心不是 padding，而是扁平 token 加长度、offset 和 KV page metadata。
