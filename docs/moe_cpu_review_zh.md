# MoE 本地开发与 Review

本轮完成无需 CUDA 的 MoE reference、权重分片加载、并行配置和四进程验证。
V100 GPU 单卡测试通过了 attention、MoE Triton kernel 和真实 rank-local
FP16 权重加载；这些结果仍不能替代 NCCL、四卡 EP forward 或完整多卡推理引擎验证。

## 阅读顺序

1. [独立 reference](../tests/moe_reference.py)：按 token、按选中专家计算，
   使用 FP64 中间值；不调用被测路由、分桶或专家实现。
2. [MoE 主流程](../miniinfer/layers/fused_moe/layer.py)：router、top-k、
   本地专家、输出归约。既有 CPU 专家实现与上面的 oracle 进行对齐。
3. [并行组](../miniinfer/layers/fused_moe/dispatcher.py)：配置检查、
   专家范围、正交 TP/EP 组、实际组大小与 rank 检查。
4. [流式加载](../miniinfer/loader/moe.py)：从标准 HF checkpoint 直接填入
   本地模型；先检查索引键集合，逐个核对 shape，再读取 tensor 或 TP 切片并转换 dtype。
5. [四进程测试](../tests/unittests/test_moe_distributed_cpu.py)：四个独立进程
   通过 Gloo broadcast/all-reduce/all-gather 验证三种布局。

## 并行契约

全局 rank 按 `global_rank = ep_rank * tp_size + tp_rank` 排列。所有 rank 都必须按相同
顺序创建各个子组。以 `TP=2, EP=2` 为例：

| 全局 rank | EP rank | TP rank | 专家范围（E=8） | 专家 intermediate 切片（I=24） |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0–3 | 0–11 |
| 1 | 0 | 1 | 0–3 | 12–23 |
| 2 | 1 | 0 | 4–7 | 0–11 |
| 3 | 1 | 1 | 4–7 | 12–23 |

TP 组为 `[0,1]`、`[2,3]`，EP 组为 `[0,2]`、`[1,3]`。当前实现先对 EP 求和，再对 TP
求和；两个归约覆盖互不重叠的专家/矩阵贡献。某个 EP rank 没有专家命中时输出零，
仍参与同样的 collective。

可以在已经初始化的分布式进程内调用：

```python
parallel = create_moe_parallel_config(tp_size=1, ep_size=4)
model = Qwen3MoeForCausalLM(config, device="cpu", precision=torch.float32, parallel=parallel)
report = load_qwen3_moe_weights(model, checkpoint_dir)
```

这里得到 MoE 层可用的并行布局；Attention 仍保持完整投影。它不是一个新的多卡服务入口，
引擎 worker、调度和真实采样同步需要在后续阶段接入。

## 权重加载契约

加载器接受当前 Qwen3-MoE 的标准 HF per-expert `gate_proj/up_proj/down_proj` 格式。
共享 Attention、router、norm、embedding 复制；专家按 EP 切编号，按 TP 切 intermediate。

- Q/K/V 直接拷到融合 QKV 参数的各自区间，不先在 GPU 上构造整份融合 state_dict。
- gate/up 分别写入本地 w13 的两个半段；down 沿输入特征维读取对应 TP 切片。
- 不属于当前 EP rank 的专家在读取数据之前就被排除。
- 不启动每个 rank 对所有 checkpoint 分片的整文件预读取。
- 分片索引重复键、缺少权重、未知权重、错误 shape、非有限数值和 FP16 溢出会明确失败。
- tie_word_embeddings 模型支持缺省 lm_head，并复用 embedding 存储。
- 兼容有索引的多分片 checkpoint，以及无索引的 safetensors 文件。

MoE ModelRunner 已切换到此加载器，不再为这条路径保存完整 `state_dicts`。
`MoELoadReport` 记录 tensor 数、所读 tensor/切片字节数和最大单 tensor 大小；
这些数值不是对实际进程峰值 RAM 或 GPU 显存的测量，后者仍需实际运行测量。
仅部分权重装入后发生错误时，应丢弃该模型实例。

## CPU 环境和执行

隔离环境位于仓库根目录下：

`.venv-moe-cpu`

CPU 验证使用 Python 3.12 / PyTorch 2.6 / Gloo；CUDA 验证使用各 GPU 节点已有的兼容环境。
依赖列表见 [moe-cpu.in](../requirements/moe-cpu.in)。需要新建环境时：

```bash
uv venv --python 3.12 .venv-moe-cpu
uv pip install --python .venv-moe-cpu/bin/python -r requirements/moe-cpu.in
MOE_CPU_PYTHON=.venv-moe-cpu/bin/python bash scripts/run_moe_cpu_validation.sh
```

复现命令：

```bash
MOE_CPU_PYTHON=.venv-moe-cpu/bin/python bash scripts/run_moe_cpu_validation.sh
```

顶层 MiniInfer 改为按需导出；CPU 导入不再拉入 CUDA/Triton kernel。需要提前加载包时仍可
显式设置 `MINIINFER_EAGER_IMPORT=1`。GPU kernel 在实际 CUDA 分支导入，未修改其运算实现。
CPU 测试禁用 CUDA，使用小型合成 checkpoint，不下载模型，不占用服务器 GPU。

## 已验证与待验证

2026-10-27 macOS 本地结果：

- 45 项新增 CPU 测试通过，无失败、无跳过；包括 4 个真实进程的 EP=4、TP=4、TP=2/EP=2。
- 17 项已有 prefill budget / metrics CPU 测试通过。
- V100 GPU：attention 与 MoE CUDA 测试合计 95 passed、14 warnings。
- 同卡对真实 Qwen3-30B-A3B checkpoint 加载 EP rank 0：5043 tensors loaded、13824 skipped；FP16 destination 为 17,577,701,376 bytes，peak allocated 为 22,979,227,648 bytes。该检查没有执行模型 forward。
- 已将真实 30B checkpoint 索引与四卡 EP 的 meta 模型目标进行对照，不读取真实权重数据。

独立 CPU 环境（PyTorch 2.10.0+cu126）在未挂载 GPU 的容器中也完成 45 项 CPU 测试，
全部通过；另有 22 项 prefill budget、metrics、decode input 和 overlap 顺序回归通过。
CPU 容器限制为 4 个 CPU / 4GiB 内存。

真实模型索引对照结果：四卡 EP 的 rank 0 需要 5043 个 tensor，checkpoint 共 18867 个
tensor；其余是远程专家权重，未发现缺少或未知键。此项检查不证明真实 tensor 数值正确。

四进程测试覆盖 checkpoint 分片加载、输入广播、专家输出归约、各 rank 输出一致性、
集中路由导致其他 EP rank 无本地 token，以及由 rank 0 广播一个选择结果。
最后一项只验证同步方式，不代表引擎 Sampler 已完成多卡接入。

逐次运行日志、环境快照和模型 checkpoint 属于本地实验产物，不随源码仓库发布；本文件记录
验证范围与汇总结论。复现实验会在运行目录中生成新的日志。

待四张同组 GPU 空闲后完成：1 MiB NCCL all-reduce 与大 batch MoE 定向诊断；真实模型
四卡引擎 forward 上曾出现 all-reduce 超时，需定位 rank/layer divergence 后再验证请求/采样
同步、prefill/decode/mixed 及性能。基础 NCCL 四类 collectives 和小型四 rank MoE all-reduce
reference 已通过，不能据此声称真实模型引擎端到端通过。

路由测试以 FP64 参考计算 top-k 边界；当第 k 与第 k+1 专家概率间隔小于 `1e-4` 时，
FP16 GEMM 的舍入可能改变专家次序。MoE kernel 数值对齐因此固定被测路由和权重，另对
边界稳定的 token 校验路由选择，避免把近似 tie 的路由差异误报为 grouped GEMM 错误。
