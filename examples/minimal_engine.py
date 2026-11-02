#!/usr/bin/env python3
"""
最小可运行的单进程 LLMEngine 示例

使用方式:
    python examples/minimal_engine.py --model /path/to/your/model

注意: model 必须是本地路径，不支持 HuggingFace Hub 直接下载

建议阅读方式:
1. 先从 main() 看起。main() 展示的是最常用的阻塞式 generate 接口：
   传入一批 prompt，等待所有请求生成完，再一次性拿到结果。
2. 再看 test_step_by_step()。它更接近推理引擎内部的执行方式：
   先把请求放入 Scheduler，然后手动一轮一轮调用 engine.step()。
3. 看完本文件后，可以按调用链继续跳转：
   LLMEngine.generate -> LLMEngine.step -> Scheduler.schedule
   -> KVCacheManager.prepare_* -> ModelRunner.forward -> Sampler。

几个入门概念:
- prompt: 用户输入的原始文本。
- token ids: tokenizer 把文本切成整数 id 后的结果，模型只认识整数 token。
- prefill: 第一次处理 prompt，把 prompt 的 KV cache 写入显存。
- decode: 每次基于已有 KV cache 生成下一个 token。
- batch: 多个请求合在一起跑一次模型，提高 GPU 利用率。
"""

import argparse
import sys
from pathlib import Path

# 这个示例文件在 examples/ 目录下，直接运行时 Python 默认只会把
# examples/ 放进 import 搜索路径。下面两行把项目根目录也加入搜索路径，
# 这样才能 import miniinfer 这个本地包。
#
# Path(__file__)        表示当前文件 examples/minimal_engine.py
# .parent               表示 examples/ 目录
# .parent.parent        表示项目根目录 MiniInfer/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# LLMEngine 是这个项目对外暴露的高级推理入口。你可以先把它理解成：
# tokenizer + scheduler + model runner + KV cache manager + detokenizer 的封装。
from miniinfer.engine.llm_engine import LLMEngine

# SamplingParams 描述“怎么采样下一个 token”，例如最多生成多少 token、
# temperature 多大、top_p/top_k 等。这里先只用 max_tokens 和 temperature。
from miniinfer.utils.sampling_params import SamplingParams


def main():
  # argparse 是 Python 标准库，用来把命令行参数解析成 args 对象。
  # 例如:
  #   python examples/minimal_engine.py --model /models/Qwen3-0.6B --max-tokens 32
  # 解析后可以通过 args.model、args.max_tokens 读取。
  parser = argparse.ArgumentParser(description="Minimal LLMEngine Example")
  parser.add_argument(
    "--model",
    type=str,
    required=True,
    # MiniInfer 当前按本地模型目录加载权重和 tokenizer。
    # 这个目录通常包含 config.json、tokenizer.json、*.safetensors 等文件。
    help="Local path to the model directory",
  )
  parser.add_argument(
    "--max-tokens",
    type=int,
    default=64,
    # max_tokens 控制“每个请求最多生成多少个新 token”。
    # 注意它不是 prompt 长度，而是输出长度上限。
    help="Maximum number of new tokens to generate for each prompt",
  )
  parser.add_argument(
    "--temperature",
    type=float,
    default=0.7,
    # temperature 越高，采样越随机；越低，输出越稳定。
    # temperature=0 通常表示 greedy decoding，即每步选概率最大的 token。
    help="Sampling temperature",
  )
  parser.add_argument(
    "--max-num-seqs",
    type=int,
    default=4,
    # max_num_seqs 是 engine 同时管理的最大请求数。
    # 对应调度器里的 batch/request pool 容量，越大越吃显存。
    help="Maximum number of sequences in a batch",
  )
  parser.add_argument(
    "--max-model-len",
    type=int,
    default=512,
    # max_model_len 是单个请求的最大上下文长度：
    # prompt tokens + generated tokens 不能超过这个上限。
    help="Maximum model context length",
  )
  args = parser.parse_args()

  print(f"Loading model: {args.model}")
  print(f"Max sequences: {args.max_num_seqs}")
  print(f"Max model length: {args.max_model_len}")
  print("-" * 50)

  # 这里放 3 个 prompt，用来演示 batch inference。
  # generate() 会把它们作为一批请求提交给 Scheduler。
  prompts = [
    "Hello, how are you?",
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
  ]

  # SamplingParams 会被复制/应用到每个 prompt。
  # 对于初学者，先关注两个参数：
  # - max_tokens: 最多生成多少 token
  # - temperature: 控制随机性
  sampling_params = SamplingParams(
    max_tokens=args.max_tokens,
    temperature=args.temperature,
  )

  # with ... as ... 是 Python 的 context manager 语法。
  # 进入 with 块时会调用 engine.start()，离开时会调用 engine.stop()。
  # 对推理引擎来说，这种写法可以集中管理模型、显存和后台资源的生命周期。
  #
  # LLMEngine 初始化时大致会做这些事：
  # 1. 读取模型配置并加载权重到 GPU/CPU。
  # 2. 根据剩余显存估算最多能放多少 KV cache token。
  # 3. 初始化 KVCacheManager、Scheduler、Tokenizer、ModelRunner。
  # 4. 如果配置允许，预热 CUDA Graph / overlap 等优化路径。
  #
  # use_multiprocess=False 表示使用单进程路径。学习代码时先看单进程最清楚。
  with LLMEngine(
    model=args.model,
    use_multiprocess=False,
    max_num_seqs=args.max_num_seqs,
    max_model_len=args.max_model_len,
  ) as engine:
    print(f"\nGenerating responses for {len(prompts)} prompts...\n")

    # generate() 是“阻塞式”高级接口：
    # - 它会先 tokenize prompt，创建 Req 对象并加入 Scheduler。
    # - 然后不断调用内部 step/step_overlap，直到所有请求完成。
    # - 最后把生成的 token ids detokenize 成文本返回。
    #
    # 如果你只是想用这个 engine，读 generate() 就够了。
    # 如果你想学 infra，下一步应该去读 LLMEngine.step() 和 Scheduler.schedule()。
    outputs = engine.generate(
      prompts=prompts,
      sampling_params=sampling_params,
      use_tqdm=True,
    )

    print("\n" + "=" * 50)
    print("Results:")
    print("=" * 50)

    # outputs 的顺序和 prompts 的顺序一致。
    # 每个 output 是一个 dict，常见字段包括：
    # - output["text"]: 生成出来的字符串
    # - output["token_ids"]: 生成出来的 token id 列表
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
      print(f"\n[Prompt {i+1}]: {prompt}")
      print(f"[Response]: {output['text']}")
      print(f"[Token count]: {len(output['token_ids'])}")
      print("-" * 50)


def test_step_by_step(model_path: str):
  """
  演示如何使用 step-by-step API 进行更精细的控制。

  main() 里的 generate() 会隐藏调度循环；这个函数把循环显式写出来，
  更适合学习 inference engine 的内部节奏。

  一次 engine.step() 通常会做一轮:
  1. Scheduler.schedule(): 决定本轮跑 prefill、decode 还是 mixed batch。
  2. ModelRunner.forward(): 执行模型前向。
  3. Sampler: 根据 logits 采样下一个 token。
  4. LLMEngine 处理输出、更新请求状态、释放完成请求的 KV cache。
  """
  print("\n" + "=" * 50)
  print("Step-by-step API Demo")
  print("=" * 50)

  # 这里不用 with，是为了演示手动 start()/stop()。
  # 真实项目里推荐优先用 with，异常时更不容易漏掉 stop()。
  engine = LLMEngine(
    model=model_path,
    use_multiprocess=False,
    max_num_seqs=2,
    max_model_len=256,
  )
  engine.start()

  try:
    # 这里两个请求共享同一个 SamplingParams。
    # add_request() 不会立刻运行模型，只是把请求加入调度器的 waiting_queue。
    sampling_params = SamplingParams(max_tokens=32, temperature=0.7)

    req_id1 = engine.add_request("What is 2+2?", sampling_params)
    req_id2 = engine.add_request("Say hello in French.", sampling_params)

    print(f"Added requests: {req_id1}, {req_id2}")

    # 手动执行 step。
    # all_outputs 用 request_id 做 key，保存每个请求最终生成的完整文本。
    step_count = 0
    all_outputs = {}

    # is_finished() 会检查 Scheduler 里是否还有 waiting/running/chunking 请求。
    # 只要还有未完成请求，就继续驱动 engine.step()。
    while not engine.is_finished():
      step_output = engine.step()
      step_count += 1

      # StepOutput 里会记录这一步处理了多少 prefill/decode token。
      # prefill token 多，说明这一轮在处理 prompt；decode token 多，
      # 说明这一轮在为已有请求各生成一个新 token。
      if step_output.num_prefill_tokens > 0:
        print(f"Step {step_count}: Prefill {step_output.num_prefill_tokens} tokens")
      if step_output.num_decode_tokens > 0:
        print(f"Step {step_count}: Decode {step_output.num_decode_tokens} tokens")

      # step_output.outputs 是本轮产生的文本增量。
      # 流式服务就是把这些 delta_text 边生成边返回给用户。
      for request_output in step_output.outputs:
        if request_output.delta_text:
          print(
            f"  [{request_output.request_id}] delta={request_output.delta_text!r}"
          )

        if request_output.finished:
          all_outputs[request_output.request_id] = request_output.full_text
          print(
            f"  -> Request {request_output.request_id} finished, "
            f"reason={request_output.finish_reason}"
          )

    print(f"\nTotal steps: {step_count}")
    print("\nOutputs:")
    for request_id, text in all_outputs.items():
      print(f"  {request_id}: {text}")

  finally:
    # finally 表示无论 try 里的代码是否报错，都会执行 engine.stop()。
    # 这对释放模型对象、显存和其他资源很重要。
    engine.stop()


if __name__ == "__main__":
  # Python 文件被直接执行时，__name__ 会等于 "__main__"。
  # 被其他文件 import 时，这里的 main() 不会自动运行。
  main()

  # 如果想测试 step-by-step API，取消下面的注释并提供模型路径
  # test_step_by_step("/path/to/your/model")
