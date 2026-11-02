def test_bench_simple_is_import_safe():
  # Import should not run the benchmark (no top-level side effects).
  import importlib

  mod = importlib.import_module("benchmark.bench_simple")
  assert hasattr(mod, "main")


def test_l3_benchmark_tools_are_import_safe():
  import importlib

  for name in (
    "benchmark.bench_l3_budget",
    "benchmark.run_l3_sweep",
    "benchmark.analyze_l3_sweep",
  ):
    mod = importlib.import_module(name)
    assert hasattr(mod, "main")
