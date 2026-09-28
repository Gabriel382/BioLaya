from biolaya.config import load_dapt_preset
from biolaya.training.dapt import plan_dapt, resolve_runtime


def test_10m_plan_is_deterministic():
    plan = plan_dapt(load_dapt_preset("10m"))
    assert plan["base_model"] == "answerdotai/ModernBERT-large"
    assert plan["token_budget"] == 10_000_000
    assert plan["sequence_length"] == 512
    assert plan["packed_sequences"] == 19_532
    assert plan["optimizer_steps"] == 1_221


def test_cpu_runtime_is_explicit():
    runtime = resolve_runtime("cpu", "auto", True)
    assert runtime.device == "cpu"
    assert runtime.precision == "float32"
    assert runtime.use_cpu is True
