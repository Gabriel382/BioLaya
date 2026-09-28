from biolaya.cloud.env import resolve_device


def test_cpu_device_is_always_selectable():
    assert resolve_device("cpu") == "cpu"


def test_auto_resolves_to_supported_device():
    assert resolve_device("auto") in {"cpu", "cuda"}
