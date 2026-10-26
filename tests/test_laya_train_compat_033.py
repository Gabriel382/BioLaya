import importlib.util
from pathlib import Path
import pytest


def load_launcher():
    spec = importlib.util.spec_from_file_location("biolaya_systemone_launcher", Path(__file__).resolve().parents[1] / "scripts" / "train_systemone.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_missing_cli_explains_dependency(monkeypatch):
    m = load_launcher()
    monkeypatch.setattr(m.shutil, "which", lambda name: None)
    monkeypatch.setattr(m, "version", lambda name: "0.3.20")
    with pytest.raises(RuntimeError, match="laya==0.4.15"):
        m.build_command({"training": {}, "data": {"train": "train.jsonl"}, "model": {"base":"x", "output":"y"}})


def test_expected_flags_preserved(monkeypatch):
    m = load_launcher()
    monkeypatch.setattr(m.shutil, "which", lambda name: "/tmp/laya-train")
    command = m.build_command({"training": {"epochs":1, "loss":"soft-ce", "max_len":1024}, "data": {"train":"my.jsonl", "dev":"my-dev.jsonl"}, "model": {"base":"convaiinnovations/laya", "output":"checkpoint"}}, dry_run=True)
    assert command[0] == '/tmp/laya-train'
    assert '--dry-run' in command
    assert '--eval-data' in command
    assert 'my-dev.jsonl' in command
    assert '--loss' in command and 'soft-ce' in command
