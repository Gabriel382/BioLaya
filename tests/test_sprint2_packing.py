from types import SimpleNamespace

import biolaya.corpus.packing as packing
from biolaya.corpus.biomedical_stream import BiomedicalDocument
from biolaya.corpus.packing import PackedMLMDataset


class DummyTokenizer:
    sep_token_id = 99
    eos_token_id = None
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, truncation=False):
        return {"input_ids": [int(x) for x in text.split()]}


def test_packing_honors_exact_nonpadding_token_budget(monkeypatch):
    docs = [BiomedicalDocument("1", "x", "1 2 3 4 5 6 7 8 9 10", {})]
    monkeypatch.setattr(packing, "BiomedicalMixture", lambda *a, **k: docs)
    ds = PackedMLMDataset({}, DummyTokenizer(), partition="train", token_budget=7, sequence_length=4, seed=1)
    rows = list(ds)
    assert len(rows) == 2
    assert sum(sum(r["attention_mask"]) for r in rows) == 7
    assert all(len(r["input_ids"]) == 4 for r in rows)
