from biolaya.config import load_dapt_preset


def test_packaged_presets_exist():
    for name, budget in [("smoke", 100_000), ("1m", 1_000_000), ("10m", 10_000_000), ("100m", 100_000_000)]:
        cfg = load_dapt_preset(name)
        assert cfg["corpus"]["token_budget"] == budget
        assert cfg["corpus"]["sources"][0]["name"] == "pubmed"
        assert cfg["corpus"]["sources"][1]["name"] == "pmc"
