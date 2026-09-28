from biolaya.corpus.biomedical_stream import _is_allowed_pmc, stable_partition


def test_stable_partition_repeats():
    assert stable_partition("123456", 1) == stable_partition("123456", 1)


def test_pmc_filter_rejects_retracted_and_noncommercial_variants():
    allowed = {"CC0", "CC BY", "CC BY-SA"}
    assert _is_allowed_pmc({"license": "CC BY", "retracted": "no"}, allowed)
    assert _is_allowed_pmc({"license": "CC BY-SA", "retracted": "no"}, allowed)
    assert not _is_allowed_pmc({"license": "CC BY-NC", "retracted": "no"}, allowed)
    assert not _is_allowed_pmc({"license": "CC BY", "retracted": "yes"}, allowed)
