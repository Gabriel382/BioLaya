from biolaya.evaluation.metrics import classification_metrics


def test_classification_metrics():
    m = classification_metrics(["a", "b", "b"], ["a", "a", "b"], [0.9, 0.6, 0.8])
    assert m["n"] == 3
    assert 0 <= m["accuracy"] <= 1
    assert 0 <= m["ece"] <= 1
