from biolaya.datasets.common import nli_example


def test_nli_becomes_laya_choice():
    ex = nli_example(
        row_id="x", dataset="bionli", split="test", premise="A", hypothesis="B",
        label="entailment", allowed_labels=("entailment", "contradiction")
    )
    questions = ex.laya_questions()
    assert questions["nli"]["type"] == "choice"
    assert set(questions["nli"]["criteria"]) == {"entailment", "contradiction"}
    assert ex.gold == "entailment"
