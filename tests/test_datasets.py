import json

from pnsearch.datasets import load_pasa


def test_load_pasa_preserves_official_splits_and_arxiv_ids(tmp_path):
    samples = {
        "AutoScholarQuery/train.jsonl": {
            "qid": "AutoScholarQuery_train_0",
            "question": "train query",
            "answer": ["Train paper"],
            "answer_arxiv_id": ["2401.00001"],
        },
        "AutoScholarQuery/dev.jsonl": {
            "qid": "AutoScholarQuery_dev_0",
            "question": "dev query",
            "answer": ["Dev paper"],
            "answer_arxiv_id": ["2401.00002"],
        },
        "AutoScholarQuery/test.jsonl": {
            "qid": "AutoScholarQuery_test_0",
            "question": "test query",
            "answer": ["Test paper"],
            "answer_arxiv_id": ["2401.00003"],
        },
    }
    for relative_path, sample in samples.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    records = load_pasa(tmp_path)

    assert [record.split for record in records] == ["train", "validation", "test"]
    assert [record.query_id for record in records] == [
        "AutoScholarQuery_train_0",
        "AutoScholarQuery_dev_0",
        "AutoScholarQuery_test_0",
    ]
    assert records[0].positive_papers == [
        {"paper_id": "2401.00001", "title": "Train paper"}
    ]
