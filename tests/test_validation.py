from pathlib import Path

import pandas as pd

from datagov_profiler.validation import create_validation_sample, score_validation


def test_validation_sample_and_scoring(tmp_path: Path) -> None:
    clusters = tmp_path / "clusters.csv"
    sample = tmp_path / "sample.csv"
    scores = tmp_path / "scores.md"
    pd.DataFrame(
        [
            {"cluster_id": 1, "canonical_term": "traffic crash", "observed_term": "road accident", "examples": "Example A"},
            {"cluster_id": 2, "canonical_term": "download url", "observed_term": "file link", "examples": "Example B"},
        ]
    ).to_csv(clusters, index=False)

    created = create_validation_sample(clusters, sample, sample_size=2)
    created["human_label"] = ["equivalent", "not_equivalent"]
    created.to_csv(sample, index=False)
    content = score_validation(sample, scores)

    assert len(created) == 2
    assert "Number reviewed: 2" in content
    assert "Estimated useful precision: 50.00%" in content
