from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from datagov_profiler.cli import app


runner = CliRunner()


def write_metadata(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "id": "dataset-1",
                "title": "Traffic Crash Records",
                "notes": "Motor vehicle collision records",
                "tags": "traffic | crash",
                "distribution_count": 1,
            }
        ]
    ).to_csv(path, index=False)


def write_distributions(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "dataset_id": "dataset-1",
                "dataset_title": "Traffic Crash Records",
                "distribution_title": "Crash CSV",
                "format": "CSV",
                "mediaType": "text/csv",
            }
        ]
    ).to_csv(path, index=False)


def test_profile_terms_accepts_metadata_and_distributions(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    distributions = tmp_path / "distributions.csv"
    out = tmp_path / "terms.csv"
    write_metadata(metadata)
    write_distributions(distributions)

    result = runner.invoke(
        app,
        [
            "profile-terms",
            "--metadata",
            str(metadata),
            "--distributions",
            str(distributions),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "csv" in pd.read_csv(out)["normalized_term"].tolist()


def test_report_accepts_distributions(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    distributions = tmp_path / "distributions.csv"
    terms = tmp_path / "terms.csv"
    clusters = tmp_path / "clusters.csv"
    out = tmp_path / "report.md"
    write_metadata(metadata)
    write_distributions(distributions)
    pd.DataFrame(
        [{"term": "traffic crash", "normalized_term": "traffic crash", "source_field": "title", "count": 1}]
    ).to_csv(terms, index=False)
    pd.DataFrame(
        [
            {
                "cluster_id": 1,
                "canonical_term": "traffic crash",
                "observed_term": "traffic crash",
                "normalized_term": "traffic crash",
                "source_fields": "title",
                "count": 1,
                "similarity_score": 1,
                "cluster_reason": "exact",
                "examples": "",
            }
        ]
    ).to_csv(clusters, index=False)

    result = runner.invoke(
        app,
        [
            "report",
            "--metadata",
            str(metadata),
            "--distributions",
            str(distributions),
            "--terms",
            str(terms),
            "--clusters",
            str(clusters),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()


def test_cluster_terms_missing_input_has_clean_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["cluster-terms", "--input", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "clusters.csv")],
    )

    assert result.exit_code == 1
    assert "Input file does not exist" in result.output
    assert "FileNotFoundError" not in result.output
