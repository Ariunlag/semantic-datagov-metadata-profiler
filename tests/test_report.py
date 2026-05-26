from pathlib import Path

from datagov_profiler.cluster import cluster_terms
from datagov_profiler.flatten import flatten_file
from datagov_profiler.report import generate_report
from datagov_profiler.terms import profile_terms


def test_report_generation_does_not_crash(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    terms = tmp_path / "terms.csv"
    clusters = tmp_path / "clusters.csv"
    report = tmp_path / "report.md"

    flatten_file(Path("tests/fixtures/catalog_records.jsonl"), metadata)
    profile_terms(metadata, terms)
    cluster_terms(terms, clusters, method="hybrid", min_count=1)
    content = generate_report(
        metadata,
        terms,
        clusters,
        report,
        case_study_dataset_id="school-harassment-bullying",
    )

    assert "DCAT Field Completeness" in content
    assert "Case Study" in content
    assert report.exists()
