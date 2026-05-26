from pathlib import Path

import pandas as pd

from datagov_profiler.cluster import ClusterMetrics, cluster_terms, cluster_quality_metrics


def write_terms(path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"term": "traffic crash", "normalized_term": "traffic crash", "source_field": "tags", "count": 2, "example_titles": "Traffic Crash Records"},
            {"term": "motor vehicle collision", "normalized_term": "motor vehicle collision", "source_field": "tags", "count": 2, "example_titles": "Motor Vehicle Collision Data"},
            {"term": "road accident", "normalized_term": "road accident", "source_field": "tags", "count": 2, "example_titles": "Road Accident Summary"},
            {"term": "keyword", "normalized_term": "keyword", "source_field": "tags", "count": 2, "example_titles": "Traffic Crash Records"},
            {"term": "tag", "normalized_term": "tag", "source_field": "tags", "count": 2, "example_titles": "Motor Vehicle Collision Data"},
            {"term": "topic", "normalized_term": "topic", "source_field": "tags", "count": 2, "example_titles": "Road Accident Summary"},
        ]
    )
    frame.to_csv(path, index=False)


def test_hybrid_clusters_seed_semantic_examples(tmp_path: Path) -> None:
    input_path = tmp_path / "terms.csv"
    output_path = tmp_path / "clusters.csv"
    write_terms(input_path)

    result = cluster_terms(input_path, output_path, method="hybrid", min_count=1, similarity_threshold=0.82)

    traffic = result[result["canonical_term"] == "traffic crash"]["normalized_term"].tolist()
    keyword = result[result["canonical_term"] == "keyword"]["normalized_term"].tolist()

    assert {"traffic crash", "motor vehicle collision", "road accident"}.issubset(set(traffic))
    assert {"keyword", "tag", "topic"}.issubset(set(keyword))
    assert output_path.exists()


def test_cluster_terms_writes_empty_output_when_no_terms_pass_filter(tmp_path: Path) -> None:
    input_path = tmp_path / "terms.csv"
    output_path = tmp_path / "clusters.csv"
    pd.DataFrame(
        [
            {
                "term": "rare term",
                "normalized_term": "rare term",
                "source_field": "tags",
                "count": 1,
                "example_titles": "Rare Dataset",
            }
        ]
    ).to_csv(input_path, index=False)

    result = cluster_terms(input_path, output_path, method="hybrid", min_count=2)

    assert result.empty
    assert list(result.columns) == [
        "cluster_id",
        "canonical_term",
        "observed_term",
        "normalized_term",
        "source_fields",
        "count",
        "similarity_score",
        "cluster_reason",
        "examples",
    ]
    assert output_path.exists()


def test_format_terms_cluster_together_without_global_cluster(tmp_path: Path) -> None:
    input_path = tmp_path / "terms.csv"
    output_path = tmp_path / "clusters.csv"
    pd.DataFrame(
        [
            {"term": "CSV", "normalized_term": "csv", "source_field": "formats", "count": 5, "example_titles": "A"},
            {"term": "text/csv", "normalized_term": "text csv", "source_field": "media_types", "count": 4, "example_titles": "B"},
            {"term": "Comma Separated Values", "normalized_term": "comma separated values", "source_field": "formats", "count": 3, "example_titles": "C"},
            {"term": "sensor", "normalized_term": "sensor", "source_field": "tags", "count": 2, "example_titles": "D"},
            {"term": "telemetry", "normalized_term": "telemetry", "source_field": "tags", "count": 2, "example_titles": "E"},
        ]
    ).to_csv(input_path, index=False)

    result = cluster_terms(input_path, output_path, method="hybrid", min_count=1)

    csv_cluster = result[result["canonical_term"] == "csv"]["normalized_term"].tolist()
    assert {"csv", "text csv", "comma separated values"}.issubset(set(csv_cluster))
    assert result.attrs["metrics"].largest_cluster_size < len(result)


def test_api_terms_cluster_together(tmp_path: Path) -> None:
    input_path = tmp_path / "terms.csv"
    output_path = tmp_path / "clusters.csv"
    pd.DataFrame(
        [
            {"term": "API", "normalized_term": "api", "source_field": "formats", "count": 3, "example_titles": "A"},
            {"term": "REST API", "normalized_term": "rest api", "source_field": "formats", "count": 2, "example_titles": "B"},
            {"term": "web service", "normalized_term": "web service", "source_field": "formats", "count": 2, "example_titles": "C"},
            {"term": "monitoring", "normalized_term": "monitoring", "source_field": "tags", "count": 2, "example_titles": "D"},
        ]
    ).to_csv(input_path, index=False)

    result = cluster_terms(input_path, output_path, method="hybrid", min_count=1)

    api_cluster = result[result["canonical_term"] == "api"]["normalized_term"].tolist()
    assert {"api", "rest api", "web service"}.issubset(set(api_cluster))


def test_sensor_domain_excludes_unrelated_stream_and_generic_terms(tmp_path: Path) -> None:
    input_path = tmp_path / "terms.csv"
    output_path = tmp_path / "clusters.csv"
    pd.DataFrame(
        [
            {"term": "sensor", "normalized_term": "sensor", "source_field": "tags", "count": 3, "example_titles": "A"},
            {"term": "telemetry", "normalized_term": "telemetry", "source_field": "tags", "count": 3, "example_titles": "B"},
            {"term": "monitoring", "normalized_term": "monitoring", "source_field": "tags", "count": 3, "example_titles": "C"},
            {"term": "application/octet-stream", "normalized_term": "application octet stream", "source_field": "media_type", "count": 50, "example_titles": "D"},
            {"term": "united states", "normalized_term": "united states", "source_field": "description", "count": 40, "example_titles": "E"},
        ]
    ).to_csv(input_path, index=False)

    result = cluster_terms(input_path, output_path, method="hybrid", min_count=1, domain="sensor_telemetry")

    assert "application octet stream" not in set(result["normalized_term"])
    assert "united states" not in set(result["normalized_term"])
    assert {"sensor", "telemetry", "monitoring"}.issubset(set(result["normalized_term"]))
    assert result.attrs["metrics"].largest_cluster_size <= 1


def test_largest_cluster_warning_triggers_for_bad_global_cluster() -> None:
    result = pd.DataFrame(
        [
            {"cluster_id": 1, "normalized_term": f"term {index}"}
            for index in range(10)
        ]
    )

    metrics = cluster_quality_metrics(result)

    assert isinstance(metrics, ClusterMetrics)
    assert metrics.warning
