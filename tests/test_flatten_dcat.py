from pathlib import Path

from datagov_profiler.flatten import flatten_distributions_file, flatten_file


def test_flatten_catalog_dcat_record(tmp_path: Path) -> None:
    out = tmp_path / "metadata.csv"
    frame = flatten_file(Path("tests/fixtures/catalog_records.jsonl"), out)

    row = frame.iloc[0]
    assert row["id"] == "school-harassment-bullying"
    assert row["dcat_type"] == "dcat:Dataset"
    assert row["dcat_accessLevel"] == "public"
    assert row["distribution_count"] == 2
    assert "bullying" in row["dcat_keyword"]


def test_flatten_distributions(tmp_path: Path) -> None:
    out = tmp_path / "distributions.csv"
    frame = flatten_distributions_file(Path("tests/fixtures/catalog_records.jsonl"), out)

    assert len(frame) == 2
    assert frame.iloc[0]["dataset_id"] == "school-harassment-bullying"
    assert frame.iloc[0]["format"] == "CSV"
    assert frame.iloc[0]["downloadURL"] == "tests/fixtures/resource_sample.csv"
