from datagov_profiler.dcat_mapping import suggest_field_mapping


def test_seed_alias_maps_to_dcat_field() -> None:
    suggestion = suggest_field_mapping("download_link", "https://example.gov/file.csv")
    assert suggestion["suggested_dcat_field"] == "distribution.downloadURL"
    assert suggestion["confidence"] == 1.0


def test_alias_to_publisher() -> None:
    suggestion = suggest_field_mapping("publisher_name", "Agency")
    assert suggestion["suggested_dcat_field"] == "publisher"


def test_unknown_field_has_no_confident_mapping() -> None:
    suggestion = suggest_field_mapping("unrelated_custom_metric", "42")
    assert suggestion["suggested_dcat_field"] == ""
