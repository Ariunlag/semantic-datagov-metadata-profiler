from pathlib import Path

import pandas as pd

from datagov_profiler.resources import profile_resources


def test_profile_resource_fixture_csv(tmp_path: Path) -> None:
    distributions = tmp_path / "distributions.csv"
    out = tmp_path / "profiles.csv"
    pd.DataFrame(
        [
            {
                "dataset_id": "school-harassment-bullying",
                "dataset_title": "School Harassment and Bullying Reports",
                "distribution_title": "HIB CSV file",
                "format": "CSV",
                "mediaType": "text/csv",
                "downloadURL": "tests/fixtures/resource_sample.csv",
                "accessURL": "",
            }
        ]
    ).to_csv(distributions, index=False)

    result = profile_resources(
        distributions,
        out,
        download_dir=tmp_path / "downloads",
        allowed_formats="CSV",
        max_files=1,
        max_file_size_mb=1,
        timeout=5,
    )

    row = result.iloc[0]
    assert row["column_count"] == 12
    assert "report_date" in row["inferred_time_fields"]
    assert "latitude" in row["inferred_geo_fields"]
    assert "incident_count" in row["inferred_metric_fields"]
