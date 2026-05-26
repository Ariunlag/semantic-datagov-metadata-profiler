from pathlib import Path
import zipfile

import pandas as pd

from datagov_profiler.resources import profile_resources


def write_distribution(path: Path, zip_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "dataset_id": "zip-dataset",
                "dataset_title": "ZIP Dataset",
                "distribution_title": "ZIP resource",
                "format": "ZIP",
                "mediaType": "application/zip",
                "downloadURL": str(zip_path),
                "accessURL": "",
            }
        ]
    ).to_csv(path, index=False)


def create_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "data/table.csv",
            "record_id,report_date,status,count\n1,2024-01-01,Open,2\n2,2024-01-02,Closed,4\n",
        )
        archive.writestr("docs/readme.txt", "small text metadata")
        archive.writestr("images/picture.png", b"not really an image")
        archive.writestr("../evil.csv", "id,value\n1,unsafe\n")


def test_zip_skipped_when_allow_zip_false(tmp_path: Path) -> None:
    zip_path = tmp_path / "resource.zip"
    create_zip(zip_path)
    distributions = tmp_path / "distributions.csv"
    out = tmp_path / "profiles.csv"
    write_distribution(distributions, zip_path)

    result = profile_resources(
        distributions,
        out,
        download_dir=tmp_path / "downloads",
        allowed_formats="CSV,XLSX",
        max_files=1,
        max_file_size_mb=1,
        timeout=5,
    )

    assert result.iloc[0]["file_type"] == "ZIP"
    assert result.iloc[0]["errors"] == "skipped_zip_allow_zip_false"


def test_zip_manifest_generation_and_csv_selection(tmp_path: Path) -> None:
    zip_path = tmp_path / "resource.zip"
    create_zip(zip_path)
    distributions = tmp_path / "distributions.csv"
    out = tmp_path / "profiles.csv"
    write_distribution(distributions, zip_path)

    result = profile_resources(
        distributions,
        out,
        download_dir=tmp_path / "downloads",
        allowed_formats="CSV,XLSX",
        max_files=1,
        max_file_size_mb=1,
        timeout=5,
        allow_zip=True,
        extract_dir=tmp_path / "extracted",
    )
    manifest = pd.read_csv(tmp_path / "zip_manifest.csv").fillna("")

    assert "record_id" in result.iloc[0]["detected_columns"]
    selected = manifest[manifest["inner_file_name"] == "data/table.csv"].iloc[0]
    assert bool(selected["selected_for_schema_profile"])
    assert selected["skip_reason"] == ""


def test_zip_path_traversal_protection(tmp_path: Path) -> None:
    zip_path = tmp_path / "resource.zip"
    create_zip(zip_path)
    distributions = tmp_path / "distributions.csv"
    out = tmp_path / "profiles.csv"
    write_distribution(distributions, zip_path)

    profile_resources(
        distributions,
        out,
        download_dir=tmp_path / "downloads",
        allowed_formats="CSV,XLSX",
        max_files=1,
        max_file_size_mb=1,
        timeout=5,
        allow_zip=True,
        extract_dir=tmp_path / "extracted",
    )
    manifest = pd.read_csv(tmp_path / "zip_manifest.csv").fillna("")

    unsafe = manifest[manifest["inner_file_name"] == "../evil.csv"].iloc[0]
    assert unsafe["skip_reason"] == "unsafe_path"
    assert not (tmp_path / "evil.csv").exists()


def test_zip_unsupported_inner_files_are_skipped(tmp_path: Path) -> None:
    zip_path = tmp_path / "resource.zip"
    create_zip(zip_path)
    distributions = tmp_path / "distributions.csv"
    out = tmp_path / "profiles.csv"
    write_distribution(distributions, zip_path)

    profile_resources(
        distributions,
        out,
        download_dir=tmp_path / "downloads",
        allowed_formats="CSV,XLSX",
        max_files=1,
        max_file_size_mb=1,
        timeout=5,
        allow_zip=True,
        extract_dir=tmp_path / "extracted",
    )
    manifest = pd.read_csv(tmp_path / "zip_manifest.csv").fillna("")

    png = manifest[manifest["inner_file_name"] == "images/picture.png"].iloc[0]
    txt = manifest[manifest["inner_file_name"] == "docs/readme.txt"].iloc[0]
    assert png["skip_reason"] == "skipped_extension"
    assert txt["skip_reason"] == "manifest_only"
