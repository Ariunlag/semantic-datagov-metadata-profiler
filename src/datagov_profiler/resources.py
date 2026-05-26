from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

from datagov_profiler.field_roles import infer_field_roles, roles_by_type

ZIP_SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".json", ".xml", ".txt", ".geojson"}
ZIP_PROFILE_EXTENSIONS = {".csv", ".xlsx"}
ZIP_SKIPPED_EXTENSIONS = {".shp", ".dbf", ".prj", ".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".zip"}


def profile_resources(
    distributions_path: Path,
    out: Path,
    *,
    download_dir: Path,
    allowed_formats: str,
    max_files: int,
    max_file_size_mb: int,
    timeout: int,
    allow_zip: bool = False,
    max_zip_size_mb: int = 50,
    max_files_per_zip: int = 20,
    extract_dir: Path = Path("data/raw/resource_samples/extracted"),
    inspect_zip_only: bool = True,
) -> pd.DataFrame:
    distributions = pd.read_csv(distributions_path).fillna("")
    download_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    allowed = {item.strip().upper() for item in allowed_formats.split(",") if item.strip()}
    rows: list[dict[str, Any]] = []
    zip_manifest_rows: list[dict[str, Any]] = []
    processed = 0
    for _, dist in distributions.iterrows():
        if processed >= max_files:
            break
        file_type = detect_file_type(dist, allowed)
        if not file_type:
            continue
        url = str(dist.get("downloadURL") or dist.get("accessURL") or "")
        if not url:
            continue
        try:
            if file_type == "ZIP":
                if not allow_zip:
                    rows.append(
                        base_profile_row(
                            dist.to_dict(),
                            url,
                            "",
                            file_type,
                            errors="skipped_zip_allow_zip_false",
                        )
                    )
                    processed += 1
                    continue
                zip_rows, manifest = profile_zip_resource(
                    dist.to_dict(),
                    url,
                    download_dir=download_dir,
                    extract_dir=extract_dir,
                    max_zip_size_mb=max_zip_size_mb,
                    max_inner_file_size_mb=max_file_size_mb,
                    max_files_per_zip=max_files_per_zip,
                    timeout=timeout,
                    inspect_zip_only=inspect_zip_only,
                )
                rows.extend(zip_rows)
                zip_manifest_rows.extend(manifest)
                processed += 1
                continue
            local_file = local_or_download(url, download_dir, file_type, max_file_size_mb=max_file_size_mb, timeout=timeout)
            rows.extend(profile_local_file(local_file, file_type, dist.to_dict(), url))
        except Exception as exc:  # noqa: BLE001 - resource profiling should be best-effort.
            rows.append(base_profile_row(dist.to_dict(), url, "", file_type, errors=str(exc)))
            if file_type == "ZIP" and allow_zip:
                zip_manifest_rows.append(
                    zip_manifest_row(dist.to_dict(), url, "", "", "", 0, False, zip_error_reason(str(exc)))
                )
        processed += 1
    result = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    if zip_manifest_rows:
        write_zip_manifest(zip_manifest_rows, out.parent / "zip_manifest.csv")
    return result


def detect_file_type(row: pd.Series, allowed: set[str]) -> str:
    format_value = str(row.get("format", "")).upper()
    media_type = str(row.get("mediaType", "")).lower()
    url = str(row.get("downloadURL") or row.get("accessURL") or "").lower()
    if "CSV" in allowed and ("CSV" in format_value or "text/csv" in media_type or url.endswith(".csv")):
        return "CSV"
    if "XLSX" in allowed and ("XLSX" in format_value or "spreadsheet" in media_type or url.endswith(".xlsx")):
        return "XLSX"
    if "ZIP" in format_value or "application/zip" in media_type or url.endswith(".zip"):
        return "ZIP"
    return ""


def local_or_download(url: str, download_dir: Path, file_type: str, *, max_file_size_mb: int, timeout: int) -> Path:
    direct_path = Path(url)
    if direct_path.exists():
        if direct_path.stat().st_size > max_file_size_mb * 1024 * 1024:
            raise RuntimeError(f"resource exceeds {max_file_size_mb} MB")
        return direct_path
    parsed = urlparse(url)
    if parsed.scheme in {"", "file"}:
        path = Path(parsed.path if parsed.scheme == "file" else url)
        if path.exists():
            if path.stat().st_size > max_file_size_mb * 1024 * 1024:
                raise RuntimeError(f"resource exceeds {max_file_size_mb} MB")
            return path
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    size = int(response.headers.get("content-length") or 0)
    if size and size > max_file_size_mb * 1024 * 1024:
        raise RuntimeError(f"resource exceeds {max_file_size_mb} MB")
    suffix = { "CSV": ".csv", "XLSX": ".xlsx", "ZIP": ".zip" }.get(file_type, "")
    local_file = download_dir / f"{safe_file_stem(url)}{suffix}"
    total = 0
    with local_file.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            total += len(chunk)
            if total > max_file_size_mb * 1024 * 1024:
                raise RuntimeError(f"resource exceeds {max_file_size_mb} MB")
            handle.write(chunk)
    return local_file


def profile_zip_resource(
    dist: dict[str, Any],
    url: str,
    *,
    download_dir: Path,
    extract_dir: Path,
    max_zip_size_mb: int,
    max_inner_file_size_mb: int,
    max_files_per_zip: int,
    timeout: int,
    inspect_zip_only: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    local_zip = local_or_download(url, download_dir, "ZIP", max_file_size_mb=max_zip_size_mb, timeout=timeout)
    if local_zip.stat().st_size > max_zip_size_mb * 1024 * 1024:
        return (
            [base_profile_row(dist, url, str(local_zip), "ZIP", errors="zip_too_large")],
            [zip_manifest_row(dist, url, str(local_zip), "", "", 0, False, "zip_too_large")],
        )
    try:
        with zipfile.ZipFile(local_zip) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()][:max_files_per_zip]
            rows: list[dict[str, Any]] = []
            manifest: list[dict[str, Any]] = []
            for info in infos:
                extension = Path(info.filename).suffix.lower()
                skip_reason = zip_skip_reason(info, extension, max_inner_file_size_mb=max_inner_file_size_mb)
                selected = skip_reason == "" and extension in ZIP_PROFILE_EXTENSIONS
                manifest.append(
                    zip_manifest_row(
                        dist,
                        url,
                        str(local_zip),
                        info.filename,
                        extension,
                        info.file_size,
                        selected,
                        skip_reason or ("manifest_only" if extension not in ZIP_PROFILE_EXTENSIONS else ""),
                    )
                )
                if not selected:
                    continue
                extracted = safe_extract_member(archive, info, extract_dir)
                rows.extend(profile_local_file(extracted, extension_to_file_type(extension), dist, url))
            if len(archive.infolist()) > max_files_per_zip:
                manifest.append(
                    zip_manifest_row(
                        dist,
                        url,
                        str(local_zip),
                        "",
                        "",
                        0,
                        False,
                        f"zip_file_limit_reached_{max_files_per_zip}",
                    )
                )
            if not rows:
                rows.append(base_profile_row(dist, url, str(local_zip), "ZIP", errors="zip_manifest_created_no_schema_profile"))
            return rows, manifest
    except zipfile.BadZipFile:
        return (
            [base_profile_row(dist, url, str(local_zip), "ZIP", errors="corrupted_zip")],
            [zip_manifest_row(dist, url, str(local_zip), "", "", 0, False, "corrupted_zip")],
        )
    except UnicodeDecodeError:
        return (
            [base_profile_row(dist, url, str(local_zip), "ZIP", errors="zip_encoding_error")],
            [zip_manifest_row(dist, url, str(local_zip), "", "", 0, False, "zip_encoding_error")],
        )


def zip_skip_reason(info: zipfile.ZipInfo, extension: str, *, max_inner_file_size_mb: int) -> str:
    if is_unsafe_zip_path(info.filename):
        return "unsafe_path"
    if extension in ZIP_SKIPPED_EXTENSIONS:
        return "skipped_extension"
    if extension not in ZIP_SUPPORTED_EXTENSIONS:
        return "unsupported_extension"
    if info.file_size > max_inner_file_size_mb * 1024 * 1024:
        return "inner_file_too_large"
    return ""


def is_unsafe_zip_path(name: str) -> bool:
    normalized = Path(name.replace("\\", "/"))
    return normalized.is_absolute() or ".." in normalized.parts


def safe_extract_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, extract_dir: Path) -> Path:
    if is_unsafe_zip_path(info.filename):
        raise RuntimeError("unsafe ZIP path traversal attempt")
    target = (extract_dir / info.filename).resolve()
    extract_root = extract_dir.resolve()
    if extract_root not in target.parents and target != extract_root:
        raise RuntimeError("unsafe ZIP extraction target")
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, target.open("wb") as destination:
        destination.write(source.read())
    return target


def extension_to_file_type(extension: str) -> str:
    return "CSV" if extension == ".csv" else "XLSX"


def zip_manifest_row(
    dist: dict[str, Any],
    zip_url: str,
    zip_local_file: str,
    inner_file_name: str,
    inner_file_extension: str,
    inner_file_size: int,
    selected: bool,
    skip_reason: str,
) -> dict[str, Any]:
    return {
        "dataset_id": dist.get("dataset_id", ""),
        "dataset_title": dist.get("dataset_title", ""),
        "distribution_title": dist.get("distribution_title", ""),
        "zip_url": zip_url,
        "zip_local_file": zip_local_file,
        "inner_file_name": inner_file_name,
        "inner_file_extension": inner_file_extension,
        "inner_file_size_bytes": inner_file_size,
        "selected_for_schema_profile": selected,
        "skip_reason": skip_reason,
    }


def write_zip_manifest(rows: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def zip_error_reason(error: str) -> str:
    if "exceeds" in error:
        return "zip_too_large"
    return error or "zip_error"


def profile_local_file(path: Path, file_type: str, dist: dict[str, Any], url: str) -> list[dict[str, Any]]:
    if file_type == "CSV":
        frame = pd.read_csv(path, nrows=1000)
        return [profile_frame(frame, dist, url, str(path), file_type, sheet_name="")]
    excel = pd.ExcelFile(path)
    rows = []
    for sheet_name in excel.sheet_names:
        raw = pd.read_excel(excel, sheet_name=sheet_name, header=None, nrows=200)
        header_row = detect_header_row(raw)
        frame = pd.read_excel(excel, sheet_name=sheet_name, header=header_row, nrows=1000)
        frame.columns = flatten_headers(frame.columns)
        rows.append(profile_frame(frame, dist, url, str(path), file_type, sheet_name=sheet_name, header_row=header_row))
    return rows


def profile_frame(
    frame: pd.DataFrame,
    dist: dict[str, Any],
    url: str,
    local_file: str,
    file_type: str,
    *,
    sheet_name: str,
    header_row: int = 0,
) -> dict[str, Any]:
    roles = infer_field_roles(frame)
    return {
        **base_profile_row(dist, url, local_file, file_type),
        "sheet_name": sheet_name,
        "row_count_estimate": len(frame),
        "column_count": len(frame.columns),
        "detected_header_row": header_row,
        "detected_columns": " | ".join(str(column) for column in frame.columns),
        "inferred_time_fields": roles_by_type(roles, "time_field") or roles_by_type(roles, "date_field"),
        "inferred_geo_fields": " | ".join(filter(None, [roles_by_type(roles, "location_field"), roles_by_type(roles, "latitude_field"), roles_by_type(roles, "longitude_field"), roles_by_type(roles, "geometry_field")])),
        "inferred_id_fields": roles_by_type(roles, "entity_id_field"),
        "inferred_category_fields": roles_by_type(roles, "category_field"),
        "inferred_status_fields": roles_by_type(roles, "status_field"),
        "inferred_metric_fields": " | ".join(filter(None, [roles_by_type(roles, "metric_field"), roles_by_type(roles, "count_field"), roles_by_type(roles, "percentage_field")])),
        "notes": "schema profile generated from opt-in resource sample",
        "errors": "",
    }


def base_profile_row(dist: dict[str, Any], url: str, local_file: str, file_type: str, errors: str = "") -> dict[str, Any]:
    return {
        "dataset_id": dist.get("dataset_id", ""),
        "dataset_title": dist.get("dataset_title", ""),
        "distribution_title": dist.get("distribution_title", ""),
        "resource_url": url,
        "local_file": local_file,
        "file_type": file_type,
        "sheet_name": "",
        "row_count_estimate": "",
        "column_count": "",
        "detected_header_row": "",
        "detected_columns": "",
        "inferred_time_fields": "",
        "inferred_geo_fields": "",
        "inferred_id_fields": "",
        "inferred_category_fields": "",
        "inferred_status_fields": "",
        "inferred_metric_fields": "",
        "notes": "",
        "errors": errors,
    }


def detect_header_row(raw: pd.DataFrame) -> int:
    best_row = 0
    best_score = -1
    for idx, row in raw.head(10).iterrows():
        values = [str(value).strip() for value in row.tolist() if pd.notna(value) and str(value).strip()]
        score = len(values) + sum(1 for value in values if re.search(r"[A-Za-z]", value))
        if score > best_score:
            best_score = score
            best_row = int(idx)
    return best_row


def flatten_headers(columns: pd.Index) -> list[str]:
    result = []
    for column in columns:
        if isinstance(column, tuple):
            result.append(" ".join(str(part) for part in column if str(part) != "nan").strip())
        else:
            result.append(str(column).strip())
    return result


def safe_file_stem(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).stem or "resource"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:80]


def sample_rows_json(frame: pd.DataFrame) -> str:
    return json.dumps(frame.head(5).to_dict(orient="records"), ensure_ascii=False)
