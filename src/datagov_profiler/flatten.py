from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from datagov_profiler.models import MetadataRecord
from datagov_profiler.normalize import compact_list


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if isinstance(value, dict):
                records.append(value)
    return records


def flatten_file(input_path: Path, out: Path) -> pd.DataFrame:
    records = read_json_or_jsonl(input_path)
    rows = [flatten_record(record) for record in records]
    frame = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame


def flatten_distributions_file(input_path: Path, out: Path) -> pd.DataFrame:
    records = read_json_or_jsonl(input_path)
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.extend(flatten_distributions_record(record))
    frame = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("results"), list):
            return [item for item in data["results"] if isinstance(item, dict)]
        return [data]
    return []


def flatten_record(raw: dict[str, Any]) -> dict[str, Any]:
    dcat = dcat_payload(raw)
    record = MetadataRecord.model_validate(raw)
    organization = record.organization or {}
    resources = record.resources or raw.get("distributions") or dcat_distributions(dcat)
    extras = extras_to_dict(record.extras)

    resource_names = [text_field(resource, "name", "title") for resource in resources]
    resource_descriptions = [text_field(resource, "description") for resource in resources]
    resource_formats = [text_field(resource, "format") for resource in resources]
    resource_media_types = [text_field(resource, "mimetype", "mediaType", "media_type") for resource in resources]
    resource_urls = [
        text_field(resource, "url", "downloadURL", "download_url", "accessURL", "access_url")
        for resource in resources
    ]
    distribution_download_urls = [text_field(resource, "downloadURL", "download_url") for resource in resources]
    distribution_access_urls = [text_field(resource, "accessURL", "access_url", "url") for resource in resources]
    publisher = dcat_dict(dcat.get("publisher"))
    contact = dcat_dict(dcat.get("contactPoint"))
    keywords = dcat.get("keyword", extras.get("keyword", extras.get("keywords", "")))
    themes = dcat.get("theme", "")

    return {
        "id": record.id or raw.get("id") or dcat.get("identifier", ""),
        "name": record.name or "",
        "title": record.title or raw.get("title") or dcat.get("title", ""),
        "notes": record.notes or raw.get("description", "") or dcat.get("description", ""),
        "organization_title": organization.get("title", "") or dcat_value(publisher, "name"),
        "organization_name": organization.get("name", "") or dcat_value(publisher, "name"),
        "organization_id": organization.get("id", ""),
        "groups": join_named_items(record.groups),
        "tags": join_named_items(record.tags) or join_values(keywords),
        "license_title": record.license_title or dcat.get("license", ""),
        "metadata_created": record.metadata_created or "",
        "metadata_modified": record.metadata_modified or dcat.get("modified", ""),
        "url": record.url or dcat.get("landingPage", ""),
        "extras": json.dumps(extras, ensure_ascii=False, sort_keys=True),
        "resources_count": len(resources),
        "resource_names": compact_list(resource_names, limit=50),
        "resource_descriptions": compact_list(resource_descriptions, limit=50),
        "resource_formats": compact_list(resource_formats, limit=50),
        "resource_media_types": compact_list(resource_media_types, limit=50),
        "resource_urls": compact_list(resource_urls, limit=50),
        "dcat_type": dcat.get("@type", raw.get("@type", "")),
        "dcat_title": dcat.get("title", raw.get("dct:title", raw.get("title", ""))),
        "dcat_description": dcat.get("description", raw.get("dct:description", raw.get("description", raw.get("notes", "")))),
        "dcat_identifier": dcat.get("identifier", raw.get("identifier", "")),
        "dcat_accessLevel": dcat.get("accessLevel", ""),
        "dcat_modified": dcat.get("modified", extras.get("modified", extras.get("date_modified", ""))),
        "dcat_issued": dcat.get("issued", ""),
        "dcat_keyword": join_values(keywords),
        "dcat_theme": join_values(themes),
        "dcat_bureauCode": join_values(dcat.get("bureauCode", "")),
        "dcat_programCode": join_values(dcat.get("programCode", "")),
        "dcat_publisher_name": dcat_value(publisher, "name") or extras.get("publisher", extras.get("publisher_name", "")),
        "dcat_contactPoint_fn": dcat_value(contact, "fn"),
        "dcat_contactPoint_hasEmail": dcat_value(contact, "hasEmail") or extras.get("contactPoint.hasEmail", extras.get("contact_email", "")),
        "dcat_landingPage": dcat.get("landingPage", ""),
        "dcat_license": dcat.get("license", ""),
        "dcat_rights": dcat.get("rights", ""),
        "dcat_spatial": dcat.get("spatial", ""),
        "dcat_temporal": dcat.get("temporal", ""),
        "distribution_count": len(resources),
        "distribution_titles": compact_list(resource_names, limit=50),
        "distribution_descriptions": compact_list(resource_descriptions, limit=50),
        "distribution_formats": compact_list(resource_formats, limit=50),
        "distribution_media_types": compact_list(resource_media_types, limit=50),
        "distribution_download_urls": compact_list(distribution_download_urls, limit=50),
        "distribution_access_urls": compact_list(distribution_access_urls, limit=50),
        "harvest_record": raw.get("harvest_record", ""),
        "harvest_record_raw": raw.get("harvest_record_raw", ""),
    }


def flatten_distributions_record(raw: dict[str, Any]) -> list[dict[str, Any]]:
    dcat = dcat_payload(raw)
    dataset_id = str(raw.get("id") or dcat.get("identifier") or "")
    dataset_title = str(raw.get("title") or dcat.get("title") or "")
    distributions = raw.get("distributions") or raw.get("resources") or dcat_distributions(dcat)
    rows = []
    for index, distribution in enumerate(distributions if isinstance(distributions, list) else []):
        if not isinstance(distribution, dict):
            continue
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_title": dataset_title,
                "distribution_index": index,
                "distribution_title": text_field(distribution, "title", "name"),
                "distribution_description": text_field(distribution, "description"),
                "format": text_field(distribution, "format"),
                "mediaType": text_field(distribution, "mediaType", "mimetype", "media_type"),
                "downloadURL": text_field(distribution, "downloadURL", "download_url"),
                "accessURL": text_field(distribution, "accessURL", "access_url", "url"),
                "conformsTo": text_field(distribution, "conformsTo"),
                "describedBy": text_field(distribution, "describedBy"),
                "describedByType": text_field(distribution, "describedByType"),
            }
        )
    return rows


def dcat_payload(raw: dict[str, Any]) -> dict[str, Any]:
    dcat = raw.get("dcat")
    if isinstance(dcat, dict):
        return dcat
    if raw.get("@type") or raw.get("distribution") or raw.get("keyword"):
        return raw
    return {}


def dcat_distributions(dcat: dict[str, Any]) -> list[dict[str, Any]]:
    distributions = dcat.get("distribution") or dcat.get("distributions") or []
    if isinstance(distributions, dict):
        return [distributions]
    if isinstance(distributions, list):
        return [item for item in distributions if isinstance(item, dict)]
    return []


def dcat_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dcat_value(value: dict[str, Any], key: str) -> str:
    result = value.get(key, "")
    return join_values(result)


def join_values(value: Any) -> str:
    if isinstance(value, list):
        return compact_list([str(item) for item in value], limit=50)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value or "")


def extras_to_dict(extras: list[dict[str, Any]] | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(extras, dict):
        return extras
    result: dict[str, Any] = {}
    if isinstance(extras, list):
        for item in extras:
            if isinstance(item, dict) and "key" in item:
                result[str(item.get("key"))] = item.get("value", "")
    return result


def join_named_items(items: list[dict[str, Any]]) -> str:
    values = []
    for item in items:
        if isinstance(item, dict):
            values.append(str(item.get("display_name") or item.get("name") or item.get("title") or "").strip())
    return compact_list(values, limit=50)


def text_field(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value:
            return str(value)
    return ""
