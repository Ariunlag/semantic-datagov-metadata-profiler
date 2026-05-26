from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from datagov_profiler.flatten import read_json_or_jsonl

ALIAS_TO_DCAT = {
    "name": "title",
    "dataset_name": "title",
    "notes": "description",
    "summary": "description",
    "abstract": "description",
    "overview": "description",
    "description_text": "description",
    "tags": "keyword",
    "topics": "keyword",
    "subjects": "keyword",
    "keywords_list": "keyword",
    "modified_at": "modified",
    "last_updated": "modified",
    "update_date": "modified",
    "date_modified": "modified",
    "agency": "publisher",
    "publisher_name": "publisher",
    "organization": "publisher",
    "contact_email": "contactPoint.hasEmail",
    "maintainer_email": "contactPoint.hasEmail",
    "download_link": "distribution.downloadURL",
    "file_url": "distribution.downloadURL",
    "csv_url": "distribution.downloadURL",
    "api_endpoint": "distribution.accessURL",
    "service_url": "distribution.accessURL",
    "endpoint_url": "distribution.accessURL",
}

REQUIRED_FIELDS = {"title", "description", "keyword", "modified", "publisher"}


def suggest_field_mapping(field_name: str, example_value: object = "") -> dict[str, object]:
    normalized = normalize_field_name(field_name)
    if normalized.startswith("distribution_"):
        normalized = normalized.removeprefix("distribution_")
    if normalized in ALIAS_TO_DCAT:
        return {
            "input_field": field_name,
            "suggested_dcat_field": ALIAS_TO_DCAT[normalized],
            "confidence": 1.0,
            "reason": "seed alias dictionary",
            "example_value": stringify_example(example_value),
            "warning_if_required_field_missing": "",
        }
    best_alias, score = best_alias_match(normalized)
    suggested = ALIAS_TO_DCAT.get(best_alias, "")
    confidence = round(score / 100, 3)
    return {
        "input_field": field_name,
        "suggested_dcat_field": suggested if confidence >= 0.82 else "",
        "confidence": confidence if confidence >= 0.82 else 0.0,
        "reason": f"fuzzy alias match: {best_alias}" if confidence >= 0.82 else "no confident mapping",
        "example_value": stringify_example(example_value),
        "warning_if_required_field_missing": "",
    }


def map_dcat_file(input_path: Path, out: Path) -> pd.DataFrame:
    rows = load_rows(input_path)
    field_examples = collect_field_examples(rows)
    suggestions = [suggest_field_mapping(field, example) for field, example in sorted(field_examples.items())]
    present = {str(item["suggested_dcat_field"]) for item in suggestions if item["suggested_dcat_field"]}
    missing = sorted(REQUIRED_FIELDS - present)
    warning = f"Missing likely required fields: {', '.join(missing)}" if missing else ""
    for item in suggestions:
        item["warning_if_required_field_missing"] = warning
    result = pd.DataFrame(suggestions)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    return result


def load_rows(input_path: Path) -> list[dict[str, Any]]:
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path).fillna("").to_dict(orient="records")
    return read_json_or_jsonl(input_path)


def collect_field_examples(rows: list[dict[str, Any]]) -> dict[str, object]:
    examples: dict[str, object] = {}
    for row in rows:
        collect_from_dict(row, examples)
    return examples


def collect_from_dict(row: dict[str, Any], examples: dict[str, object], prefix: str = "") -> None:
    for key, value in row.items():
        field_name = f"{prefix}.{key}" if prefix else str(key)
        examples.setdefault(field_name, value)
        if key == "extras" and isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "key" in item:
                    examples.setdefault(str(item["key"]), item.get("value", ""))
        if key == "resources" and isinstance(value, list):
            for resource in value:
                if isinstance(resource, dict):
                    for resource_key, resource_value in resource.items():
                        examples.setdefault(f"distribution.{resource_key}", resource_value)


def best_alias_match(normalized_field: str) -> tuple[str, float]:
    scores = [(alias, fuzz.token_set_ratio(normalized_field, alias)) for alias in ALIAS_TO_DCAT]
    return max(scores, key=lambda item: item[1])


def normalize_field_name(field_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(field_name).lower()).strip("_")
    return normalized


def stringify_example(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)[:300]
    return str(value)[:300]
