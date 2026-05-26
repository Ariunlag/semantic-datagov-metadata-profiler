from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from datagov_profiler.normalize import compact_list, normalize_text, split_compound_terms

TERM_FIELDS = [
    "title",
    "notes",
    "tags",
    "groups",
    "organization_title",
    "organization_name",
    "resource_names",
    "resource_formats",
    "resource_media_types",
    "dcat_keyword",
    "dcat_theme",
    "dcat_publisher_name",
    "distribution_titles",
    "distribution_formats",
    "distribution_media_types",
]

TERM_COLUMNS = [
    "term",
    "normalized_term",
    "source_field",
    "count",
    "example_record_ids",
    "example_titles",
]


def profile_terms(input_csv: Path, out: Path, distributions_csv: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(input_csv).fillna("")
    if distributions_csv is not None:
        frame = pd.concat([frame, distribution_terms_frame(distributions_csv)], ignore_index=True, sort=False).fillna("")
    buckets: dict[tuple[str, str, str], dict[str, list[str] | int]] = defaultdict(
        lambda: {"count": 0, "record_ids": [], "titles": []}
    )
    for _, row in frame.iterrows():
        record_id = str(row.get("id", ""))
        title = str(row.get("title", ""))
        for field in TERM_FIELDS:
            for term in extract_terms(row.get(field, ""), field):
                normalized = normalize_text(term, remove_stopwords=True)
                if not normalized:
                    continue
                key = (term, normalized, field)
                buckets[key]["count"] = int(buckets[key]["count"]) + 1
                buckets[key]["record_ids"].append(record_id)
                buckets[key]["titles"].append(title)

    rows = [
        {
            "term": term,
            "normalized_term": normalized,
            "source_field": source_field,
            "count": values["count"],
            "example_record_ids": compact_list(values["record_ids"], limit=5),  # type: ignore[arg-type]
            "example_titles": compact_list(values["titles"], limit=5),  # type: ignore[arg-type]
        }
        for (term, normalized, source_field), values in buckets.items()
    ]
    result = pd.DataFrame(rows, columns=TERM_COLUMNS).sort_values(["count", "normalized_term"], ascending=[False, True])
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    return result


def distribution_terms_frame(distributions_csv: Path) -> pd.DataFrame:
    distributions = pd.read_csv(distributions_csv).fillna("")
    rows = []
    for _, row in distributions.iterrows():
        rows.append(
            {
                "id": row.get("dataset_id", ""),
                "title": row.get("dataset_title", ""),
                "distribution_titles": row.get("distribution_title", ""),
                "distribution_formats": row.get("format", ""),
                "distribution_media_types": row.get("mediaType", ""),
                "notes": row.get("distribution_description", ""),
            }
        )
    return pd.DataFrame(rows)


def extract_terms(value: object, source_field: str) -> list[str]:
    if value is None:
        return []
    raw = str(value)
    if not raw.strip():
        return []
    if source_field in {
        "tags",
        "groups",
        "resource_names",
        "resource_formats",
        "resource_media_types",
        "dcat_keyword",
        "dcat_theme",
        "distribution_titles",
        "distribution_formats",
        "distribution_media_types",
    }:
        return split_compound_terms(raw.replace(" | ", ";"))
    normalized = normalize_text(raw, remove_stopwords=True)
    if not normalized:
        return []
    phrases = [normalized]
    words = normalized.split()
    if len(words) > 2:
        phrases.extend([" ".join(words[i : i + 2]) for i in range(len(words) - 1)])
        phrases.extend([" ".join(words[i : i + 3]) for i in range(len(words) - 2)])
    return dedupe(phrases)


def dedupe(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen
