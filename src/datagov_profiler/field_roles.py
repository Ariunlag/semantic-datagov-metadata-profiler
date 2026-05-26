from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class FieldRole:
    field_name: str
    inferred_role: str
    confidence: float
    reason: str


ROLE_PATTERNS = {
    "title_field": [r"\btitle\b", r"\bname\b"],
    "description_field": [r"desc", r"summary", r"abstract", r"notes?"],
    "time_field": [r"\btime\b", r"timestamp", r"hour"],
    "date_field": [r"\bdate\b", r"created", r"modified", r"updated"],
    "location_field": [r"address", r"location", r"city", r"county", r"state", r"zip"],
    "latitude_field": [r"\blat\b", r"latitude"],
    "longitude_field": [r"\blon\b", r"\blng\b", r"longitude"],
    "geometry_field": [r"geometry", r"geom", r"wkt", r"shape"],
    "entity_id_field": [r"(^|_)id($|_)", r"identifier", r"case_number", r"record"],
    "category_field": [r"category", r"type", r"class", r"group"],
    "status_field": [r"status", r"state", r"outcome", r"disposition"],
    "metric_field": [r"amount", r"value", r"score", r"rate", r"measure"],
    "percentage_field": [r"percent", r"percentage", r"\bpct\b"],
    "count_field": [r"count", r"total", r"number", r"\bnum\b"],
    "url_field": [r"url", r"link", r"website"],
    "contact_email_field": [r"email", r"e_mail"],
}


def infer_field_roles(frame: pd.DataFrame, sample_size: int = 50) -> list[FieldRole]:
    roles: list[FieldRole] = []
    sample = frame.head(sample_size)
    for column in frame.columns:
        roles.extend(infer_column_roles(str(column), sample[column]))
    return sorted(roles, key=lambda item: (item.field_name, -item.confidence, item.inferred_role))


def infer_column_roles(column: str, values: Iterable[object]) -> list[FieldRole]:
    name = normalize_name(column)
    sample = [str(value) for value in values if pd.notna(value) and str(value).strip()]
    roles: list[FieldRole] = []
    for role, patterns in ROLE_PATTERNS.items():
        score = max((0.75 for pattern in patterns if re.search(pattern, name)), default=0.0)
        reasons = []
        if score:
            reasons.append("header name match")
        value_score, value_reason = score_values(role, sample)
        if value_score:
            score = max(score, value_score)
            reasons.append(value_reason)
        if score:
            roles.append(FieldRole(column, role, round(min(score, 0.98), 2), "; ".join(reasons)))
    return roles or [FieldRole(column, "category_field", 0.35, "fallback low-cardinality/string candidate")]


def score_values(role: str, sample: list[str]) -> tuple[float, str]:
    if not sample:
        return 0.0, ""
    joined = "\n".join(sample[:20])
    if role == "contact_email_field" and re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", joined):
        return 0.9, "email-like sample values"
    if role == "url_field" and re.search(r"https?://|www\.", joined):
        return 0.9, "URL-like sample values"
    if role in {"date_field", "time_field"} and re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", joined):
        return 0.8, "date-like sample values"
    numeric = sum(1 for value in sample if re.fullmatch(r"-?\d+(\.\d+)?%?", value.strip()))
    if numeric / max(len(sample), 1) >= 0.7:
        if role in {"metric_field", "count_field"}:
            return 0.72, "mostly numeric sample values"
        if role == "percentage_field" and any("%" in value for value in sample):
            return 0.86, "percentage-like sample values"
    return 0.0, ""


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def roles_by_type(roles: list[FieldRole], role: str) -> str:
    return " | ".join(item.field_name for item in roles if item.inferred_role == role and item.confidence >= 0.5)
