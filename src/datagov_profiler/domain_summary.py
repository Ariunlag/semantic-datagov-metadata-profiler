from __future__ import annotations

from pathlib import Path

import pandas as pd

from datagov_profiler.cluster import EXCLUDED_TERMS, is_sensor_telemetry_term, normalize_for_filter

SENSOR_PATTERNS = {
    "IoT exact count": ["iot"],
    "sensor count": ["sensor", "sensors"],
    "telemetry count": ["telemetry"],
    "monitoring count": ["monitoring"],
    "real-time count": ["real time", "realtime", "real-time"],
    "streamgage/streamflow count": ["streamgage", "stream gage", "streamflow"],
    "UAS/drone count": ["uas", "uav", "drone"],
}

RECOMMENDED_SENSOR_EXPANSIONS = [
    "sensor",
    "sensors",
    "telemetry",
    "environmental monitoring",
    "real-time monitoring",
    "streamgage",
    "stream gage",
    "streamflow",
    "station",
    "observation",
    "water quality",
    "device",
    "UAS",
    "UAV",
    "drone",
]


def generate_domain_summary(
    terms_path: Path,
    clusters_path: Path,
    *,
    domain: str,
    out: Path,
) -> str:
    terms = pd.read_csv(terms_path).fillna("")
    clusters = pd.read_csv(clusters_path).fillna("")
    if domain != "sensor_telemetry":
        raise ValueError("Only domain='sensor_telemetry' is currently supported")
    lines = [
        "# Sensor Telemetry Domain Summary",
        "",
        "## Term Counts",
        "",
    ]
    for label, patterns in SENSOR_PATTERNS.items():
        lines.append(f"- {label}: {count_matching_terms(terms, patterns)}")
    noisy = noisy_terms_detected(terms)
    lines.extend(
        [
            "",
            "## Interpretable Clusters",
            "",
            cluster_table(clusters),
            "",
            "## Noisy Terms Detected",
            "",
            ", ".join(noisy) if noisy else "_No configured noisy terms detected._",
            "",
            "## Recommended Semantic Expansion Terms",
            "",
            ", ".join(RECOMMENDED_SENSOR_EXPANSIONS),
            "",
        ]
    )
    content = "\n".join(lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return content


def count_matching_terms(terms: pd.DataFrame, patterns: list[str]) -> int:
    if terms.empty or "normalized_term" not in terms:
        return 0
    count = 0
    for _, row in terms.iterrows():
        value = normalize_for_filter(str(row.get("normalized_term", "")))
        if any(pattern in value for pattern in patterns):
            count += int(row.get("count", 1) or 1)
    return count


def noisy_terms_detected(terms: pd.DataFrame) -> list[str]:
    if terms.empty or "normalized_term" not in terms:
        return []
    values = {normalize_for_filter(str(value)) for value in terms["normalized_term"].tolist()}
    noisy = sorted((values & EXCLUDED_TERMS) | {value for value in values if value == "application octet stream"})
    return noisy


def cluster_table(clusters: pd.DataFrame) -> str:
    if clusters.empty:
        return "_No domain clusters available._"
    domain_clusters = clusters[clusters["normalized_term"].astype(str).map(is_sensor_telemetry_term)]
    if domain_clusters.empty:
        return "_No sensor/telemetry clusters found._"
    grouped = domain_clusters.groupby(["cluster_id", "canonical_term"], as_index=False).agg(
        observed_terms=("observed_term", lambda values: " | ".join(sorted(set(str(value) for value in values))[:8])),
        count=("count", "sum"),
    )
    rows = ["| cluster_id | canonical_term | observed_terms | count |", "| --- | --- | --- | --- |"]
    for _, row in grouped.sort_values("count", ascending=False).head(20).iterrows():
        rows.append(
            f"| {row['cluster_id']} | {row['canonical_term']} | {str(row['observed_terms']).replace('|', ';')} | {row['count']} |"
        )
    return "\n".join(rows)
