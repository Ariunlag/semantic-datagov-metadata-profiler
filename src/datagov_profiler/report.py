from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def generate_report(
    metadata_path: Path,
    terms_path: Path,
    clusters_path: Path,
    out: Path,
    *,
    distributions_path: Path | None = None,
    resource_profiles_path: Path | None = None,
    case_study_dataset_id: str = "",
) -> str:
    metadata = pd.read_csv(metadata_path).fillna("")
    terms = pd.read_csv(terms_path).fillna("")
    clusters = pd.read_csv(clusters_path).fillna("")
    distributions = read_optional_csv(distributions_path)
    resource_profiles = read_optional_csv(resource_profiles_path)

    record_count = len(metadata)
    org_count = count_unique(metadata, "organization_name")
    unique_tags = unique_split_count(metadata.get("tags", pd.Series(dtype=str)).tolist())
    normalized_terms = terms["normalized_term"].nunique() if "normalized_term" in terms else 0
    distribution_total = (
        len(distributions)
        if not distributions.empty
        else int(metadata.get("distribution_count", metadata.get("resources_count", pd.Series(dtype=int))).replace("", 0).astype(int).sum()) if record_count else 0
    )

    lines = [
        "# Data.gov Metadata Heterogeneity Report",
        "",
        f"Collection date: {datetime.now(UTC).date().isoformat()}",
        "",
        "## Metadata Snapshot Summary",
        "",
        f"- Metadata records analyzed: {record_count}",
        f"- Organizations/publishers represented: {org_count}",
        f"- Total distributions/resources: {distribution_total}",
        f"- Unique tags/keywords: {unique_tags}",
        f"- Normalized terms: {normalized_terms}",
        "",
        "### Top Formats",
        "",
        value_counts_table(metadata, "distribution_formats"),
        "",
        "### Top Media Types",
        "",
        value_counts_table(metadata, "distribution_media_types"),
        "",
        "### Access Level Distribution",
        "",
        value_counts_table(metadata, "dcat_accessLevel"),
        "",
        "## Terminology Heterogeneity",
        "",
        "## Top Duplicate Terms",
        "",
        table_from_frame(top_duplicates(terms), ["normalized_term", "count", "source_field"]),
        "",
        "## Top Semantic Clusters",
        "",
        table_from_frame(top_clusters(clusters), ["cluster_id", "canonical_term", "observed_term", "count", "cluster_reason"]),
        "",
        "## Examples Of Same Concept Expressed Differently",
        "",
        cluster_examples(clusters, minimum=5),
        "",
        "## DCAT Field Completeness",
        "",
        table_from_frame(dcat_completeness(metadata), ["field", "count_present", "percent_present", "examples_missing"]),
        "",
        "## Distribution-Level Heterogeneity",
        "",
        distribution_heterogeneity(metadata),
        "",
        case_study_section(metadata, resource_profiles, case_study_dataset_id),
        "",
        "## Limitations",
        "",
        "- Metadata may be stale.",
        "- Similarity clusters are candidate evidence, not final proof.",
        "- Human validation is required.",
        "- This MVP does not download raw data.",
        "- The tool analyzes Data.gov metadata records, not the underlying dataset contents.",
        "",
        "## Next Steps",
        "",
        "- Add human validation of cluster quality.",
        "- Integrate a DCAT-US validator.",
        "- Add optional embedding models for stronger synonym discovery.",
        "- Build larger reproducible snapshots and benchmark corpora.",
        "",
        "## NIW / Research Evidence Note",
        "",
        "This prototype provides early evidence that open government metadata contains semantic heterogeneity across titles, descriptions, keywords, distribution labels, and resource formats. It supports a larger semantic data-understanding system by quantifying metadata inconsistency and producing normalized candidate mappings.",
        "",
    ]
    content = "\n".join(lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return content


def read_optional_csv(path: Path | None) -> pd.DataFrame:
    if path and path.exists():
        return pd.read_csv(path).fillna("")
    return pd.DataFrame()


def count_unique(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return frame[column].astype(str).replace("", pd.NA).dropna().nunique()


def unique_split_count(values: list[object]) -> int:
    unique: set[str] = set()
    for value in values:
        for part in str(value).split("|"):
            clean = part.strip()
            if clean:
                unique.add(clean)
    return len(unique)


def top_duplicates(terms: pd.DataFrame) -> pd.DataFrame:
    if terms.empty:
        return terms
    grouped = terms.groupby(["normalized_term", "source_field"], as_index=False)["count"].sum()
    return grouped.sort_values("count", ascending=False).head(10)


def top_clusters(clusters: pd.DataFrame) -> pd.DataFrame:
    if clusters.empty:
        return clusters
    grouped = clusters.groupby(["cluster_id", "canonical_term"], as_index=False).agg(
        observed_term=("observed_term", lambda values: " | ".join(str(value) for value in values[:5])),
        count=("count", "sum"),
        cluster_reason=("cluster_reason", lambda values: " | ".join(sorted(set(str(value) for value in values)))),
    )
    grouped = grouped[grouped["observed_term"].str.contains(r"\|", regex=True)]
    return grouped.sort_values("count", ascending=False).head(10)


def dcat_completeness(metadata: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "title": ["title", "dcat_title"],
        "description": ["notes", "dcat_description"],
        "identifier": ["dcat_identifier", "id"],
        "accessLevel": ["dcat_accessLevel"],
        "publisher": ["dcat_publisher_name", "organization_name"],
        "contactPoint": ["dcat_contactPoint_hasEmail", "dcat_contactPoint_fn"],
        "keyword": ["dcat_keyword", "tags"],
        "modified": ["dcat_modified", "metadata_modified"],
        "distribution": ["distribution_count", "resources_count"],
        "downloadURL/accessURL": ["distribution_download_urls", "distribution_access_urls", "resource_urls"],
    }
    rows = []
    total = len(metadata)
    for field, columns in checks.items():
        present_mask = pd.Series(False, index=metadata.index)
        for column in columns:
            if column in metadata:
                values = metadata[column].astype(str).str.strip()
                present_mask = present_mask | values.ne("") & values.ne("0")
        missing = metadata.loc[~present_mask, "title"].astype(str).head(3).tolist() if "title" in metadata else []
        rows.append(
            {
                "field": field,
                "count_present": int(present_mask.sum()),
                "percent_present": f"{(present_mask.sum() / total * 100 if total else 0):.1f}%",
                "examples_missing": " | ".join(missing),
            }
        )
    return pd.DataFrame(rows)


def distribution_heterogeneity(metadata: pd.DataFrame) -> str:
    examples = []
    for column, label in [
        ("distribution_formats", "Formats"),
        ("distribution_media_types", "Media types"),
        ("distribution_titles", "Distribution labels"),
    ]:
        counts = split_counts(metadata, column)
        if counts:
            examples.append(f"- {label}: " + ", ".join(f"`{key}` ({value})" for key, value in counts[:8]))
    return "\n".join(examples) if examples else "_No distribution fields available._"


def value_counts_table(metadata: pd.DataFrame, column: str) -> str:
    counts = split_counts(metadata, column)
    if not counts:
        return "_No values available._"
    frame = pd.DataFrame(counts[:10], columns=[column, "count"])
    return table_from_frame(frame, [column, "count"])


def split_counts(metadata: pd.DataFrame, column: str) -> list[tuple[str, int]]:
    if column not in metadata:
        return []
    counts: dict[str, int] = {}
    for value in metadata[column].astype(str):
        for part in value.split("|"):
            clean = part.strip()
            if clean:
                counts[clean] = counts.get(clean, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def case_study_section(metadata: pd.DataFrame, resource_profiles: pd.DataFrame, dataset_id: str) -> str:
    if not dataset_id:
        return "## Case Study\n\n_No case-study dataset id provided._"
    if "id" not in metadata:
        return "## Case Study\n\n_Metadata file has no `id` column._"
    matches = metadata[metadata["id"].astype(str) == str(dataset_id)]
    if matches.empty:
        return f"## Case Study\n\n_No metadata row found for `{dataset_id}`._"
    row = matches.iloc[0]
    lines = [
        "## Case Study",
        "",
        f"- Dataset id: `{dataset_id}`",
        f"- Title: {row.get('title', '')}",
        f"- Description: {row.get('notes', '')}",
        f"- Keywords: {row.get('tags', row.get('dcat_keyword', ''))}",
        f"- Distributions: {row.get('distribution_count', row.get('resources_count', ''))}",
        f"- Resource formats: {row.get('distribution_formats', row.get('resource_formats', ''))}",
        "",
        "Catalog metadata reveals dataset-level topic, publisher, keywords, and resource pointers. Metadata alone does not reveal the actual columns, sheets, field roles, dashboard actions, or deeper resource semantics.",
    ]
    if not resource_profiles.empty and "dataset_id" in resource_profiles:
        profiles = resource_profiles[resource_profiles["dataset_id"].astype(str) == str(dataset_id)]
        if not profiles.empty:
            lines.extend(["", "Resource-level profiling reveals:"])
            for _, profile in profiles.head(5).iterrows():
                lines.append(
                    f"- `{profile.get('distribution_title', '')}` columns: {profile.get('detected_columns', '')}; roles: {profile.get('inferred_time_fields', '')} {profile.get('inferred_geo_fields', '')} {profile.get('inferred_metric_fields', '')}".strip()
                )
    return "\n".join(lines)


def table_from_frame(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows available._"
    available = [column for column in columns if column in frame]
    header = "| " + " | ".join(available) + " |"
    separator = "| " + " | ".join(["---"] * len(available)) + " |"
    rows = []
    for _, row in frame[available].iterrows():
        rows.append("| " + " | ".join(escape_md(row[column]) for column in available) + " |")
    return "\n".join([header, separator, *rows])


def cluster_examples(clusters: pd.DataFrame, *, minimum: int) -> str:
    if clusters.empty:
        return "_No cluster examples available._"
    grouped = top_clusters(clusters)
    if len(grouped) < minimum:
        grouped = clusters.groupby(["cluster_id", "canonical_term"], as_index=False).agg(
            observed_term=("observed_term", lambda values: " | ".join(str(value) for value in values[:5])),
            count=("count", "sum"),
            cluster_reason=("cluster_reason", lambda values: " | ".join(sorted(set(str(value) for value in values)))),
        ).sort_values("count", ascending=False).head(minimum)
    lines = []
    for _, row in grouped.head(max(minimum, 5)).iterrows():
        lines.append(
            f"- `{escape_md(row['canonical_term'])}` appears as: {escape_md(row['observed_term'])}."
        )
    return "\n".join(lines) if lines else "_No cluster examples available._"


def escape_md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
