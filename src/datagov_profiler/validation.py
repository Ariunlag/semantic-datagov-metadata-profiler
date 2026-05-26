from __future__ import annotations

from pathlib import Path

import pandas as pd


VALID_LABELS = {"equivalent", "related", "not_equivalent", "unsure"}


def create_validation_sample(clusters_path: Path, out: Path, *, sample_size: int) -> pd.DataFrame:
    clusters = pd.read_csv(clusters_path).fillna("")
    grouped = clusters.groupby(["cluster_id", "canonical_term"], as_index=False).agg(
        observed_terms=("observed_term", lambda values: " | ".join(sorted(set(str(value) for value in values)))),
        example_records=("examples", lambda values: " | ".join(str(value) for value in values if str(value))[:500]),
    )
    grouped = grouped.head(sample_size)
    grouped["predicted_equivalent"] = True
    grouped["human_label"] = ""
    grouped["human_notes"] = ""
    out.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out, index=False)
    return grouped


def score_validation(input_path: Path, out: Path) -> str:
    frame = pd.read_csv(input_path).fillna("")
    labels = frame["human_label"].astype(str).str.strip().str.lower() if "human_label" in frame else pd.Series(dtype=str)
    reviewed = labels[labels.isin(VALID_LABELS)]
    counts = {label: int((reviewed == label).sum()) for label in sorted(VALID_LABELS)}
    useful = counts["equivalent"] + counts["related"]
    precision = useful / len(reviewed) if len(reviewed) else 0.0
    content = "\n".join(
        [
            "# Manual Validation Scores",
            "",
            f"- Number reviewed: {len(reviewed)}",
            f"- Equivalent: {counts['equivalent']}",
            f"- Related: {counts['related']}",
            f"- Not equivalent: {counts['not_equivalent']}",
            f"- Unsure: {counts['unsure']}",
            f"- Estimated useful precision: {precision:.2%}",
            "",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return content
