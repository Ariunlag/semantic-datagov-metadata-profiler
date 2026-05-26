from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from datagov_profiler.normalize import SYNONYM_CANONICALS, compact_list

CLUSTER_COLUMNS = [
    "cluster_id",
    "canonical_term",
    "observed_term",
    "normalized_term",
    "source_fields",
    "count",
    "similarity_score",
    "cluster_reason",
    "examples",
]

EXCLUDED_TERMS = {
    "access url",
    "download url",
    "original metadata",
    "web page",
    "text html",
    "application json",
    "application zip",
    "application octet stream",
    "this dataset",
    "digital data",
    "united states",
    "department",
    "data gov",
}

DOMAIN_TERMS = {
    "sensor",
    "sensors",
    "telemetry",
    "monitoring",
    "streamgage",
    "stream gage",
    "streamflow",
    "real time",
    "realtime",
    "device",
    "station",
    "uas",
    "uav",
    "drone",
    "observation",
    "environmental monitoring",
    "water quality",
}

NOISY_STREAM_TERMS = {"application octet stream", "octet stream", "stream"}
BLOCK_STOP_TOKENS = {
    "data",
    "dataset",
    "datasets",
    "file",
    "files",
    "resource",
    "resources",
    "service",
    "services",
    "state",
    "states",
    "county",
    "city",
    "public",
    "national",
    "information",
    "metadata",
    "system",
    "systems",
    "program",
    "project",
    "report",
    "reports",
    "map",
    "maps",
}


@dataclass(frozen=True)
class ClusterMetrics:
    total_terms: int
    total_clusters: int
    largest_cluster_size: int
    singleton_clusters: int
    multi_term_clusters: int
    warning: str = ""


@dataclass(frozen=True)
class ClusterEdge:
    left: str
    right: str
    score: float
    reason: str


def cluster_terms(
    input_csv: Path,
    out: Path,
    *,
    method: str = "hybrid",
    min_count: int = 2,
    similarity_threshold: float = 0.82,
    domain: str = "",
    include_singletons: bool = True,
) -> pd.DataFrame:
    frame = pd.read_csv(input_csv).fillna("")
    if "count" in frame:
        frame = frame[frame["count"].astype(int) >= min_count]
    frame = filter_terms_frame(frame, domain=domain)
    terms = sorted({str(value) for value in frame["normalized_term"].tolist() if str(value).strip()})
    edges = build_edges(terms, method=method, similarity_threshold=similarity_threshold)
    assignments = connected_components(terms, edges)
    assignments = split_oversized_components(assignments, max_cluster_terms=500)
    rows = build_cluster_rows(frame, assignments, edges)
    result = pd.DataFrame(rows, columns=CLUSTER_COLUMNS).sort_values(
        ["cluster_id", "count", "observed_term"], ascending=[True, False, True]
    )
    if not include_singletons and not result.empty:
        sizes = result.groupby("cluster_id")["normalized_term"].transform("nunique")
        result = result[sizes > 1]
    result.attrs["metrics"] = cluster_quality_metrics(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    return result


def filter_terms_frame(frame: pd.DataFrame, *, domain: str = "") -> pd.DataFrame:
    if frame.empty or "normalized_term" not in frame:
        return frame
    filtered = frame.copy()
    terms = filtered["normalized_term"].astype(str)
    mask = terms.map(lambda value: not should_exclude_term(value))
    if domain == "sensor_telemetry":
        mask = mask & terms.map(is_sensor_telemetry_term)
    return filtered[mask]


def should_exclude_term(term: str) -> bool:
    value = normalize_for_filter(term)
    if len(value) < 3:
        return True
    if value in EXCLUDED_TERMS:
        return True
    if is_uuid_like(value) or is_date_fragment(value):
        return True
    return False


def normalize_for_filter(term: str) -> str:
    return re.sub(r"\s+", " ", str(term).lower().replace("/", " ").replace("-", " ")).strip()


def is_uuid_like(term: str) -> bool:
    return bool(re.fullmatch(r"[a-f0-9]{8}([ -]?[a-f0-9]{4}){3}[ -]?[a-f0-9]{12}", term))


def is_date_fragment(term: str) -> bool:
    return bool(
        re.fullmatch(r"\d{4}", term)
        or re.fullmatch(r"\d{4}\s+\d{1,2}\s+\d{1,2}", term)
        or re.fullmatch(r"\d{1,2}\s+\d{1,2}\s+\d{2,4}", term)
    )


def is_sensor_telemetry_term(term: str) -> bool:
    value = normalize_for_filter(term)
    if value in NOISY_STREAM_TERMS:
        return False
    return any(keyword in value for keyword in DOMAIN_TERMS)


def build_edges(terms: list[str], *, method: str, similarity_threshold: float) -> list[ClusterEdge]:
    edges: list[ClusterEdge] = []
    canonical_pairs = {
        tuple(sorted((alias, canonical)))
        for alias, canonical in SYNONYM_CANONICALS.items()
        if alias in terms and canonical in terms and alias != canonical
    }
    for left, right in canonical_pairs:
        edges.append(ClusterEdge(left, right, 1.0, "hybrid" if method == "hybrid" else "exact"))

    if method in {"fuzzy", "hybrid"}:
        for block in comparison_blocks(terms):
            for i, left in enumerate(block):
                for right in block[i + 1 :]:
                    if not can_compare_terms(left, right):
                        continue
                    score = fuzz.token_set_ratio(left, right) / 100
                    if score >= similarity_threshold:
                        edges.append(ClusterEdge(left, right, score, "fuzzy" if method == "fuzzy" else "hybrid"))

    if method in {"tfidf", "hybrid"} and len(terms) <= 50_000:
        for block in comparison_blocks(terms):
            edges.extend(tfidf_edges(block, similarity_threshold, reason="tfidf" if method == "tfidf" else "hybrid"))

    if method == "embedding":
        edges.extend(embedding_edges(terms, similarity_threshold))
    return dedupe_edges(edges)


def tfidf_edges(terms: list[str], threshold: float, *, reason: str) -> list[ClusterEdge]:
    if len(terms) < 2:
        return []
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    matrix = vectorizer.fit_transform(terms)
    scores = cosine_similarity(matrix)
    edges: list[ClusterEdge] = []
    for i, left in enumerate(terms):
        for j in range(i + 1, len(terms)):
            if not can_compare_terms(left, terms[j]):
                continue
            score = float(scores[i, j])
            if score >= threshold:
                edges.append(ClusterEdge(left, terms[j], score, reason))
    return edges


def comparison_blocks(terms: list[str], *, max_block_size: int = 150) -> list[list[str]]:
    blocks: dict[str, set[str]] = {}
    for term in terms:
        tokens = term.split()
        keys = block_keys(term, tokens)
        for key in keys:
            blocks.setdefault(key, set()).add(term)
    result: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for values in blocks.values():
        if len(values) < 2:
            continue
        ordered = tuple(sorted(values))
        if ordered in seen:
            continue
        seen.add(ordered)
        if len(ordered) <= max_block_size:
            result.append(list(ordered))
            continue
        subblocks: dict[str, list[str]] = {}
        for term in ordered:
            tokens = [token for token in term.split() if token not in BLOCK_STOP_TOKENS]
            key = (tokens[0][:6] if tokens else term[:6])
            subblocks.setdefault(key, []).append(term)
        for values in subblocks.values():
            if 1 < len(values) <= max_block_size:
                result.append(values)
    return result


def block_keys(term: str, tokens: list[str]) -> set[str]:
    keys = {token for token in tokens if token not in BLOCK_STOP_TOKENS}
    if term in SYNONYM_CANONICALS:
        keys.add(SYNONYM_CANONICALS[term])
    for alias, canonical in SYNONYM_CANONICALS.items():
        if term == canonical:
            keys.add(alias)
    if tokens:
        keys.add(tokens[0])
    return {key for key in keys if len(key) >= 3}


def embedding_edges(terms: list[str], threshold: float) -> list[ClusterEdge]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install semantic-datagov-metadata-profiler[embeddings] to use --method embedding") from exc
    if len(terms) < 2:
        return []
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(terms, normalize_embeddings=True)
    scores = cosine_similarity(embeddings)
    edges: list[ClusterEdge] = []
    for i, left in enumerate(terms):
        for j in range(i + 1, len(terms)):
            score = float(scores[i, j])
            if score >= threshold:
                edges.append(ClusterEdge(left, terms[j], score, "embedding"))
    return edges


def connected_components(terms: list[str], edges: list[ClusterEdge]) -> dict[str, int]:
    parent = {term: term for term in terms}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for edge in edges:
        union(edge.left, edge.right)
    grouped_roots = sorted({find(term) for term in terms})
    root_to_id = {root: index + 1 for index, root in enumerate(grouped_roots)}
    return {term: root_to_id[find(term)] for term in terms}


def split_oversized_components(assignments: dict[str, int], *, max_cluster_terms: int) -> dict[str, int]:
    grouped: dict[int, list[str]] = {}
    for term, cluster_id in assignments.items():
        grouped.setdefault(cluster_id, []).append(term)
    next_id = max(grouped, default=0) + 1
    fixed: dict[str, int] = {}
    for cluster_id, terms in grouped.items():
        if len(terms) <= max_cluster_terms:
            for term in terms:
                fixed[term] = cluster_id
            continue
        for term in terms:
            fixed[term] = next_id
            next_id += 1
    renumber = {cluster_id: index + 1 for index, cluster_id in enumerate(sorted(set(fixed.values())))}
    return {term: renumber[cluster_id] for term, cluster_id in fixed.items()}


def can_compare_terms(left: str, right: str) -> bool:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if left in EXCLUDED_TERMS or right in EXCLUDED_TERMS:
        return False
    if left in SYNONYM_CANONICALS and SYNONYM_CANONICALS[left] == right:
        return True
    if right in SYNONYM_CANONICALS and SYNONYM_CANONICALS[right] == left:
        return True
    if len(left_tokens & right_tokens) == 0:
        return False
    if min(len(left), len(right)) < 5 and left != right:
        return False
    if len(left_tokens) == 1 or len(right_tokens) == 1:
        return left_tokens == right_tokens
    overlap = len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
    length_ratio = min(len(left), len(right)) / max(len(left), len(right))
    if overlap < 0.67 or length_ratio < 0.55:
        return False
    return True


def build_cluster_rows(
    frame: pd.DataFrame, assignments: dict[str, int], edges: list[ClusterEdge]
) -> list[dict[str, object]]:
    score_lookup = edge_lookup(edges)
    grouped = frame.groupby("normalized_term", dropna=False)
    cluster_terms_map: dict[int, list[str]] = {}
    for normalized in assignments:
        cluster_terms_map.setdefault(assignments[normalized], []).append(normalized)
    canonical = {
        cluster_id: choose_canonical(values, grouped)
        for cluster_id, values in cluster_terms_map.items()
        if value_in_group(values, grouped)
    }
    rows: list[dict[str, object]] = []
    for normalized, group in grouped:
        normalized = str(normalized)
        if normalized not in assignments:
            continue
        cluster_id = assignments[normalized]
        source_fields = compact_list([str(value) for value in group["source_field"].tolist()], limit=10)
        observed_terms = compact_list([str(value) for value in group["term"].tolist()], limit=10)
        canonical_term = canonical.get(cluster_id, normalized)
        score, reason = score_lookup.get(tuple(sorted((canonical_term, normalized))), (1.0, "exact"))
        rows.append(
            {
                "cluster_id": cluster_id,
                "canonical_term": canonical_term,
                "observed_term": observed_terms or normalized,
                "normalized_term": normalized,
                "source_fields": source_fields,
                "count": int(group["count"].astype(int).sum()),
                "similarity_score": round(float(score), 3),
                "cluster_reason": reason if canonical_term != normalized else "exact",
                "examples": compact_list([str(value) for value in group["example_titles"].tolist()], limit=5),
            }
        )
    return rows


def value_in_group(values: list[str], grouped: pd.core.groupby.generic.DataFrameGroupBy) -> bool:
    return any(value in grouped.groups for value in values)


def choose_canonical(values: list[str], grouped: pd.core.groupby.generic.DataFrameGroupBy) -> str:
    preferred = sorted(set(SYNONYM_CANONICALS.values()) & set(values))
    if preferred:
        return preferred[0]
    return sorted(values, key=lambda value: (-int(grouped.get_group(value)["count"].sum()), value))[0]


def edge_lookup(edges: list[ClusterEdge]) -> dict[tuple[str, str], tuple[float, str]]:
    lookup: dict[tuple[str, str], tuple[float, str]] = {}
    for edge in edges:
        key = tuple(sorted((edge.left, edge.right)))
        current = lookup.get(key, (0.0, "exact"))
        if edge.score >= current[0]:
            lookup[key] = (edge.score, edge.reason)
    return lookup


def dedupe_edges(edges: list[ClusterEdge]) -> list[ClusterEdge]:
    lookup: dict[tuple[str, str], ClusterEdge] = {}
    for edge in edges:
        left, right = sorted((edge.left, edge.right))
        if left == right:
            continue
        key = (left, right)
        current = lookup.get(key)
        if current is None or edge.score > current.score:
            lookup[key] = ClusterEdge(left, right, edge.score, edge.reason)
    return list(lookup.values())


def cluster_quality_metrics(result: pd.DataFrame) -> ClusterMetrics:
    if result.empty:
        return ClusterMetrics(0, 0, 0, 0, 0)
    sizes = result.groupby("cluster_id")["normalized_term"].nunique()
    total_rows = len(result)
    largest = int(sizes.max()) if not sizes.empty else 0
    warning = ""
    if total_rows and largest > total_rows * 0.05:
        warning = f"largest cluster has {largest} rows (>5% of clustered rows)"
    return ClusterMetrics(
        total_terms=int(result["normalized_term"].nunique()),
        total_clusters=int(sizes.size),
        largest_cluster_size=largest,
        singleton_clusters=int((sizes == 1).sum()),
        multi_term_clusters=int((sizes > 1).sum()),
        warning=warning,
    )


def format_cluster_metrics(metrics: ClusterMetrics) -> list[str]:
    lines = [
        f"total terms: {metrics.total_terms}",
        f"total clusters: {metrics.total_clusters}",
        f"largest cluster size: {metrics.largest_cluster_size}",
        f"singleton clusters: {metrics.singleton_clusters}",
        f"multi-term clusters: {metrics.multi_term_clusters}",
    ]
    if metrics.warning:
        lines.append(f"WARNING: {metrics.warning}")
    return lines
