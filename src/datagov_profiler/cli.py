from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from datagov_profiler.cluster import cluster_terms, format_cluster_metrics
from datagov_profiler.dcat_mapping import map_dcat_file
from datagov_profiler.domain_summary import generate_domain_summary
from datagov_profiler.fetch import fetch_catalog_records, fetch_records
from datagov_profiler.flatten import flatten_distributions_file, flatten_file
from datagov_profiler.harvest import fetch_harvest_records
from datagov_profiler.report import generate_report
from datagov_profiler.resources import profile_resources
from datagov_profiler.terms import profile_terms
from datagov_profiler.validation import create_validation_sample, score_validation

app = typer.Typer(help="Profile semantic heterogeneity in Data.gov metadata records.")
console = Console()


@app.command()
def fetch(
    rows: Annotated[int, typer.Option(help="Total number of metadata records to request.")] = 1000,
    page_size: Annotated[int, typer.Option(help="Catalog results per request.")] = 100,
    query: Annotated[str, typer.Option(help="Catalog search query.")] = "*:*",
    out: Annotated[Path, typer.Option(help="Output JSONL path.")] = Path("data/raw/datagov_metadata_sample.jsonl"),
    sleep: Annotated[float, typer.Option(help="Seconds to sleep between requests.")] = 0.2,
) -> None:
    """Fetch Data.gov Catalog API records as JSONL."""
    count = fetch_records(rows=rows, page_size=page_size, query=query, out=out, sleep_seconds=sleep)
    if count == 0:
        console.print("[red]No records were fetched. Check the API endpoint or query.[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Wrote {count} metadata records to {out}[/green]")


@app.command("fetch-catalog")
def fetch_catalog(
    rows: Annotated[int, typer.Option(help="Total number of catalog search results to request.")] = 1000,
    per_page: Annotated[int, typer.Option(help="Catalog results per page.")] = 100,
    query: Annotated[str, typer.Option(help="Catalog search query.")] = "",
    out: Annotated[Path, typer.Option(help="Output JSONL path.")] = Path("data/raw/catalog_search_results.jsonl"),
    sleep: Annotated[float, typer.Option(help="Seconds to sleep between requests.")] = 0.2,
) -> None:
    """Fetch Data.gov Catalog API search results as JSONL."""
    count = fetch_catalog_records(rows, per_page, query, out, sleep)
    if count == 0:
        console.print("[red]No records were fetched. Check the API endpoint or query.[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Wrote {count} catalog search results to {out}[/green]")


@app.command("fetch-harvest-raw")
def fetch_harvest_raw(
    input: Annotated[Path, typer.Option("--input", help="Input JSONL from fetch-catalog or fetch.")],
    out_dir: Annotated[Path, typer.Option(help="Directory for raw harvest payloads.")] = Path("data/raw/harvest_records"),
    limit: Annotated[int, typer.Option(help="Maximum records to inspect.")] = 100,
    sleep: Annotated[float, typer.Option(help="Seconds to sleep between requests.")] = 0.2,
) -> None:
    """Fetch raw publisher harvest records referenced by catalog metadata."""
    frame = fetch_harvest_records(
        input,
        out_dir,
        index_out=Path("reports/harvest_raw_index.csv"),
        limit=limit,
        sleep_seconds=sleep,
    )
    console.print(f"[green]Wrote harvest raw index with {len(frame)} rows[/green]")


@app.command("fetch-harvest-transformed")
def fetch_harvest_transformed(
    input: Annotated[Path, typer.Option("--input", help="Input JSONL from fetch-catalog or fetch.")],
    out_dir: Annotated[Path, typer.Option(help="Directory for transformed harvest payloads.")] = Path("data/raw/harvest_transformed"),
    limit: Annotated[int, typer.Option(help="Maximum records to inspect.")] = 100,
    sleep: Annotated[float, typer.Option(help="Seconds to sleep between requests.")] = 0.2,
) -> None:
    """Fetch transformed harvest records when discoverable in metadata."""
    frame = fetch_harvest_records(
        input,
        out_dir,
        index_out=Path("reports/harvest_transformed_index.csv"),
        limit=limit,
        sleep_seconds=sleep,
        transformed=True,
    )
    console.print(f"[green]Wrote harvest transformed index with {len(frame)} rows[/green]")


@app.command()
def flatten(
    input: Annotated[Path, typer.Option("--input", help="Input JSON or JSONL metadata file.")],
    out: Annotated[Path, typer.Option(help="Output flattened CSV path.")] = Path("data/processed/metadata_flat.csv"),
) -> None:
    """Flatten raw metadata into a tabular CSV."""
    frame = flatten_file(input, out)
    console.print(f"[green]Wrote {len(frame)} flattened records to {out}[/green]")


@app.command("flatten-distributions")
def flatten_distributions(
    input: Annotated[Path, typer.Option("--input", help="Input JSON or JSONL metadata file.")],
    out: Annotated[Path, typer.Option(help="Output distribution CSV path.")] = Path("data/processed/distributions_flat.csv"),
) -> None:
    """Flatten DCAT/CKAN distributions into one row per resource."""
    frame = flatten_distributions_file(input, out)
    console.print(f"[green]Wrote {len(frame)} flattened distributions to {out}[/green]")


@app.command("profile-terms")
def profile_terms_command(
    metadata: Annotated[
        Path | None,
        typer.Option("--metadata", "--input", help="Input flattened metadata CSV."),
    ] = None,
    distributions: Annotated[Path | None, typer.Option(help="Optional flattened distributions CSV.")] = None,
    out: Annotated[Path, typer.Option(help="Output term frequency CSV path.")] = Path("reports/term_frequency.csv"),
) -> None:
    """Extract and normalize terms from flattened metadata."""
    if metadata is None:
        console.print("[red]Missing option '--metadata' (or backward-compatible '--input').[/red]")
        raise typer.Exit(2)
    if not metadata.exists():
        console.print(f"[red]Input file does not exist: {metadata}[/red]")
        raise typer.Exit(1)
    if distributions is not None and not distributions.exists():
        console.print(f"[red]Input file does not exist: {distributions}[/red]")
        raise typer.Exit(1)
    frame = profile_terms(metadata, out, distributions_csv=distributions)
    console.print(f"[green]Wrote {len(frame)} term rows to {out}[/green]")


@app.command("cluster-terms")
def cluster_terms_command(
    input: Annotated[Path, typer.Option("--input", help="Input term frequency CSV.")],
    out: Annotated[Path, typer.Option(help="Output semantic clusters CSV path.")] = Path("reports/semantic_term_clusters.csv"),
    method: Annotated[str, typer.Option(help="Clustering method: fuzzy, tfidf, hybrid, embedding.")] = "hybrid",
    min_count: Annotated[int, typer.Option(help="Minimum term count to include.")] = 2,
    similarity_threshold: Annotated[float, typer.Option(help="Similarity threshold from 0 to 1.")] = 0.82,
    domain: Annotated[str, typer.Option(help="Optional domain mode, e.g. sensor_telemetry.")] = "",
) -> None:
    """Cluster exact, fuzzy, and semantically similar terms."""
    allowed = {"fuzzy", "tfidf", "hybrid", "embedding"}
    if method not in allowed:
        raise typer.BadParameter(f"method must be one of: {', '.join(sorted(allowed))}")
    if not input.exists():
        console.print(f"[red]Input file does not exist: {input}[/red]")
        raise typer.Exit(1)
    frame = cluster_terms(
        input,
        out,
        method=method,
        min_count=min_count,
        similarity_threshold=similarity_threshold,
        domain=domain,
    )
    console.print(f"[green]Wrote {len(frame)} cluster rows to {out}[/green]")
    for line in format_cluster_metrics(frame.attrs["metrics"]):
        if line.startswith("WARNING"):
            console.print(f"[yellow]{line}[/yellow]")
        else:
            console.print(line)


@app.command("domain-summary")
def domain_summary(
    terms: Annotated[Path, typer.Option(help="Input term frequency CSV.")],
    clusters: Annotated[Path, typer.Option(help="Input semantic clusters CSV.")],
    domain: Annotated[str, typer.Option(help="Domain name. Currently: sensor_telemetry.")],
    out: Annotated[Path, typer.Option(help="Output Markdown domain summary.")],
) -> None:
    """Generate a focused domain summary from terms and clusters."""
    generate_domain_summary(terms, clusters, domain=domain, out=out)
    console.print(f"[green]Wrote domain summary to {out}[/green]")


@app.command("map-dcat")
def map_dcat(
    input: Annotated[Path, typer.Option("--input", help="Input JSON, JSONL, or CSV metadata file.")],
    out: Annotated[Path, typer.Option(help="Output DCAT mapping suggestions CSV path.")] = Path("reports/dcat_mapping_suggestions.csv"),
) -> None:
    """Suggest DCAT-US mappings for messy or custom metadata fields."""
    frame = map_dcat_file(input, out)
    console.print(f"[green]Wrote {len(frame)} DCAT mapping suggestions to {out}[/green]")


@app.command("profile-resources")
def profile_resources_command(
    distributions: Annotated[Path, typer.Option(help="Flattened distributions CSV.")] = Path("data/processed/distributions_flat.csv"),
    out: Annotated[Path, typer.Option(help="Output resource schema profile CSV.")] = Path("reports/resource_schema_profiles.csv"),
    download_dir: Annotated[Path, typer.Option(help="Directory for opt-in resource samples.")] = Path("data/raw/resource_samples"),
    allowed_formats: Annotated[str, typer.Option(help="Comma-separated allowed resource formats.")] = "CSV,XLSX",
    max_files: Annotated[int, typer.Option(help="Maximum resource files to profile.")] = 10,
    max_file_size_mb: Annotated[int, typer.Option(help="Maximum downloaded resource size in MB.")] = 25,
    timeout: Annotated[int, typer.Option(help="HTTP timeout in seconds.")] = 30,
    allow_zip: Annotated[bool, typer.Option(help="Opt in to ZIP download and safe manifest inspection.")] = False,
    max_zip_size_mb: Annotated[int, typer.Option(help="Maximum ZIP size in MB.")] = 50,
    max_files_per_zip: Annotated[int, typer.Option(help="Maximum inner ZIP entries to inspect.")] = 20,
    extract_dir: Annotated[Path, typer.Option(help="Directory for selected safe ZIP members.")] = Path("data/raw/resource_samples/extracted"),
    inspect_zip_only: Annotated[bool, typer.Option(help="Inspect ZIP manifest first; only selected safe CSV/XLSX members are profiled.")] = True,
) -> None:
    """Opt-in schema profiling for CSV/XLSX resources and explicit safe ZIP inspection."""
    frame = profile_resources(
        distributions,
        out,
        download_dir=download_dir,
        allowed_formats=allowed_formats,
        max_files=max_files,
        max_file_size_mb=max_file_size_mb,
        timeout=timeout,
        allow_zip=allow_zip,
        max_zip_size_mb=max_zip_size_mb,
        max_files_per_zip=max_files_per_zip,
        extract_dir=extract_dir,
        inspect_zip_only=inspect_zip_only,
    )
    console.print(f"[green]Wrote {len(frame)} resource schema profile rows to {out}[/green]")


@app.command("create-validation-sample")
def create_validation_sample_command(
    clusters: Annotated[Path, typer.Option(help="Semantic clusters CSV.")] = Path("reports/semantic_term_clusters.csv"),
    out: Annotated[Path, typer.Option(help="Manual validation sample CSV.")] = Path("reports/manual_validation_sample.csv"),
    sample_size: Annotated[int, typer.Option(help="Number of clusters to sample.")] = 100,
) -> None:
    """Create a human-review CSV for semantic cluster validation."""
    frame = create_validation_sample(clusters, out, sample_size=sample_size)
    console.print(f"[green]Wrote {len(frame)} validation rows to {out}[/green]")


@app.command("score-validation")
def score_validation_command(
    input: Annotated[Path, typer.Option("--input", help="Manual validation CSV with human_label values.")],
    out: Annotated[Path, typer.Option(help="Markdown validation score output.")] = Path("reports/manual_validation_scores.md"),
) -> None:
    """Score a completed manual validation sample."""
    score_validation(input, out)
    console.print(f"[green]Wrote validation scores to {out}[/green]")


@app.command()
def report(
    metadata: Annotated[Path, typer.Option(help="Flattened metadata CSV.")],
    terms: Annotated[Path, typer.Option(help="Term frequency CSV.")],
    clusters: Annotated[Path, typer.Option(help="Semantic clusters CSV.")],
    distributions: Annotated[Path | None, typer.Option(help="Optional flattened distributions CSV.")] = None,
    resource_profiles: Annotated[Path | None, typer.Option(help="Optional resource schema profiles CSV.")] = None,
    case_study_dataset_id: Annotated[str, typer.Option(help="Optional dataset id for a focused case study section.")] = "",
    out: Annotated[Path, typer.Option(help="Output Markdown report path.")] = Path("reports/metadata_heterogeneity_report.md"),
) -> None:
    """Generate a Markdown metadata heterogeneity report."""
    generate_report(
        metadata,
        terms,
        clusters,
        out,
        distributions_path=distributions,
        resource_profiles_path=resource_profiles,
        case_study_dataset_id=case_study_dataset_id,
    )
    console.print(f"[green]Wrote report to {out}[/green]")


if __name__ == "__main__":
    app()
