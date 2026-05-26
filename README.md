# semantic-datagov-metadata-profiler

A small open-source Python tool for profiling semantic heterogeneity in Data.gov metadata records.

Data.gov and the Data.gov CKAN API expose metadata about datasets, including titles, descriptions, tags, organizations, resource labels, formats, media types, access URLs, and download URLs. They do not expose or guarantee access to the raw contents of every dataset through the metadata API. This project deliberately analyzes metadata only.

## Problem Statement

Open data catalogs often describe similar concepts with different words. One agency may publish a dataset about "traffic crashes"; another may use "motor vehicle collisions"; a third may describe "road accidents." These differences make discovery, integration, semantic search, governance, and standards alignment harder.

This MVP collects a configurable sample of Data.gov metadata records and creates early evidence of semantic heterogeneity across titles, descriptions, tags, organizations, publishers, resource labels, formats, media types, and URL labels.

## Data.gov And DCAT-US Background

This tool preserves an important boundary: it analyzes Data.gov metadata, not raw dataset files.

Official references:

- [Data.gov CKAN API dataset page](https://catalog.data.gov/dataset/data-gov-ckan-api)
- [GSA Data.gov API documentation](https://open.gsa.gov/api/datadotgov/)
- [Data.gov open data how-to: harvested agency metadata source files and stored records](https://resources.data.gov/resources/data-gov-open-data-howto/)
- [DCAT-US / Project Open Data metadata field mapping](https://resources.data.gov/resources/podm-field-mapping/)

Data.gov harvests metadata from agency source files and displays stored catalog records. The fields in those records are shaped by CKAN, Project Open Data, DCAT-US mappings, and agency publishing practices. That makes the catalog a useful research surface for studying metadata vocabulary variation without downloading the underlying datasets.

## Why Metadata-Only Is Not Enough

Catalog metadata can identify dataset-level topics, publishers, keywords, descriptions, access levels, and resource pointers such as `distribution[].downloadURL` and `distribution[].accessURL`. That is enough to study many kinds of metadata heterogeneity, but it is not enough to understand the actual columns, sheets, field roles, metrics, dashboard actions, or deeper resource semantics inside a dataset.

This prototype therefore has two layers:

1. Catalog metadata profiling: fetch and analyze Data.gov/CKAN/DCAT metadata records without downloading the actual dataset resources.
2. Optional resource schema profiling: only when explicitly requested, download small CSV/XLSX resource samples and infer columns, sheets, field roles, and schema clues.

Resource profiling is opt-in. Normal fetch and metadata profiling commands do not download files behind `downloadURL` or `accessURL`.

## Installation

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Quickstart

```bash
datagov-profiler fetch-catalog --rows 500 --out data/raw/catalog_500.jsonl
datagov-profiler flatten --input data/raw/catalog_500.jsonl --out data/processed/metadata_500.csv
datagov-profiler flatten-distributions --input data/raw/catalog_500.jsonl --out data/processed/distributions_500.csv
datagov-profiler profile-terms --metadata data/processed/metadata_500.csv --distributions data/processed/distributions_500.csv --out reports/terms_500.csv
datagov-profiler cluster-terms --input reports/terms_500.csv --out reports/clusters_500.csv
datagov-profiler report --metadata data/processed/metadata_500.csv --distributions data/processed/distributions_500.csv --terms reports/terms_500.csv --clusters reports/clusters_500.csv --out reports/report_500.md
```

Optional resource workflow:

```bash
datagov-profiler profile-resources --distributions data/processed/distributions_flat.csv --max-files 10 --allowed-formats CSV,XLSX --out reports/resource_schema_profiles.csv
datagov-profiler report --metadata data/processed/metadata_500.csv --distributions data/processed/distributions_500.csv --terms reports/terms_500.csv --clusters reports/clusters_500.csv --resource-profiles reports/resource_schema_profiles.csv --out reports/metadata_and_schema_report.md
```

Compatibility workflow:

```bash
datagov-profiler fetch --rows 1000
datagov-profiler flatten --input data/raw/datagov_metadata_sample.jsonl
datagov-profiler profile-terms --metadata data/processed/metadata_flat.csv
datagov-profiler cluster-terms --input reports/term_frequency.csv
datagov-profiler report --metadata data/processed/metadata_flat.csv --terms reports/term_frequency.csv --clusters reports/semantic_term_clusters.csv
```

Offline fixture workflow:

```bash
datagov-profiler flatten --input tests/fixtures/sample_records.jsonl
datagov-profiler profile-terms --metadata data/processed/metadata_flat.csv
datagov-profiler cluster-terms --input reports/term_frequency.csv --min-count 1
datagov-profiler map-dcat --input tests/fixtures/sample_records.jsonl
datagov-profiler report --metadata data/processed/metadata_flat.csv --terms reports/term_frequency.csv --clusters reports/semantic_term_clusters.csv
```

## Commands

### `fetch`

Fetches records from the current Data.gov Catalog API search endpoint:

`https://catalog.data.gov/search`

It uses `q`, `per_page`, and cursor pagination with `after`, writes one catalog result per JSONL line, and sleeps between requests. It does not download files referenced by `downloadURL`, `accessURL`, resource URLs, or any distribution URL.

### `fetch-catalog`

Fetches Data.gov Catalog API search results from `https://catalog.data.gov/search` and preserves the full result object, including `dcat`, publisher, harvest provenance, harvest raw links, and distribution/resource pointers when present.

### `fetch-harvest-raw` and `fetch-harvest-transformed`

Fetch raw or transformed harvest payloads referenced by catalog metadata. These commands save JSON, XML, or text payloads and write index CSV files. They are intended for comparing publisher source metadata, transformed Data.gov metadata, and search API DCAT objects.

### `flatten`

Converts JSONL metadata records into a CSV with key fields such as title, notes, organization, tags, groups, extras, resource titles, formats, media types, URLs, and DCAT-like fields when present.

### `flatten-distributions`

Creates a long-form table with one row per distribution/resource, including title, description, format, media type, download URL, access URL, `conformsTo`, `describedBy`, and `describedByType`.

### `profile-terms`

Extracts terms from flattened metadata and optional flattened distributions, then emits normalized term frequencies with example record IDs and titles. Use `--metadata` for the metadata CSV and optionally `--distributions` for distribution-level labels and formats. The older `--input` option still works as an alias for `--metadata`.

### `cluster-terms`

Groups exact, fuzzy, TF-IDF, or hybrid-similar terms into candidate semantic clusters. `sentence-transformers` can be installed separately for future embedding work, but it is not required.

### `map-dcat`

Suggests canonical DCAT-US-style mappings for messy or custom fields using a seed alias dictionary.

### `report`

Creates a Markdown summary report suitable for early research notes and NIW evidence preparation.

### `profile-resources`

Opt-in CSV/XLSX schema profiling. The command skips non-allowed formats and large files, detects columns/sheets, infers simple field roles, and writes `reports/resource_schema_profiles.csv`. It is never run by normal metadata fetch commands.

## ZIP Resources

ZIP files are common in Data.gov distributions, especially when agencies publish grouped tables, geospatial sidecars, documentation, or exports from legacy systems. The tool does not download or extract ZIP files by default.

ZIP inspection is explicit:

```bash
datagov-profiler profile-resources --allow-zip --max-zip-size-mb 50 --max-files-per-zip 20
```

The first ZIP output is a manifest at `reports/zip_manifest.csv`. The manifest records each inspected inner file name, extension, size, whether it was selected for schema profiling, and the skip reason. The profiler only considers small `.csv`, `.xlsx`, `.json`, `.xml`, `.txt`, and `.geojson` entries, and only CSV/XLSX entries are passed into schema profiling. Shapefile sidecars, images, PDFs, nested ZIPs, unsupported extensions, unsafe paths, and oversized inner files are skipped.

Selected CSV/XLSX members are extracted individually into `data/raw/resource_samples/extracted`. The extraction path is checked to prevent path traversal, and the tool never writes ZIP entries outside the extraction directory. This protects disk space and avoids unsafe blind extraction.

### `create-validation-sample` and `score-validation`

Create a human review CSV for candidate semantic clusters and score reviewed labels (`equivalent`, `related`, `not_equivalent`, `unsure`) as an estimated useful precision metric.

## Example Output

`reports/semantic_term_clusters.csv` includes:

| cluster_id | canonical_term | observed_term | cluster_reason |
| --- | --- | --- | --- |
| 1 | traffic crash | motor vehicle collision | hybrid |
| 1 | traffic crash | road accident | hybrid |
| 2 | download url | file link | hybrid |
| 3 | keyword | topic | hybrid |

`reports/metadata_heterogeneity_report.md` summarizes record counts, organizations, unique tags, normalized terms, duplicate terms, semantic clusters, examples, limitations, and next steps.

## Research Use

This project can support exploratory research into:

- vocabulary variation in public metadata catalogs
- semantic interoperability barriers
- metadata quality and standardization gaps
- DCAT-US / Project Open Data alignment challenges
- evidence generation for ontology, taxonomy, and semantic search work

The outputs are intentionally reviewable CSV and Markdown files so researchers can inspect candidate clusters before making claims.

## NIW Evidence Use

For National Interest Waiver evidence preparation, this tool can help produce early artifacts showing a concrete technical problem, a reproducible method, and preliminary evidence from a nationally significant open government data catalog. The reports should be treated as supporting research material, not legal advice or final proof.

## Limitations

- Metadata may be stale.
- Similarity clusters are candidate evidence, not final proof.
- Human validation is required.
- This MVP does not download raw data.
- CKAN metadata fields vary across agencies and harvest sources.
- TF-IDF and fuzzy matching can miss true synonyms or over-group generic terms.

## Roadmap

- Human validation UI
- DCAT-US validator integration
- Embeddings
- Full Data.gov snapshot support
- Chicago / Cook County / Illinois corpus support
- Paper-ready benchmark dataset

## Development

```bash
pytest
datagov-profiler --help
```

Large generated outputs should stay out of git. The repository keeps only placeholders under `data/raw`, `data/processed`, and `reports`, plus small examples under `examples`.
