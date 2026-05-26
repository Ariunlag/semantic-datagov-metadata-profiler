from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests
from rich.console import Console

CATALOG_SEARCH_URL = "https://catalog.data.gov/search"
console = Console()


def fetch_records(
    *,
    rows: int,
    page_size: int,
    query: str,
    out: Path,
    sleep_seconds: float,
) -> int:
    return fetch_catalog_records(rows, page_size, query, out, sleep_seconds)


def fetch_catalog_records(
    rows: int,
    per_page: int,
    query: str,
    out: Path,
    sleep_seconds: float,
) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    after = ""
    with out.open("w", encoding="utf-8") as handle:
        while written < rows:
            current_size = min(per_page, rows - written)
            try:
                payload = request_catalog_page(query=query, per_page=current_size, after=after)
            except RuntimeError as exc:
                console.print(f"[red]Data.gov Catalog API request failed: {exc}[/red]")
                break
            results = extract_catalog_results(payload)
            if not results:
                break
            for result in results[:current_size]:
                record = catalog_output_record(result)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if written >= rows:
                    break
            after_value = payload.get("after")
            if not after_value or not isinstance(after_value, str):
                break
            after = after_value
            time.sleep(sleep_seconds)
    return written


def request_catalog_page(*, query: str, per_page: int, after: str = "", endpoint: str = CATALOG_SEARCH_URL) -> dict[str, Any]:
    params: dict[str, Any] = {"per_page": per_page}
    if query and query != "*:*":
        params["q"] = query
    if after:
        params["after"] = after
    try:
        response = requests.get(endpoint, params=params, timeout=30)
        if response.status_code != 200:
            body = response.text[:500].replace("\n", " ")
            raise RuntimeError(f"HTTP {response.status_code} from {response.url}: {body}")
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(str(exc)) from exc
    except ValueError as exc:
        raise RuntimeError("invalid JSON response") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("catalog response was not an object")
    return payload


def extract_catalog_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("results")
    if candidates is None and isinstance(payload.get("data"), dict):
        candidates = payload["data"].get("results")
    if candidates is None and isinstance(payload.get("result"), dict):
        candidates = payload["result"].get("results")
    if not isinstance(candidates, list):
        return []
    return [item for item in candidates if isinstance(item, dict)]


def catalog_output_record(result: dict[str, Any]) -> dict[str, Any]:
    dcat = result.get("dcat") if isinstance(result.get("dcat"), dict) else {}
    distributions = (
        result.get("distribution")
        or result.get("distributions")
        or result.get("resources")
        or dcat.get("distribution")
        or dcat.get("distributions")
        or []
    )
    record = dict(result)
    record.setdefault("id", result.get("id") or dcat.get("identifier") or result.get("identifier"))
    record.setdefault("title", result.get("title") or dcat.get("title"))
    record.setdefault("description", result.get("description") or dcat.get("description"))
    record.setdefault("identifier", result.get("identifier") or dcat.get("identifier"))
    record.setdefault("keyword", result.get("keyword") or dcat.get("keyword"))
    record.setdefault("publisher", result.get("publisher") or dcat.get("publisher"))
    record.setdefault("dcat", dcat)
    record.setdefault("distributions", distributions)
    record.setdefault("resources", result.get("resources") or distributions)
    record.setdefault("harvest_record", result.get("harvest_record"))
    record.setdefault("harvest_record_raw", result.get("harvest_record_raw"))
    record["raw_result"] = result
    return record


def fetch_url_payload(url: str, *, timeout: int = 30) -> tuple[bytes, str]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content, response.headers.get("content-type", "")
