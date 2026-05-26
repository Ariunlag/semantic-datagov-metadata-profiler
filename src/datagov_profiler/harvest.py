from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from datagov_profiler.fetch import fetch_url_payload
from datagov_profiler.flatten import read_json_or_jsonl


def fetch_harvest_records(
    input_path: Path,
    out_dir: Path,
    *,
    index_out: Path,
    limit: int,
    sleep_seconds: float,
    transformed: bool = False,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in read_json_or_jsonl(input_path)[:limit]:
        dataset_id = str(record.get("id") or record.get("identifier") or "")
        title = str(record.get("title") or record.get("dcat", {}).get("title") or "")
        url = transformed_url(record) if transformed else raw_url(record)
        row = {
            "dataset_id": dataset_id,
            "title": title,
            "harvest_record_raw_url": url,
            "content_type": "",
            "local_file": "",
            "parse_status": "skipped",
            "error": "",
        }
        if not url:
            row["error"] = "no transformed harvest URL found" if transformed else "no harvest_record_raw URL found"
            rows.append(row)
            continue
        try:
            content, content_type = fetch_url_payload(url)
            suffix, parse_status = classify_payload(content, content_type)
            local_file = out_dir / f"{safe_name(dataset_id or title or 'record')}{suffix}"
            local_file.write_bytes(content)
            row.update(
                {
                    "content_type": content_type,
                    "local_file": str(local_file),
                    "parse_status": parse_status,
                }
            )
        except Exception as exc:  # noqa: BLE001 - index should record per-record failures.
            row["parse_status"] = "error"
            row["error"] = str(exc)
        rows.append(row)
        time.sleep(sleep_seconds)
    frame = pd.DataFrame(rows)
    index_out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(index_out, index=False)
    return frame


def raw_url(record: dict[str, Any]) -> str:
    value = record.get("harvest_record_raw") or record.get("harvest_record_raw_url")
    return str(value or "")


def transformed_url(record: dict[str, Any]) -> str:
    for key in ("harvest_record", "harvest_record_transformed", "harvest_record_transformed_url"):
        if record.get(key):
            return str(record[key])
    raw = raw_url(record)
    if not raw:
        return ""
    return raw.replace("/raw", "/transformed").replace("raw=true", "transformed=true")


def classify_payload(content: bytes, content_type: str) -> tuple[str, str]:
    text = content[:4096].decode("utf-8", errors="ignore").strip()
    lowered = content_type.lower()
    if "json" in lowered or text.startswith("{") or text.startswith("["):
        try:
            data = json.loads(content.decode("utf-8"))
            if isinstance(data, dict) and data.get("@type") == "dcat:Dataset":
                return ".json", "dcat_json"
            return ".json", "json"
        except ValueError:
            return ".txt", "text"
    if "xml" in lowered or text.startswith("<"):
        return ".xml", "xml"
    return ".txt", "text"


def safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return clean[:80] or "record"
