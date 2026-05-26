from __future__ import annotations

import json
from pathlib import Path

from datagov_profiler.fetch import fetch_catalog_records


class FakeResponse:
    def __init__(self, payload: dict[str, object], *, url: str = "https://catalog.data.gov/search") -> None:
        self._payload = payload
        self.status_code = 200
        self.url = url
        self.text = json.dumps(payload)

    def json(self) -> dict[str, object]:
        return self._payload


def test_fetch_catalog_records_uses_cursor_search(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    pages = [
        {"results": [{"id": "one", "title": "One"}], "after": "cursor-1"},
        {"results": [{"id": "two", "title": "Two"}]},
    ]

    def fake_get(url: str, *, params: dict[str, object], timeout: int) -> FakeResponse:
        calls.append((url, params))
        return FakeResponse(pages[len(calls) - 1], url=url)

    monkeypatch.setattr("datagov_profiler.fetch.requests.get", fake_get)
    out = tmp_path / "catalog.jsonl"

    count = fetch_catalog_records(10, 2, "*:*", out, 0)

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert count == 2
    assert [record["id"] for record in records] == ["one", "two"]
    assert all(url == "https://catalog.data.gov/search" for url, _ in calls)
    assert all("/api/3/action/package_search" not in url for url, _ in calls)
    assert "q" not in calls[0][1]
    assert calls[1][1]["after"] == "cursor-1"
