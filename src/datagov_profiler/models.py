from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ResourceSummary(BaseModel):
    name: str | None = None
    description: str | None = None
    format: str | None = None
    mimetype: str | None = None
    media_type: str | None = Field(default=None, alias="mediaType")
    url: str | None = None
    download_url: str | None = Field(default=None, alias="downloadURL")
    access_url: str | None = Field(default=None, alias="accessURL")

    model_config = {"populate_by_name": True, "extra": "allow"}


class MetadataRecord(BaseModel):
    id: str | None = None
    name: str | None = None
    title: str | None = None
    notes: str | None = None
    organization: dict[str, Any] | None = None
    groups: list[dict[str, Any]] = Field(default_factory=list)
    tags: list[dict[str, Any]] = Field(default_factory=list)
    license_title: str | None = None
    metadata_created: str | None = None
    metadata_modified: str | None = None
    url: str | None = None
    extras: list[dict[str, Any]] | dict[str, Any] | None = None
    resources: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"extra": "allow"}
