from __future__ import annotations

import re
import unicodedata

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

SYNONYM_CANONICALS = {
    "comma separated values": "csv",
    "text csv": "csv",
    "csv file": "csv",
    "rest api": "api",
    "web service": "api",
    "arcgis rest service": "api",
    "download link": "download url",
    "file link": "download url",
    "file url": "download url",
    "csv url": "download url",
    "api endpoint": "access url",
    "endpoint url": "access url",
    "service url": "access url",
    "tag": "keyword",
    "topic": "keyword",
    "subject": "keyword",
    "keywords": "keyword",
    "motor vehicle collision": "traffic crash",
    "vehicle collision": "traffic crash",
    "road accident": "traffic crash",
    "traffic accident": "traffic crash",
}


def normalize_text(value: object, *, remove_stopwords: bool = False) -> str:
    if value is None:
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[_/\\|:;,\.\-\(\)\[\]\{\}\"'`]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if remove_stopwords:
        text = " ".join(token for token in text.split() if token not in STOPWORDS)
    return SYNONYM_CANONICALS.get(text, text)


def split_compound_terms(value: object) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    parts = re.split(r"\s*[;,\|]\s*", str(value))
    terms = [normalize_text(part) for part in parts if normalize_text(part)]
    return terms or [text]


def compact_list(values: list[str], *, limit: int = 5) -> str:
    seen: list[str] = []
    for value in values:
        clean = str(value).strip()
        if clean and clean not in seen:
            seen.append(clean)
        if len(seen) >= limit:
            break
    return " | ".join(seen)
