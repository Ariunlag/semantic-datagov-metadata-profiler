from datagov_profiler.normalize import normalize_text, split_compound_terms


def test_normalize_text_lowercases_ascii_and_punctuation() -> None:
    assert normalize_text("Traffic-Crash Records!", remove_stopwords=True) == "traffic crash records"


def test_normalize_maps_seed_synonyms() -> None:
    assert normalize_text("Motor Vehicle Collision") == "traffic crash"
    assert normalize_text("CSV URL") == "download url"
    assert normalize_text("Topic") == "keyword"


def test_split_compound_terms() -> None:
    assert split_compound_terms("Traffic Crash; Safety | Roadway") == ["traffic crash", "safety", "roadway"]
