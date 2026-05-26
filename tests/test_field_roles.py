import pandas as pd

from datagov_profiler.field_roles import infer_field_roles


def test_infer_field_roles_from_names_and_values() -> None:
    frame = pd.DataFrame(
        {
            "incident_id": [1, 2],
            "report_date": ["2024-01-01", "2024-01-02"],
            "latitude": [41.8, 41.9],
            "longitude": [-87.6, -87.7],
            "status": ["Open", "Closed"],
            "percent_resolved": ["50%", "100%"],
            "detail_url": ["https://example.gov/1", "https://example.gov/2"],
            "contact_email": ["data@example.gov", "data@example.gov"],
        }
    )
    roles = {(role.field_name, role.inferred_role) for role in infer_field_roles(frame)}
    assert ("incident_id", "entity_id_field") in roles
    assert ("report_date", "date_field") in roles
    assert ("latitude", "latitude_field") in roles
    assert ("longitude", "longitude_field") in roles
    assert ("status", "status_field") in roles
    assert ("percent_resolved", "percentage_field") in roles
    assert ("detail_url", "url_field") in roles
    assert ("contact_email", "contact_email_field") in roles
