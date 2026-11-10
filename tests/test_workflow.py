import csv
import io

import pytest
from fastapi.testclient import TestClient

from backend import main as api


@pytest.fixture()
def client(tmp_path, monkeypatch):
    database = tmp_path / "visits.sqlite3"
    monkeypatch.setattr(api, "DB_PATH", database)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    api.init_db()
    with TestClient(api.app) as test_client:
        yield test_client


def record_payload(**overrides):
    fields = {key: None for key in api.FIELD_NAMES}
    fields.update({"person_name": "Asha", "village": "Sample"})
    fields.update(overrides.pop("fields", {}))
    payload = {
        "fields": fields,
        "transcript": "Visited Asha in Sample village.",
        "evidence": {"person_name": "Visited Asha", "village": "Sample village"},
        "confidence": {},
        "validation_messages": [],
        "sources": [],
        "clarification_rounds": 0,
        "processing_mode": "demo",
        "status": "draft",
    }
    payload.update(overrides)
    return payload


def test_fixed_evaluation_corpus_has_forty_cases_and_edges():
    cases = api.evaluation_cases()
    assert len(cases) == 40
    identifiers = {case["id"] for case in cases}
    assert {"empty-note", "contradictory-age", "ambiguous-weekday", "invalid-age", "invalid-weight"} <= identifiers


def test_demo_extractor_preserves_unknowns_conflicts_and_exact_evidence():
    transcript = "Visited Leela in Sample village. Age 28 years old, later age 82 years old. Weight not recorded."
    fields, evidence, _ = api.conservative_demo_extract(transcript, [])
    assert fields.person_name == "Leela"
    assert fields.village == "Sample"
    assert fields.age is None
    assert fields.weight_kg is None
    assert evidence["person_name"] in transcript
    assert evidence["weight_kg"] in transcript


def test_validation_flags_illustrative_ranges_and_bad_dates():
    fields = api.VisitFields(person_name="Asha", village="Sample", age=150, weight_kg=0, follow_up_date="next Monday")
    _, messages = api.validate_fields(fields)
    assert {message["field"] for message in messages if message["severity"] == "error"} == {"age", "weight_kg", "follow_up_date"}
    with pytest.raises(Exception):
        api.VisitFields(follow_up_required="sometimes")


def test_http_extract_reports_demo_and_relevant_references(client, monkeypatch):
    expected_source = {"source": "register-guide.md", "section": "Register field definitions", "text": "Unknown values remain null.", "score": 0.81}
    monkeypatch.setattr(api.kb, "search", lambda query, limit=5: [expected_source])
    response = client.post("/api/visits/extract", json={"transcript": "Visited Asha in Sample village. Age 26 years old."})
    assert response.status_code == 200
    body = response.json()
    assert body["processing_mode"] == "demo"
    assert body["fields"]["age"] == 26
    assert body["sources"] == [expected_source]


def test_two_round_clarification_stays_explicit_and_unresolved(client):
    extraction = api.process_transcript("Visited Asha.").model_dump()
    assert extraction["clarification_field"] == "village"
    first = client.post("/api/visits/clarify", json={"transcript": "Visited Asha.", "extraction": extraction, "field": "village", "answer": "Sample", "clarification_rounds": 0})
    assert first.status_code == 200
    body = first.json()
    assert body["fields"]["village"] == "Sample"
    assert body["status"] == "ready"
    assert body["evidence"]["village"] == "Clarification: Sample"
    # A second unanswered required item exhausts the allowed two-round loop.
    body["missing_fields"] = ["person_name"]
    body["fields"]["person_name"] = None
    body["clarification_field"] = "person_name"
    body["clarification_question"] = "What is the person name?"
    second = client.post("/api/visits/clarify", json={"transcript": "Visited Asha.", "extraction": body, "field": "person_name", "answer": "Unknown", "clarification_rounds": 1})
    assert second.status_code == 200
    assert second.json()["status"] == "unresolved"
    assert second.json()["fields"]["person_name"] is None


def test_records_require_explicit_approval_and_export_real_csv(client):
    created = client.post("/api/visits", json=record_payload(fields={"follow_up_required": True}))
    assert created.status_code == 201
    record_id = created.json()["id"]
    assert created.json()["status"] == "draft"
    assert client.get("/api/visits").json()["total"] == 1
    csv_response = client.get("/api/visits/export/csv")
    assert csv_response.status_code == 200
    row = next(csv.DictReader(io.StringIO(csv_response.text)))
    assert row["person_name"] == "Asha"
    approved = client.post(f"/api/visits/{record_id}/approve")
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"
    assert approved.json()["approved_at"]
    assert client.patch(f"/api/visits/{record_id}", json=record_payload()).status_code == 409


def test_approval_refuses_missing_required_and_bad_values(client):
    payload = record_payload(fields={"village": None, "age": 150})
    record = client.post("/api/visits", json=payload)
    rejected = client.post(f"/api/visits/{record.json()['id']}/approve")
    assert rejected.status_code == 422
    assert rejected.json()["detail"]["missing_fields"] == ["village"]


def test_cancel_only_removes_unapproved_records(client):
    record = client.post("/api/visits", json=record_payload()).json()
    assert client.delete(f"/api/visits/{record['id']}").status_code == 204
    assert client.get(f"/api/visits/{record['id']}").status_code == 404


def test_visit_search_date_filters_and_boolean_schema(client):
    payload = record_payload(fields={"visit_date": "2026-10-27"})
    assert client.post("/api/visits", json=payload).status_code == 201
    assert client.get("/api/visits", params={"search": "Sample"}).json()["total"] == 1
    assert client.get("/api/visits", params={"start_date": "not-a-date"}).status_code == 422
    assert client.post("/api/visits/extract", json={"transcript": 12}).status_code == 422
