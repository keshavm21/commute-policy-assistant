"""End-to-end through HTTP: JWT, request validation, status codes, admin ingestion."""

import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import LLMError, ScriptedFakeLLM
from app.main import create_app
from tests.conftest import POLICIES_DIR

GOOD = json.dumps({"status": "answered", "answer": "Yes, after 8:00 PM.", "citations": ["C1"]})


@pytest.fixture
def make_client(tmp_path):
    def factory(*llm_steps):
        settings = Settings(var_dir=tmp_path / "var", llm_provider="fake",
                            jwt_secret="api-test-secret-long-enough-for-hs256")
        app = create_app(settings, llm=ScriptedFakeLLM(*(llm_steps or (GOOD,))))
        return TestClient(app)
    return factory


@pytest.fixture
def client(make_client):
    with make_client() as c:  # "with" runs the startup (seeding) and shutdown hooks
        yield c


def login(client, employee_id):
    resp = client.post("/auth/token", json={"employee_id": employee_id})
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def test_ask_requires_a_token(client):
    assert client.post("/ask", json={"question": "Saturday cab?"}).status_code == 401
    bad = {"Authorization": "Bearer not.a.token"}
    assert client.post("/ask", json={"question": "Saturday cab?"}, headers=bad).status_code == 401


def test_client_cannot_choose_the_company(client):
    headers = login(client, "Z1001")
    resp = client.post("/ask", headers=headers,
                       json={"question": "Saturday cab?", "company_id": "meridian"})
    assert resp.status_code == 422
    resp = client.post("/ask?company_id=meridian", headers=headers | {"X-Company-Id": "meridian"},
                       json={"question": "Do women need an escort on late drops?"})
    assert resp.status_code == 200
    assert {c["document"] for c in resp.json()["citations"]} == {
        "Zentrix Software Employee Transport Policy"}


def test_answer_shape(client):
    resp = client.post("/ask", headers=login(client, "M2001"),
                       json={"question": "Do women need an escort on late drops?"})
    body = resp.json()
    assert resp.status_code == 200 and body["status"] == "answered"
    assert body["citations"][0]["document"] == "Meridian Health Services Commute Policy"
    assert set(body["citations"][0]) == {"document", "version", "effective_date", "section",
                                         "heading", "excerpt"}
    assert "trace" not in body  # internals never leak to the client


@pytest.mark.parametrize("question", ["", "   ", "x" * 501, "\x00\x01"])
def test_bad_questions_are_rejected(client, question):
    resp = client.post("/ask", headers=login(client, "Z1001"), json={"question": question})
    assert resp.status_code == 422


def test_llm_failure_is_a_503_try_again(make_client):
    with make_client(LLMError("HTTP 429", retryable=False)) as c:
        resp = c.post("/ask", headers=login(c, "K3001"),
                      json={"question": "Can I book a cab on Saturday?"})
    assert resp.status_code == 503
    assert resp.json()["status"] == "try_again" and resp.json()["citations"] == []


def test_unknown_employee_gets_no_token(client):
    assert client.post("/auth/token", json={"employee_id": "nobody"}).status_code == 401


def test_admin_endpoints_need_admin_role(client):
    doc = (POLICIES_DIR / "zentrix_v2.md").read_text()
    resp = client.post("/admin/policies", headers=login(client, "Z1001"), json={"content": doc})
    assert resp.status_code == 403


def test_admin_cannot_upload_for_another_company(client):
    doc = (POLICIES_DIR / "meridian_v1.md").read_text().replace("version: 1", "version: 9")
    resp = client.post("/admin/policies", headers=login(client, "Z9001"), json={"content": doc})
    assert resp.status_code == 400 and "not 'zentrix'" in resp.json()["detail"]


def test_policy_update_through_the_api_switches_answers(client):
    admin = login(client, "Z9001")
    v3 = (POLICIES_DIR / "zentrix_v2.md").read_text() \
        .replace("version: 2", "version: 3").replace("more than 4 km", "more than 2 km")
    resp = client.post("/admin/policies", headers=admin, json={"content": v3})
    assert resp.status_code == 200 and resp.json()["version"] == 3

    versions = client.get("/admin/policies", headers=admin).json()
    assert [(v["version"], v["status"]) for v in versions] == [
        (1, "retired"), (2, "retired"), (3, "active")]

    resp = client.post("/ask", headers=login(client, "Z1001"),
                       json={"question": "How far from the office must I live to get a cab?"})
    assert {c["version"] for c in resp.json()["citations"]} == {3}


def test_invalid_upload_changes_nothing(client):
    admin = login(client, "Z9001")
    resp = client.post("/admin/policies", headers=admin, json={"content": "no front matter"})
    assert resp.status_code == 400
    statuses = [v["status"] for v in client.get("/admin/policies", headers=admin).json()]
    assert statuses == ["retired", "active"]


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"
