from __future__ import annotations

from fastapi.testclient import TestClient

from main import app
import services.flow_executor as flow_executor
import services.prompt_executor as prompt_executor
import api.routes as routes

client = TestClient(app)


def test_list_forms() -> None:
    response = client.get("/forms")

    assert response.status_code == 200
    body = response.json()
    assert any(item["id"] == "customer_summary" for item in body)


def test_read_flow() -> None:
    response = client.get("/flows/client_assessment")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "client_assessment"
    assert body["steps"][0]["prompt_form_id"] == "customer_summary"


def test_execute_flow_saves_context_and_output(monkeypatch) -> None:
    def fake_execute_prompt(**kwargs: object) -> str:
        return "Client recap result"

    monkeypatch.setattr(prompt_executor, "execute_prompt", fake_execute_prompt)
    monkeypatch.setattr(flow_executor, "execute_prompt", fake_execute_prompt)

    response = client.post(
        "/flows/client_assessment/execute",
        json={
            "values": {
                "company": "Acme Corp",
                "industry": "Healthcare",
                "tone": "Professional",
            },
            "context": {
                "client_recap": "Existing context value",
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["context"]["client_recap"] == "Client recap result"
    assert body["steps"][0]["result"] == "Client recap result"
    assert body["steps"][0]["prompt"] == (
        "Summarize Acme Corp in the Healthcare industry using a Professional tone."
    )


def test_execute_form_still_works(monkeypatch) -> None:
    def fake_execute_prompt(**kwargs: object) -> str:
        return "Form result"

    monkeypatch.setattr(prompt_executor, "execute_prompt", fake_execute_prompt)
    monkeypatch.setattr(routes, "execute_prompt", fake_execute_prompt)

    response = client.post(
        "/execute/customer_summary",
        json={
            "values": {
                "company": "Acme Corp",
                "industry": "Healthcare",
                "tone": "Professional",
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result"] == "Form result"
    assert "Acme Corp" in body["prompt"]
