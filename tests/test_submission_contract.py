from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_reset_returns_wrapped_observation():
    response = client.post("/reset", json={})
    payload = response.json()

    assert response.status_code == 200
    assert "observation" in payload
    assert "nitrogen" in payload["observation"]


def test_tasks_endpoint_exposes_three_graded_tasks():
    response = client.get("/tasks")
    payload = response.json()

    assert response.status_code == 200
    assert payload["count"] >= 3
    assert [task["id"] for task in payload["tasks"]] == ["easy", "medium", "hard"]
    assert all(task["grader"].startswith("/grade/") for task in payload["tasks"])


def test_grade_endpoints_return_normalized_scores_and_rewards():
    for task_id in ("easy", "medium", "hard"):
        response = client.get(f"/grade/{task_id}")
        payload = response.json()

        assert response.status_code == 200
        assert 0.0 <= payload["score"] <= 1.0
        assert payload["reward"] == payload["score"]
