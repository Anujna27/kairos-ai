from pathlib import Path

from fastapi.testclient import TestClient

import modules.community_api as community_api


def test_community_api_requires_firebase_token(tmp_path: Path) -> None:
    client = TestClient(community_api.build_community_api(tmp_path / "community.db"))
    response = client.get("/questions")
    assert response.status_code == 401


def test_community_api_accepts_verified_user_and_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(community_api, "verify_id_token", lambda token: {"uid": "user-1"})
    client = TestClient(community_api.build_community_api(tmp_path / "community.db"))
    headers = {"Authorization": "Bearer test-token"}
    created = client.post("/questions", headers=headers, json={"title": "Motor sound", "description": "New tone"})
    assert created.status_code == 200
    question_id = created.json()["question_id"]
    listed = client.get("/questions", headers=headers)
    assert listed.status_code == 200
    assert listed.json()[0]["question_id"] == question_id
