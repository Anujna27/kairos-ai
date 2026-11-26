"""Authenticated Community API mounted below the FastHTML dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .community_filter import redact_profanity, validate_submission
from .community_store import CommunityAuthorizationError, CommunityStore
from .firebase_core import FirebaseConfigurationError, InvalidFirebaseToken, verify_id_token


class QuestionInput(BaseModel):
    title: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=5000)


class AnswerInput(BaseModel):
    answer_text: str = Field(default="", max_length=5000)


def build_community_api(db_path: str | Path) -> FastAPI:
    api = FastAPI(title="KAIROS Community API", docs_url=None, redoc_url=None)

    def user_from_header(authorization: Annotated[str | None, Header()] = None) -> dict[str, str]:
        token = (authorization or "").removeprefix("Bearer ").strip()
        try:
            return verify_id_token(token)
        except (InvalidFirebaseToken, FirebaseConfigurationError) as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    def store() -> CommunityStore:
        return CommunityStore(db_path)

    @api.get("/questions")
    def list_questions(user: dict[str, str] = Depends(user_from_header)) -> list[dict]:
        del user
        db = store()
        try:
            return db.list_questions()
        finally:
            db.close()

    @api.post("/questions")
    def create_question(payload: QuestionInput, user: dict[str, str] = Depends(user_from_header)) -> dict:
        error = validate_submission(payload.title, "Question title") or validate_submission(payload.description, "Description")
        if error:
            raise HTTPException(status_code=400, detail=error)
        db = store()
        try:
            question_id = db.create_question(user["uid"], redact_profanity(payload.title), redact_profanity(payload.description))
            return {"ok": True, "question_id": question_id}
        finally:
            db.close()

    @api.get("/questions/{question_id}")
    def get_question(question_id: int, user: dict[str, str] = Depends(user_from_header)) -> dict:
        db = store()
        try:
            question = db.get_question(question_id, user["uid"])
            if question is None:
                raise HTTPException(status_code=404, detail="Question does not exist.")
            question["answers"] = db.list_answers(question_id)
            return question
        finally:
            db.close()

    @api.post("/questions/{question_id}/answers")
    def create_answer(question_id: int, payload: AnswerInput, user: dict[str, str] = Depends(user_from_header)) -> dict:
        error = validate_submission(payload.answer_text, "Answer")
        if error:
            raise HTTPException(status_code=400, detail=error)
        db = store()
        try:
            if db.get_question(question_id) is None:
                raise HTTPException(status_code=404, detail="Question does not exist.")
            answer_id = db.create_answer(question_id, user["uid"], redact_profanity(payload.answer_text))
            return {"ok": True, "answer_id": answer_id}
        finally:
            db.close()

    @api.post("/questions/{question_id}/like")
    def toggle_like(question_id: int, user: dict[str, str] = Depends(user_from_header)) -> dict:
        db = store()
        try:
            return {"liked": db.toggle_like(question_id, user["uid"])}
        except (CommunityAuthorizationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            db.close()

    return api
